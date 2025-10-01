import os
import numpy as np
import torch
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render
from lpipsPyTorch.modules.lpips import LPIPS
import sys
from scene import Scene, MeshGaussianModel
from utils.general_utils import safe_state
from utils.sh_utils import eval_sh
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from time import time

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
import torch.nn.functional as F
from pytorch3d.structures.meshes import Meshes

def cycle(dl: torch.utils.data.DataLoader):
    while True:
        for data in dl:
            yield data

def convert_SH(
    shs_view,
    viewpoint_camera,
    pc: MeshGaussianModel,
    position: torch.tensor,
    rotation: torch.tensor = None,
):
    shs_view = shs_view.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
    dir_pp = position - viewpoint_camera.camera_center.repeat(shs_view.shape[0], 1)
    if rotation is not None:
        dir_pp = torch.matmul(rotation, dir_pp.unsqueeze(2)).squeeze(2)

    dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
    sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
    colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)

    return colors_precomp

def training(dataset, opt, pipe, testing_iterations, saving_iterations):
    tb_writer = prepare_output_and_logger(dataset)

    gaussians = MeshGaussianModel(dataset.sh_degree, device="cuda")
    scene = Scene(dataset, gaussians, return_type="image", device="cuda", load_timestep=None)

    train_dataloader = torch.utils.data.DataLoader(
        scene.train_dataset,
        batch_size=1,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        collate_fn=scene.collate_fn,
    )
    train_dataloader = cycle(train_dataloader)

    test_dataloader = torch.utils.data.DataLoader(
        scene.test_dataset,
        batch_size=1,
        shuffle=False,
        drop_last=True,
        num_workers=0,
        collate_fn=scene.collate_fn,
    )

    laplacian_matrix = Meshes(verts=[gaussians.verts_orig[0]],
                              faces=[gaussians.faces],
                              ).laplacian_packed()
    
    lpips_net = LPIPS("vgg").to("cuda")

    bg_color = [1, 1, 1] if scene.white_bkgd else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    bg = torch.rand((3), device="cuda") if opt.random_background else background

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    loss_weights = {"scale": 1.0, "iso": 20, "normal": 0.1, "eq_faces_weight": 1000, "opacity": 0.05, "area": 50, "scale_ratio": 10, "scale_edge_ratio": 1000, "scale_edge_ratio_var": 1, "scale_max": 1000, "offset": 1., "tv": 1., "laplacian": 5., 'xyz': 1.0}
    
    gaussians.training_setup(opt)

    tb_image_idx = np.linspace(0, scene.test_frame_num-1, 5).astype(np.int32)

    iterations = opt.iterations
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(iterations), desc=f"Training progress")
    
    total_update_spent_time = 0
    update_spent_time = 0
    for iteration in range(1, iterations+1):
        iter_start.record()
        data = next(train_dataloader)
        cam = data["cam"][0]
        camera_idx = data["camera_idx"][0]
        frame_idx = data["frame_idx"][0]
        gt_rgb = data["rgb"][0].cuda()
        gt_msk = data["msk"][0].cuda()

        gt_image = gt_rgb * gt_msk
        if scene.white_bkgd:
            gt_image += (1.0 - gt_msk)

        gaussians.update_learning_rate(iteration)
        
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()
        
        add_offset = True if opt.first_frame_verts_opt else frame_idx != 0
        gaussians.select_mesh_by_timestep(frame_idx, add_offset=add_offset)

        shadow_map = gaussians.shadow_net(gaussians.ao_maps[frame_idx])["shadow_map"]
        shadow = F.grid_sample(shadow_map, gaussians.uv_coord, mode='bilinear', align_corners=False).squeeze()[..., None][gaussians.binding]
        
        colors_precomp = shadow * convert_SH(gaussians.get_features, cam, gaussians, gaussians.get_xyz)
        render_pkg = render(cam, gaussians, pipe, bg, override_color=colors_precomp)

        image = render_pkg["render"] * torch.exp(gaussians.cam_m[camera_idx])[:, None, None] + gaussians.cam_c[camera_idx][:, None, None]
        image = (image * render_pkg["mask"]).clip(0., 1.)

        viewspace_point_tensor, visibility_filter, radii = render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        Ll1 = l1_loss(image, gt_image)
        lpips_loss = lpips_net(image.unsqueeze(0), gt_image.unsqueeze(0))
        img_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)) + opt.lambda_lpips * lpips_loss
        
        reg_losses = {}
        reg_losses["normal"] = gaussians.normal_loss()
        reg_losses["opacity"] = gaussians.opacity_loss()
        reg_losses["iso"] = gaussians.iso_loss()
        reg_losses["eq_faces_weight"] = gaussians.area_loss()
        # if opt.laplacian_type == 0:
        #     reg_losses['laplacian'] = laplacian_matrix.mm(gaussians.verts).norm(dim=1).mean()
        # elif opt.laplacian_type == 1:
        #     loss_weights["laplacian"] = 100.0
        #     reg_losses['laplacian'] = laplacian_matrix.mm(gaussians.verts - gaussians.verts_orig[frame_idx]).norm(dim=1).mean()
        # reg_losses["offset"] = gaussians.verts_offset.norm(dim=-1).mean()
        reg_losses['xyz'] = F.relu(gaussians._xyz[visibility_filter].norm(dim=1) - opt.threshold_xyz).mean()
        reg_losses['scale'] = F.relu(torch.exp(gaussians._scaling[visibility_filter]) - opt.threshold_scale).norm(dim=1).mean()

        reg_loss = sum([loss_weights[k] * v for k, v in reg_losses.items()])
            
        loss = img_loss + reg_loss

        update_start_time = time()
        loss.backward()
        update_spent_time += time() - update_start_time
        total_update_spent_time += update_spent_time

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            
            if iteration == iterations:
                progress_bar.close()

            if tb_writer:
                tb_writer.add_scalar('train_loss/l1_loss', Ll1.item(), iteration)
                tb_writer.add_scalar('train_loss/lpips_loss', lpips_loss.item(), iteration)
                for k, v in reg_losses.items():
                    tb_writer.add_scalar('train_loss/{}_loss'.format(k), v.item(), iteration)
                tb_writer.add_scalar('train_loss/total_loss', loss.item(), iteration)
                tb_writer.add_scalar('iter_time', iter_start.elapsed_time(iter_end), iteration)
                tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
                tb_writer.add_scalar('update_time', update_spent_time, iteration)
                tb_writer.add_scalar('update_time_per_it', total_update_spent_time / iteration, iteration)

            # Report test and samples of training set
            if iteration in testing_iterations:
                torch.cuda.empty_cache()
                
                l1_test_list = []
                psnr_test_list = []
                lpips_test_list = []
                for test_data in tqdm(test_dataloader, desc="Evaluation progress"):
                    test_cam = test_data["cam"][0]
                    test_camera_idx = scene.test_camera_index[test_data["camera_idx"][0]]
                    test_frame_idx = test_data["frame_idx"][0]
                    test_gt_rgb = test_data["rgb"][0].cuda()
                    test_gt_msk = test_data["msk"][0].cuda()
                    test_gt_image = test_gt_rgb * test_gt_msk
                    if scene.white_bkgd:
                        test_gt_image += (1.0 - test_gt_msk)
                    
                    gaussians.select_mesh_by_timestep(test_frame_idx)
                    
                    shadow_map = gaussians.shadow_net(gaussians.ao_maps[test_frame_idx])["shadow_map"]
                    shadow = F.grid_sample(shadow_map, gaussians.uv_coord, mode='bilinear', align_corners=False).squeeze()[..., None][gaussians.binding]
                    colors_precomp = shadow * convert_SH(gaussians.get_features, test_cam, gaussians, gaussians.get_xyz)
                    
                    render_pkg = render(test_cam, gaussians, pipe, bg, override_color=colors_precomp)
                    test_image = render_pkg["render"] * torch.exp(gaussians.cam_m[test_camera_idx])[:, None, None] + gaussians.cam_c[test_camera_idx][:, None, None]
                    test_image = (test_image * render_pkg["mask"]).clip(0., 1.)

                    if tb_writer and test_frame_idx in tb_image_idx:
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(f"view_{test_cam.camera_id}_frame_{test_frame_idx}/ground_truth", test_gt_image.clip(0.,1.)[None], global_step=iteration)
                        
                        tb_writer.add_images(f"view_{test_cam.camera_id}_frame_{test_frame_idx}/render", test_image.clip(0.,1.)[None], global_step=iteration)
                        
                        ao_color = F.grid_sample(gaussians.ao_maps[test_frame_idx].unsqueeze(0), gaussians.uv_coord, mode='bilinear', align_corners=False).squeeze()[..., None].repeat(1, 3)[gaussians.binding]
                        render_pkg = render(test_cam, gaussians, pipe, bg, override_color=ao_color)
                        tb_writer.add_images(f"view_{test_cam.camera_id}_frame_{test_frame_idx}/ao_image", render_pkg["render"].clip(0.,1.)[None], global_step=iteration)
                        
                        colors_precomp = convert_SH(gaussians.get_features, test_cam, gaussians, gaussians.get_xyz)
                        render_pkg = render(test_cam, gaussians, pipe, bg, override_color=colors_precomp)
                        test_image_no_shadow = render_pkg["render"] * torch.exp(gaussians.cam_m[test_camera_idx])[:, None, None] + gaussians.cam_c[test_camera_idx][:, None, None]
                        tb_writer.add_images(f"view_{test_cam.camera_id}_frame_{test_frame_idx}/render_no_shadow", test_image_no_shadow.clip(0.,1.)[None], global_step=iteration)
                        
                        colors_precomp = shadow.repeat(1, 3)
                        render_pkg = render(test_cam, gaussians, pipe, bg, override_color=colors_precomp)
                        test_image_shadow = render_pkg["render"]
                        tb_writer.add_images(f"view_{test_cam.camera_id}_frame_{test_frame_idx}/render_shadow", test_image_shadow.clip(0.,1.)[None], global_step=iteration)
                    
                    l1_test_list.append(l1_loss(test_image, test_gt_image).mean().double().item())
                    psnr_test_list.append(psnr(test_image, test_gt_image).mean().double().item())
                    lpips_test_list.append(lpips_net(test_image.unsqueeze(0), test_gt_image.unsqueeze(0)).mean().double().item())
                l1_test = sum(l1_test_list) / len(l1_test_list)
                psnr_test = sum(psnr_test_list) / len(psnr_test_list)
                lpips_test = sum(lpips_test_list) / len(lpips_test_list)
                print(f"\n[Iteration {iteration}] Evaluating: L1 {l1_test} PSNR {psnr_test} LPIPS {lpips_test}")
                
                if tb_writer:
                    tb_writer.add_scalar('test/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar('test/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar('test/loss_viewpoint - lpips', lpips_test, iteration)
                
                torch.cuda.empty_cache()
            
            update_spent_time = 0
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                update_start_time = time()
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.train_dataset.scene_radius, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (scene.white_bkgd and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()
                update_spent_time += time() - update_start_time

            # Optimizer step
            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none = True)

            if iteration in saving_iterations:
                print(f"\n[Iteration {iteration}] Saving Gaussians")
                point_cloud_path = os.path.join(dataset.model_path, f"point_cloud/timestep_{iteration:06d}")
                gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer
    
if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[1])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[1])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.test_iterations.extend(list(range(5_000, 30_000+1, 5_000)))
    args.save_iterations.extend(list(range(5_000, 30_000+1, 5_000)))
    
    print("Optimizing " + args.trained_model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations)

    # All done
    print("\nTraining complete.")
