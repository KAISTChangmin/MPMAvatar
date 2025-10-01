import argparse
import os
import sys
import pickle
import numpy as np
from tqdm import tqdm
from glob import glob
from PIL import Image
import logging

import torch
import torch.nn.functional as F
from accelerate.utils import ProjectConfiguration
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from accelerate import Accelerator, DistributedDataParallelKwargs

from arguments import ModelParams, PipelineParams, OptimizationParams
from scene import Scene, MeshGaussianModel
from gaussian_renderer import render
from utils.general_utils import read_obj
from utils.smplx_deformer import SmplxDeformer
from utils.sh_utils import eval_sh
from utils.demo_utils import get_sand, get_spherical_cam, get_extra_attr, prune_faces
from utils.subprocess_utils import run_subprocess

import warp as wp
from warp_mpm.mpm_data_structure import (
    MPMStateStruct,
    MPMModelStruct,
)
from warp_mpm.mpm_solver import MPMWARP


logger = get_logger(__name__, log_level="INFO")

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


class Trainer:
    def __init__(self, args, opt, pipe):
        self.args = args
        os.environ["WANDB__SERVICE_WAIT"] = "600"

        logging_dir = os.path.join(args.output_dir, opt.save_name)
        accelerator_project_config = ProjectConfiguration(logging_dir=logging_dir)
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        accelerator = Accelerator(
            gradient_accumulation_steps=1,
            mixed_precision="no",
            log_with="wandb",
            project_config=accelerator_project_config,
            kwargs_handlers=[ddp_kwargs],
        )
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        logger.info(accelerator.state, main_process_only=False)

        set_seed(opt.seed + accelerator.process_index)
        print("process index", accelerator.process_index)
        if accelerator.is_main_process:
            output_path = os.path.join(logging_dir, f"seed{opt.seed}")
            os.makedirs(output_path, exist_ok=True)
            self.output_path = output_path

        # setup the dataset
        self.split_idx_path = args.split_idx_path
        self.verts_start_idx = args.verts_start_idx
        self.model_path = args.model_path
        self.lbs_deformer = SmplxDeformer(model_path=os.path.join(args.dataset_dir, "body_models"), gender=args.smplx_gender, num_betas=300, use_pca=False)
        self.load_gaussians(args)
        self.load_smplx()
        self.smplx_f = torch.from_numpy(self.lbs_deformer.smplx_model.faces.astype(int))

        bg_color = [1, 1, 1] if self.scene.white_bkgd else [0, 0, 0]
        self.bg = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        self.pipe = pipe

        train_dataloader = torch.utils.data.DataLoader(
            self.scene.train_dataset,
            batch_size=1,
            shuffle=True,
            drop_last=False,
            num_workers=0,
            collate_fn=self.scene.collate_fn,
        )
        train_dataloader = accelerator.prepare(train_dataloader)
        self.train_dataloader = cycle(train_dataloader)

        test_dataloader = torch.utils.data.DataLoader(
            self.scene.test_dataset,
            batch_size=1,
            shuffle=False,
            drop_last=True,
            num_workers=0,
            collate_fn=self.scene.collate_fn,
        )
        self.test_dataloader = accelerator.prepare(test_dataloader)

        self.accelerator = accelerator

        self.init_D = args.init_D
        self.init_E = args.init_E
        self.init_nu = args.init_nu
        self.init_gamma = args.init_gamma
        self.init_kappa = args.init_kappa

        self.best_params = {k: v for k, v in np.load(os.path.join(self.scene.dataset_dir, "demo/a1_phys_param.npz")).items()}
        self.torch_param = {
            'D': torch.tensor(float(self.best_params["D"]), requires_grad=False),
            'E': torch.tensor(float(self.best_params["E"] / 100.0), requires_grad=False),
            'H': torch.tensor(float(self.best_params["H"]), requires_grad=False),
        }

        self.mesh_friction_coeff = args.mesh_friction_coeff
        self.friction_angle = args.friction_angle
        
        self.setup_simulation(grid_size=250)

    def load_gaussians(self, args):
        split_idx = np.load(self.split_idx_path)

        self.num_joint_v = int(split_idx["num_joint_v"])
        self.num_joint_f = int(split_idx["num_joint_f"])

        self.reordered_cloth_v_idx = torch.tensor(split_idx["reordered_cloth_v_idx"]).int().cuda()
        self.reordered_cloth_f_idx = torch.tensor(split_idx["reordered_cloth_f_idx"]).int().cuda()

        self.reordered_human_v_idx = torch.tensor(split_idx["reordered_human_v_idx"]).int().cuda()
        self.reordered_human_f_idx = torch.tensor(split_idx["reordered_human_f_idx"]).int().cuda()

        self.joint_v_idx = self.reordered_cloth_v_idx[:self.num_joint_v]
        self.joint_f_idx = self.reordered_cloth_f_idx[:self.num_joint_f]

        self.new_cloth_faces = torch.tensor(split_idx["new_cloth_faces"]).int().cuda()
        self.new_human_faces = torch.tensor(split_idx["new_human_faces"]).int().cuda()
        
        self.gaussians = MeshGaussianModel(args.sh_degree, device="cuda")
        self.scene = Scene(args, self.gaussians, return_type="video", device="cuda", load_timestep=-1)

        verts = self.gaussians.verts_orig + self.gaussians.verts_offset
        self.first_frame_verts = verts[0]

        self.cloth_opacity = self.gaussians.get_opacity[self.reordered_cloth_f_idx]
        self.human_opacity = self.gaussians.get_opacity[self.reordered_human_f_idx]

        self.cloth_scale = self.gaussians.get_scaling[self.reordered_cloth_f_idx]
        self.human_scale = self.gaussians.get_scaling[self.reordered_human_f_idx]

        self.cloth_shs = self.gaussians.get_features[self.reordered_cloth_f_idx]
        self.human_shs = self.gaussians.get_features[self.reordered_human_f_idx]

        return
    
    def load_smplx(self):
        smplx_param_path = os.path.join(self.scene.dataset_dir, f"a{self.scene.actor}_s{self.scene.sequence}", "smplx_fitted", f"{self.verts_start_idx:06d}.pth")
        if not os.path.exists(smplx_param_path):
            smplx_param_path = os.path.join(self.scene.dataset_dir, f"a{self.scene.actor}_s{self.scene.sequence}", "smplx_fitted", f"{self.verts_start_idx:06d}", "smplx_icp_param.pth")
            smplx_param_first = {k: torch.from_numpy(v).cuda() for k, v in torch.load(smplx_param_path).items()}
        else:
            smplx_param_first = {k: v.cuda() for k, v in torch.load(smplx_param_path).items()}
        smplx_out_first = self.lbs_deformer.smplx_forward(smplx_param_first)
        
        lbs_w = np.load(os.path.join(self.scene.dataset_dir, f"a{self.scene.actor}_s{self.scene.sequence}", f"{self.args.lbs_w}.npy"))
        lbs_w = torch.tensor(lbs_w).float().cuda()

        test_smplx_params_dict_numpy = np.load(os.path.join(self.scene.dataset_dir, "demo/a1_sitting.npz"))
        self.scene.test_frame_num = test_smplx_params_dict_numpy["trans"].shape[0]
        test_smplx_params_dict = {k: torch.from_numpy(v).cuda() for k, v in test_smplx_params_dict_numpy.items()}
        test_smplx_out = self.lbs_deformer.smplx_forward(test_smplx_params_dict)

        t_verts_v, _, lbs_w = self.lbs_deformer.transform_to_t_pose(
             self.first_frame_verts.unsqueeze(0),
             smplx_out_first,
             smplx_param_first['trans'],
             smplx_param_first['scale'],
             lbs_w=lbs_w.unsqueeze(0)
        )

        deformed_verts, _ = self.lbs_deformer.transform_to_pose(
            t_verts_v[0].repeat(self.scene.test_frame_num, 1, 1),
            lbs_w.repeat(self.scene.test_frame_num, 1, 1),
            test_smplx_out,
            test_smplx_params_dict['trans'],
            test_smplx_params_dict['scale']
        )
        self.test_frame_smplx = test_smplx_out["vertices"].detach()
        self.test_frame_smplx_velo = (self.test_frame_smplx[1:] - self.test_frame_smplx[:-1]) * 25
        self.test_frame_verts = deformed_verts.squeeze().detach()
        self.test_frame_verts_velo = (self.test_frame_verts[1:] - self.test_frame_verts[:-1]) * 25.
        
        self.num_frames = self.scene.test_frame_num + 130
        return

    def setup_simulation(self, grid_size=100):

        device = "cuda:{}".format(self.accelerator.process_index)        

        verts_whole = self.first_frame_verts.detach()
        verts = verts_whole[self.reordered_cloth_v_idx]
        faces = self.new_cloth_faces
        
        self.init_sand, mpm_traditional_vol = get_sand()

        min_pos = torch.min(verts, 0)[0]
        max_pos = torch.max(verts, 0)[0]
        max_diff = torch.max(max_pos - min_pos)
        original_mean_pos = (min_pos + max_pos) / 2.0
        max_diff = (2.2 - original_mean_pos[1])
        self.scale = 1.0 / max_diff
        self.shift = torch.tensor([[1.0, 1.0, 1.0]]).float().cuda() - original_mean_pos * self.scale

        self.wld2sim = lambda p: p * self.scale + self.shift
        self.sim2wld = lambda p: (p - self.shift) / self.scale

        n_elements = faces.shape[0]
        n_vertices = verts.shape[0]

        mpm_init_verts = self.wld2sim(verts)
        mpm_init_elts = mpm_init_verts[faces].mean(1)

        mpm_init_traditional = self.wld2sim(self.init_sand)
        mpm_traditional_vol *= (self.scale**3)
        
        n_traditional = mpm_init_traditional.shape[0]
        n_particles = n_traditional + n_elements + n_vertices
        
        mpm_init_pos = torch.cat([mpm_init_elts, mpm_init_traditional, mpm_init_verts], 0)

        mpm_init_dir, mpm_rest_dir, mpm_element_vol, mpm_vertex_vol = self.compute_dir_vol(mpm_init_verts, faces, thickness=1e-5)
        mpm_init_dir_inv = torch.linalg.inv(mpm_init_dir)
        mpm_rest_dir = mpm_rest_dir
        mpm_rest_dir_inv = self.compute_rest_dir_inv(mpm_rest_dir)

        mpm_init_vol = torch.cat([mpm_element_vol, mpm_traditional_vol, mpm_vertex_vol], 0)
        mpm_init_cov = torch.zeros((n_particles - n_vertices, 6))
        
        mpm_init_velo = torch.zeros_like(mpm_init_pos)

        wp.init()
        # wp.config.mode = "debug"
        # wp.config.verify_cuda = True

        mpm_state = MPMStateStruct()
        mpm_state.init(n_particles, n_elements, n_vertices, device=device, requires_grad=True)

        self.particle_init_position = mpm_init_pos.clone()
        self.particle_init_dir = mpm_init_dir.clone()
        self.particle_init_velo = mpm_init_velo.clone()
        self.particle_init_rest_dir = mpm_rest_dir.clone()

        self.vertices_init_position = self.wld2sim(self.first_frame_verts[self.reordered_cloth_v_idx]).clone()

        self.n_particles = n_particles
        self.n_elements = n_elements
        self.n_vertices = n_vertices
        self.n_traditional = n_traditional
        
        particle_traditional = np.zeros((self.n_particles,), dtype=np.int32)
        particle_traditional[n_elements:n_elements+n_traditional] = 1
        particle_vertices = np.zeros((self.n_particles,), dtype=np.int32)
        particle_vertices[n_elements+n_traditional:] = 1
        particle_elements = np.zeros((self.n_particles,), dtype=np.int32)
        particle_elements[:n_elements] = 1

        mpm_state.from_torch(
            self.particle_init_position.clone(),
            mpm_init_vol.to(device).clone(),
            mpm_init_dir_inv.to(device).clone(),
            mpm_rest_dir_inv.to(device).clone(),
            faces.to(device).clone(),
            particle_traditional,
            particle_vertices,
            particle_elements,
            mpm_init_cov.to(device).clone(),
            device=device,
            requires_grad=True,
            n_grid=grid_size,
            grid_lim=2.0,
        )
        mpm_model = MPMModelStruct()
        mpm_model.init(n_particles, device=device, requires_grad=True)
        mpm_model.init_other_params(n_grid=grid_size, grid_lim=2.0, device=device)

        material_params = {
            "material": "sand",
            "g": [0.0, -9.8, 0.0],
            "density": 1.0, # kg / m^3
            "grid_v_damping_scale": 1.1,
            "friction_angle": self.friction_angle,
        }

        self.v_damping = material_params["grid_v_damping_scale"]
        self.material_name = material_params["material"]

        mesh_vertices = self.wld2sim(self.test_frame_smplx[0]).cpu().numpy()
        mesh_faces = self.smplx_f.cpu().numpy()

        human_vertices = self.wld2sim(self.first_frame_verts[self.reordered_human_v_idx]).cpu().numpy()
        human_faces = self.new_human_faces.cpu().numpy() + mesh_vertices.shape[0]
        mesh_vertices = np.concatenate([mesh_vertices, human_vertices], 0)
        mesh_faces = np.concatenate([mesh_faces, human_faces], 0)

        cv, cf = read_obj(os.path.join(self.scene.dataset_dir, "demo/chair.obj"))

        self.chair_vertices = self.wld2sim(torch.tensor(cv).cuda())
        chair_faces = cf + mesh_vertices.shape[0]
        mesh_vertices = np.concatenate([mesh_vertices, self.chair_vertices.cpu().numpy()], 0)
        mesh_faces = np.concatenate([mesh_faces, chair_faces], 0)

        mpm_solver = MPMWARP(
            n_particles, n_elements, n_vertices, n_grid=grid_size, grid_lim=2.0, mesh_vertices=mesh_vertices, mesh_faces=mesh_faces, num_joint_t=0, num_joint_v=self.num_joint_v, num_joint_f=self.num_joint_f, device=device
        )
        mpm_solver.set_parameters_dict(mpm_model, mpm_state, material_params)

        self.mpm_state, self.mpm_model, self.mpm_solver = (
            mpm_state,
            mpm_model,
            mpm_solver,
        )

        density = (
            torch.ones_like(self.particle_init_position[..., 0])
            * self.torch_param['D']
        )
        youngs_modulus = (
            torch.ones_like(self.particle_init_position[..., 0])
            * self.torch_param['E'] * 100
        )

        poisson_ratio = torch.ones_like(self.particle_init_position[..., 0]) * self.init_nu
        gamma = torch.ones_like(self.particle_init_position[..., 0]) * self.init_gamma
        kappa = torch.ones_like(self.particle_init_position[..., 0]) * self.init_kappa

        self.density = density
        self.young_modulus = youngs_modulus
        self.poisson_ratio = poisson_ratio
        self.gamma = gamma
        self.kappa = kappa

        # set density, youngs, poisson
        mpm_state.reset_density(
            density.clone(),
            torch.ones_like(density).type(torch.int),
            device,
            update_mass=True,
        )
        mpm_solver.set_E_nu_from_torch(
            mpm_model, youngs_modulus.clone(), poisson_ratio.clone(), gamma.clone(), kappa.clone(), device
        )
        mpm_solver.prepare_mu_lam(mpm_model, mpm_state, device)

        mpm_solver.add_surface_collider([0., 0.1, 0.], [0., 1., 0.])
        mpm_solver.add_mesh_collider(mpm_solver.mesh.id, n_grid=mpm_model.n_grid, friction=self.mesh_friction_coeff)
        mpm_solver.add_particle_mover(n_grid=mpm_model.n_grid)

    def compute_rest_dir_inv(self, rest_dir):
        R11 = rest_dir[:, 0]
        R12 = rest_dir[:, 1]
        R22 = rest_dir[:, 2]
        iR11 = 1.0 / R11
        iR22 = 1.0 / R22
        iR12 = - R12 * iR11 * iR22
        return torch.stack([iR11, iR12, iR22], -1)

    def compute_rest_dir_inv_from_vf(self, vertices, faces):
        # Compute Initial Direction Matrices
        d1 = vertices[faces[:,1]] - vertices[faces[:,0]]
        d2 = vertices[faces[:,2]] - vertices[faces[:,0]]

        # Compute Inverse of Direction Matrices (QR composed)
        R11 = d1.norm(dim=1)
        R12 = (d1*d2).sum(dim=1)/R11
        R22 = (d2 - (R12/R11)[:, None] * d1).norm(dim=1)
        
        iR11 = 1.0 / R11
        iR22 = 1.0 / R22
        iR12 = - R12 * iR11 * iR22
        
        return torch.stack([iR11, iR12, iR22], -1)

    def compute_dir_vol(self, vertices, faces, thickness):
        # Compute Initial Direction Matrices
        d1 = vertices[faces[:,1]] - vertices[faces[:,0]]
        d2 = vertices[faces[:,2]] - vertices[faces[:,0]]
        d3 = d1.cross(d2)
        d3 /= d3.norm(dim=1, keepdim=True)
        init_dir = torch.stack([d1, d2, d3], -1)

        # Compute Inverse of Direction Matrices (QR composed)
        R11 = d1.norm(dim=1)
        R12 = (d1*d2).sum(dim=1)/R11
        R22 = (d2 - (R12/R11)[:, None] * d1).norm(dim=1)
        rest_dir = torch.stack([R11, R12, R22], -1)

        # Compute Particle Volumes
        area = 0.5 * torch.norm(d1.cross(d2), dim=1)
        element_vol = 0.25 * thickness * area
        vertex_vol = torch.zeros(vertices.shape[0]).cuda().to(torch.float32)
        vertex_vol.index_add_(0, faces.reshape(-1), element_vol[:, None].repeat(1, 3).reshape(-1))

        return init_dir, rest_dir, element_vol, vertex_vol
    
    def get_material_params(self, device):
        youngs_modulus = (
            torch.ones_like(self.particle_init_position[..., 0])
            * self.torch_param["E"] * 100
        )

        density = (
            torch.ones_like(self.particle_init_position[..., 0])
            * self.torch_param['D']
        )
        ret_poisson = self.poisson_ratio.detach().clone()
        ret_gamma = self.gamma.detach().clone()
        ret_kappa = self.kappa.detach().clone()

        return density, youngs_modulus, ret_poisson, ret_gamma, ret_kappa
    
    @torch.no_grad()
    def demo(
        self,
        skip_sim=False,
        skip_render=False,
        skip_video=False,
    ):
        if not skip_sim:
            device = "cuda:{}".format(self.accelerator.process_index)

            density, youngs_modulus, poisson, gamma, kappa = self.get_material_params(device)

            init_verts = self.wld2sim(self.test_frame_verts[0, self.reordered_cloth_v_idx].clone())
            init_elts = init_verts[self.new_cloth_faces].mean(1)
            init_traditionals = self.wld2sim(self.init_sand)
            init_xyzs = torch.cat([init_elts, init_traditionals, init_verts], 0)
            init_velocity = torch.zeros_like(init_xyzs)
            particle_d, _, _, _ = self.compute_dir_vol(init_verts, self.new_cloth_faces, thickness=1e-5)
            particle_R_inv = self.compute_rest_dir_inv_from_vf(torch.stack([self.vertices_init_position[:, 0], self.vertices_init_position[:, 1] * self.torch_param['H'], self.vertices_init_position[:, 2]], 1), self.new_cloth_faces)
            
            delta_time = 1.0 / 25
            substep_size = delta_time / self.args.substep
            num_substeps = int(delta_time / substep_size)

            self.mpm_state.reset_state(
                self.n_vertices,
                init_xyzs.clone(),
                particle_d.clone(),
                None,
                init_velocity.clone(),
                tensor_R_inv=particle_R_inv.clone(),
                device=device,
                requires_grad=True,
            )

            density, youngs_modulus, poisson, gamma, kappa = self.get_material_params(device)
            density[self.n_elements:self.n_elements+self.n_traditional] *= 0.1
        
            self.mpm_state.reset_density(
                density.clone(), None, device, update_mass=True
            )
            self.mpm_solver.set_E_nu_from_torch(
                self.mpm_model, youngs_modulus.clone(), poisson.clone(), gamma.clone(), kappa.clone(), device
            )
            self.mpm_solver.prepare_mu_lam(self.mpm_model, self.mpm_state, device)
            
            vt_f, ft = [], []
            with open(self.scene.uv_path, "r") as f:
                for line in f:
                    if line[:2] == "vt":
                        vt_f.append(line)
                    elif line[:2] == "f ":
                        parts = line.strip().split()
                        ft.append([int(parts[1].split("/")[1]), int(parts[2].split("/")[1]), int(parts[3].split("/")[1])])
            vt_f += [f"f {v[0]}/{vt[0]} {v[1]}/{vt[1]} {v[2]}/{vt[2]}\n" for v, vt in zip(self.gaussians.faces.cpu().numpy()+1, ft)]
            
            mesh_dir = os.path.join(self.output_path, "uvmesh")
            os.makedirs(mesh_dir, exist_ok=True)
            with open(os.path.join(mesh_dir, f"{0:03d}.obj"), "w") as f:
                f.writelines([f"v {v[0]} {v[1]} {v[2]}\n" for v in self.test_frame_verts[0].detach().cpu().numpy()])
                f.writelines(vt_f)
            
            sand_dir = os.path.join(self.output_path, "sand")
            os.makedirs(sand_dir, exist_ok=True)
            with open(os.path.join(sand_dir, f"{0:03d}.obj"), "w") as f:
                f.writelines([f"v {v[0]} {v[1]} {v[2]}\n" for v in self.init_sand.detach().cpu().numpy()])
            
            all_verts = [self.test_frame_verts[0].detach()]
            all_xyzs = [self.init_sand.detach()]

            for i in tqdm(range(self.num_frames-1), desc="Simulation progress"):
                idx = i if i < len(self.test_frame_smplx) else -1
                mesh_x = self.wld2sim(self.test_frame_smplx[idx].clone())
                mesh_v = self.test_frame_smplx_velo[i].clone() * self.scale if i < len(self.test_frame_smplx_velo) else torch.zeros_like(self.test_frame_smplx_velo[0])
                human_x = self.wld2sim(self.test_frame_verts[idx, self.reordered_human_v_idx].clone())
                mesh_x = torch.cat([mesh_x, human_x, self.chair_vertices], 0)
                human_v = self.test_frame_verts_velo[i, self.reordered_human_v_idx].clone() * self.scale if i < len(self.test_frame_verts_velo) else torch.zeros_like(self.test_frame_verts_velo[0, self.reordered_human_v_idx])
                mesh_v = torch.cat([mesh_v, human_v, torch.zeros_like(self.chair_vertices)], 0)
                joint_verts_v = self.test_frame_verts_velo[i, self.joint_v_idx].clone() * self.scale if i < len(self.test_frame_verts_velo) else torch.zeros_like(self.test_frame_verts_velo[0, self.joint_v_idx])
                joint_faces_v = joint_verts_v[self.new_cloth_faces[:self.num_joint_f]].mean(1).clone()
                joint_traditional_v = torch.zeros_like(self.init_sand)[:max(self.n_traditional-max(i-100, 0)*1000, 0)]

                for substep_local in range(num_substeps):
                    mesh_x_curr = mesh_x + substep_size * substep_local * mesh_v
                    self.mpm_solver.p2g2p(
                        self.mpm_model, self.mpm_state, substep_size, mesh_x=mesh_x_curr, mesh_v=mesh_v, joint_traditional_v=joint_traditional_v, joint_verts_v=joint_verts_v, joint_faces_v=joint_faces_v, device=device
                    )

                pos = wp.to_torch(self.mpm_state.particle_x).clone()
                cloth_verts = self.sim2wld(pos[self.n_elements+self.n_traditional:])
                human_verts = self.test_frame_verts[i+1, self.reordered_human_v_idx] if i < len(self.test_frame_verts) - 1 else human_verts
                
                verts = torch.zeros_like(self.first_frame_verts)
                verts[self.reordered_cloth_v_idx] = cloth_verts
                verts[self.reordered_human_v_idx] = human_verts
                    
                with open(os.path.join(mesh_dir, f"{i+1:03d}.obj"), "w") as f:
                    f.writelines([f"v {v[0]} {v[1]} {v[2]}\n" for v in verts.detach().cpu().numpy()])
                    f.writelines(vt_f)
                all_verts.append(verts)

                xyzs = self.sim2wld(pos[self.n_elements:self.n_elements+self.n_traditional])
                with open(os.path.join(sand_dir, f"{i+1:03d}.obj"), "w") as f:
                    f.writelines([f"v {v[0]} {v[1]} {v[2]}\n" for v in xyzs.detach().cpu().numpy()])
                all_xyzs.append(xyzs)
        
        if not skip_render:
            command_bake = ["blender", "-b", "-P", "blender/bake.py", "--", "--output_path", self.output_path]
            run_subprocess(command_bake, label="AO Map Baking by blender")

            if skip_sim:
                all_verts = []
                for frame in tqdm(range(self.num_frames), desc="Reading simulated mesh vertices..."):
                    verts, _ = read_obj(os.path.join(self.output_path, "uvmesh", f"{frame:03d}.obj"))
                    all_verts.append(torch.from_numpy(verts).float().cuda())
                all_xyzs = []
                for frame in tqdm(range(self.num_frames), desc="Reading simulated sand particles..."):
                    xyzs, _ = read_obj(os.path.join(self.output_path, "sand", f"{frame:03d}.obj"))
                    all_xyzs.append(torch.from_numpy(xyzs).float().cuda())
        
            ao_maps = []
            for filename in tqdm(sorted(glob(os.path.join(self.output_path, "aomap/*.png"))), desc="Reading test frame AO Map..."):
                ao_map = np.array(Image.open(filename).convert("L")).astype(np.float32) / 255.
                ao_maps.append(ao_map)
            ao_maps = torch.from_numpy(np.array(ao_maps)).unsqueeze(1).contiguous().float().cuda()
            
            prune_faces(self.gaussians, os.path.join(self.scene.dataset_dir, "demo/a1_prune_f_idx.npy"))

            ref_cam = self.scene.test_dataset.camera_list[0]
            ref_camera_idx = self.scene.test_camera_index[0]
            
            chair_model = torch.load(os.path.join(self.scene.dataset_dir, "demo/chair_gs.pt"))
            chair_xyz = chair_model["xyz"]
            chair_color = convert_SH(chair_model["shs"], ref_cam, self.gaussians, chair_model["xyz"])
            extra_attr, extra_chair, sand_color = get_extra_attr(chair_model, chair_color, self.init_sand)

            savedir = os.path.join(self.output_path, "video")
            imgdir = os.path.join(savedir, "frames")
            os.makedirs(imgdir, exist_ok=True)
            
            spherical_cam = get_spherical_cam(ref_cam, self.num_frames)

            for i in tqdm(range(self.num_frames), desc="Rendering progress"):
                cam, verts, xyzs = spherical_cam[i], all_verts[i], all_xyzs[i]
                self.gaussians.set_mesh_by_verts(verts)
                shadow_map = self.gaussians.shadow_net(ao_maps[i])["shadow_map"]
                shadow = F.grid_sample(shadow_map, self.gaussians.uv_coord, mode='bilinear', align_corners=False).squeeze()[..., None][self.gaussians.binding]

                chair_color = convert_SH(chair_model["shs"], cam, self.gaussians, chair_xyz)
                extra_chair[1] = chair_color
                extra_attr[1] = torch.cat([sand_color, chair_color], 0)
                colors_precomp = shadow * convert_SH(self.gaussians.get_features, cam, self.gaussians, self.gaussians.get_xyz)

                if i > 100:
                    extra = extra_attr
                    extra[0] = torch.cat([xyzs, chair_xyz], 0)
                else:
                    extra = extra_chair
                    extra[0] = chair_xyz
                
                render_pkg = render(cam, self.gaussians, self.pipe, self.bg, override_color=colors_precomp, extra=extra)
                rendering = render_pkg["render"] * torch.exp(self.gaussians.cam_m[ref_camera_idx])[:, None, None] + self.gaussians.cam_c[ref_camera_idx][:, None, None]
                rendering = rendering * render_pkg["mask"] + (1.0 - render_pkg["mask"])

                img = (torch.clamp(rendering.permute(1, 2, 0), 0., 1.).detach().cpu().numpy() * 255.).astype(np.uint8)
                img_path = os.path.join(imgdir, f"{i:04d}.png")
                Image.fromarray(img).save(img_path)

            if not skip_video:
                os.system(f"ffmpeg -y -hide_banner -loglevel error -i {imgdir}/%4d.png -pix_fmt yuv420p -vf scale='trunc(iw/2)*2:trunc(ih/2)*2' {savedir}/video.mp4")

        return
def parse_args():
    parser = argparse.ArgumentParser()
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--skip_sim", action="store_true", default=False)
    parser.add_argument("--skip_render", action="store_true", default=False)
    parser.add_argument("--skip_video", action="store_true", default=False)
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    args = parser.parse_args(sys.argv[1:])

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return lp.extract(args), op.extract(args), pp.extract(args), args.skip_sim, args.skip_render, args.skip_video

if __name__ == "__main__":
    args, opt, pipe, skip_sim, skip_render, skip_video = parse_args()
    trainer = Trainer(args, opt, pipe)
    trainer.demo(skip_sim, skip_render, skip_video)