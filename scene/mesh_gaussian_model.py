# 
# Toyota Motor Europe NV/SA and its affiliated companies retain all intellectual 
# property and proprietary rights in and to this software and related documentation. 
# Any commercial use, reproduction, disclosure or distribution of this software and 
# related documentation without an express license agreement from Toyota Motor Europe NV/SA 
# is strictly prohibited.
#

import os
from pathlib import Path
import numpy as np
import torch
from scene.gaussian_model import GaussianModel
from utils.graphics_utils import compute_face_orientation
from utils.general_utils import read_obj, inverse_sigmoid, find_adjacent_faces, get_expon_lr_func
from utils.sh_utils import RGB2SH
from scene.shadow import ShadowUNet
from roma import rotmat_to_unitquat, quat_xyzw_to_wxyz
from glob import glob
from tqdm import tqdm
from torch import nn
from PIL import Image

class MeshGaussianModel(GaussianModel):
    def __init__(self, sh_degree : int, device):
        super().__init__(sh_degree)
        self.device = device
        self.shadow_net = None

    def init_from_trained_model(self, trained_model_path, spatial_lr_scale, uv_path, device):
        self.spatial_lr_scale = spatial_lr_scale
        
        sort_key = lambda p: int(p[:-4].split("_")[-1])
        params_file = sorted(glob(os.path.join(trained_model_path, "params_*.npz")), key=sort_key)
        assert len(params_file) > 0
        
        self.num_timesteps = len(params_file)
        
        verts_orig = []
        for idx, param_file in enumerate(tqdm(params_file, desc="Reading params...")):
            params_numpy = dict(np.load(param_file))
            aomap_file = param_file.replace("params_", "aomap/mesh_cloth_").replace(".npz", ".png")
            ao_map = np.array(Image.open(aomap_file).convert("L")).astype(np.float32) / 255.

            if idx == 0:
                # scaling = torch.from_numpy(params_numpy['log_scales'], device=torch.float32, device=device)
                # opacity = torch.from_numpy(params_numpy['logit_opacities'], dtype=torch.float32, device=device)
                cam_m = torch.from_numpy(params_numpy['cam_m']).to(dtype=torch.float32, device=device)
                cam_c = torch.from_numpy(params_numpy['cam_c']).to(dtype=torch.float32, device=device)
                rgb_colors_list = [torch.from_numpy(params_numpy['rgb_colors']).to(dtype=torch.float32, device=device)]

                self.faces = torch.from_numpy(params_numpy['faces']).to(dtype=torch.int32, device=device)
                verts_orig = [torch.from_numpy(params_numpy['vertices']).to(dtype=torch.float32, device=device)]
                ao_maps = [ao_map]
            else:
                rgb_colors_list.append(torch.from_numpy(params_numpy['rgb_colors']).to(dtype=torch.float32, device=device))
                verts_orig.append(torch.from_numpy(params_numpy['vertices']).to(dtype=torch.float32, device=device))
                ao_maps.append(ao_map)
        
        num_faces = len(self.faces)
        self.binding = torch.arange(num_faces, device=device)
        self.binding_counter = torch.ones(num_faces, dtype=torch.int32, device=device)
        
        rgb_color = torch.stack(rgb_colors_list).clip(0, 1).mean(dim=0)
        features = torch.zeros((num_faces, 3, (self.max_sh_degree + 1) ** 2), dtype=torch.float32, device=device)
        features[:, :3, 0 ] = RGB2SH(rgb_color)
        features[:, 3:, 1:] = 0.0

        scales = torch.log(0.1 * torch.ones((num_faces, 3), dtype=torch.float32, device=device))
        rots = torch.zeros((num_faces, 4), dtype=torch.float32, device=device)
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((num_faces, 1), dtype=torch.float, device=device))
        
        self._xyz = nn.Parameter(torch.zeros((num_faces, 3), dtype=torch.float32, device=device, requires_grad=True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((num_faces), device=device)
        
        self.verts_orig = torch.stack(verts_orig, 0)
        self.verts_offset = nn.Parameter(torch.zeros_like(self.verts_orig).requires_grad_(True))
        self.cam_m = nn.Parameter(cam_m.requires_grad_(True))
        self.cam_c = nn.Parameter(cam_c.requires_grad_(True))
        
        face_neighbors = find_adjacent_faces(self.faces.cpu().numpy())
        face_neighbors = torch.from_numpy(face_neighbors).to(dtype=torch.int32, device=device)

        faces_center = self.verts_orig[0, self.faces].mean(dim=1)
        means3D_neighbors = faces_center[face_neighbors]
        neighbor_sq_dist = means3D_neighbors - faces_center.unsqueeze(1)
        neighbor_sq_dist = (neighbor_sq_dist ** 2).sum(-1)

        self.face_neighbors = face_neighbors
        self.neighbor_weight = torch.exp(-2000 * neighbor_sq_dist)
        self.neighbor_dist = torch.sqrt(neighbor_sq_dist)

        self.ao_maps = torch.from_numpy(np.stack(ao_maps, 0)).to(dtype=torch.float32, device=device).unsqueeze(1).contiguous()

        vt_list, face_list = [], []
        with open(uv_path, "r") as f:
            for line in f:
                if line[:2] == "vt":
                    vt_list.append([float(vt) for vt in line[2:].split()])
                elif line[:2] == "f ":
                    face_list.append([int(v_vt.split("/")[1])-1 for v_vt in line[2:].split()])
        uv_coord = torch.from_numpy(np.array(vt_list)[np.array(face_list)].mean(1)).to(dtype=torch.float32, device=device)[None, None] * 2.0 - 1
        uv_coord[..., 1] *= -1
        self.uv_coord = uv_coord

        self.shadow_net = ShadowUNet(
            ao_mean=self.ao_maps.mean(dim=0).cpu().numpy(),
            interp_mode="bilinear",
            biases=False,
            uv_size=256,
            shadow_size=256,
            n_dims=4
        ).to(self.device)

    def select_mesh_by_timestep(self, timestep, add_offset=True):
        self.timestep = timestep
        self.verts = self.verts_orig[self.timestep]
        if add_offset:
             self.verts = self.verts + self.verts_offset[self.timestep]
        triangles = self.verts[self.faces]

        # position
        self.face_center = triangles.mean(dim=-2)

        # orientation and scale
        self.face_orien_mat, self.face_scaling = compute_face_orientation(self.verts, self.faces, return_scale=True)
        self.face_orien_quat = quat_xyzw_to_wxyz(rotmat_to_unitquat(self.face_orien_mat))  # roma
    

    def set_mesh_by_verts(self, verts):
        self.verts = verts
        triangles = self.verts[self.faces]

        # position
        self.face_center = triangles.mean(dim=-2)

        # orientation and scale
        self.face_orien_mat, self.face_scaling = compute_face_orientation(self.verts, self.faces, return_scale=True)
        self.face_orien_quat = quat_xyzw_to_wxyz(rotmat_to_unitquat(self.face_orien_mat))  # roma

    def training_setup(self, training_args):
        super().training_setup(training_args)
        
        param_verts = {'params': [self.verts_offset], 'lr': training_args.verts_lr_init * self.spatial_lr_scale, "name": "verts"}
        self.optimizer.add_param_group(param_verts)
        self.verts_scheduler_args = get_expon_lr_func(lr_init=training_args.verts_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.verts_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.verts_lr_delay_mult,
                                                    max_steps=training_args.verts_lr_max_steps)
        
        param_cams = {'params': [self.cam_m, self.cam_c], 'lr': 1e-4, "name": "cams"}
        self.optimizer.add_param_group(param_cams)

        param_shadow = {'params': self.shadow_net.parameters(), 'lr': 1e-4, "name": "shadow"}
        self.optimizer.add_param_group(param_shadow)
    
    def update_learning_rate(self, iteration):
        super().update_learning_rate(iteration)
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "verts":
                lr = self.verts_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def save_ply(self, path, for_viewer=True):
        super().save_ply(path)
        
        if for_viewer:
            ply_for_viewer_path = Path(path).parent / "point_cloud_viewer.ply"
            super().save_ply(ply_for_viewer_path, for_viewer=True)

        verts_offset_path = Path(path).parent / "verts_offset.npy"
        np.save(str(verts_offset_path), self.verts_offset.detach().cpu().numpy())

        cams_path = Path(path).parent / "cams.npz"
        np.savez(str(cams_path), cam_m=self.cam_m.detach().cpu().numpy(), cam_c=self.cam_c.detach().cpu().numpy())
        
        shadow_net_path = Path(path).parent / "shadow_net.pt"
        torch.save(self.shadow_net.state_dict(), shadow_net_path)
    
    def load_ply(self, path, **kwargs):
        super().load_ply(path, **kwargs)
        
        verts_offset_path = Path(path).parent / "verts_offset.npy"
        self.verts_offset = torch.from_numpy(np.load(str(verts_offset_path))).to(dtype=torch.float32, device=self.device)

        cams_path = Path(path).parent / "cams.npz"
        cams = np.load(str(cams_path))
        self.cam_m = torch.from_numpy(cams["cam_m"]).to(dtype=torch.float32, device=self.device)
        self.cam_c = torch.from_numpy(cams["cam_c"]).to(dtype=torch.float32, device=self.device)

        shadow_net_path = Path(path).parent / "shadow_net.pt"
        shadow_net_weight = torch.load(shadow_net_path)
        self.shadow_net.load_state_dict(shadow_net_weight)
    
    def normal_loss(self):
        vert_faces = self.verts[self.faces]
        v1 = vert_faces[:, 0]
        v2 = vert_faces[:, 1]
        v3 = vert_faces[:, 2]

        d1 = v2 - v1
        d2 = v3 - v1
        d3 = d1.cross(d2)
        
        face_normals = d3 / d3.norm(dim=1, keepdim=True)
        neighbor_normals = face_normals[self.face_neighbors]

        normal_dot = face_normals.unsqueeze(1) * neighbor_normals
        normal_dot = normal_dot.sum(-1)
        norm_mean = normal_dot.mean(-1)
        
        return (norm_mean - 1.0).abs().mean()
    
    def opacity_loss(self):
        return (1.0 - self.get_opacity).mean()
    
    def iso_loss(self):
        xyz = self.verts[self.faces].mean(dim=1)
        neighbor_pts = xyz[self.face_neighbors]
        curr_offset = neighbor_pts - xyz[:, None]
        curr_offset_mag = torch.sqrt((curr_offset ** 2).sum(-1) + 1e-20)
        diff = (curr_offset_mag - self.neighbor_dist) ** 2
        return torch.sqrt(diff * self.neighbor_weight + 1e-20).mean()
    
    def area_loss(self):
        vert_faces = self.verts[self.faces]
        v1 = vert_faces[:, 0]
        v2 = vert_faces[:, 1]
        v3 = vert_faces[:, 2]

        d1 = v2 - v1
        d2 = v3 - v1

        face_area = 0.5 * torch.norm(d1.cross(d2), dim=1)

        mean_area = face_area.mean()
        
        return (face_area - mean_area).abs().mean()