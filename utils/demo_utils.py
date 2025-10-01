import numpy as np
import torch
from scene.cameras import Camera


def get_sand(center=[-0.4, 1.8, -0.1], length=[0.8, 0.04, 0.2], res=[200, 10, 50], noise=0.01):
    init_sand = torch.stack(
                    torch.meshgrid(
                        torch.arange(res[1]),
                        torch.arange(res[2]),
                        torch.arange(res[0])
                    ),
                -1)
    init_sand = init_sand.reshape(-1, 3).float()[:, [2, 0, 1]]
    init_sand /= torch.tensor([[res[0]-1, res[1]-1, res[2]-1]]).float()
    init_sand *= torch.tensor([length])
    init_sand += torch.tensor([center])
    init_sand += torch.randn_like(init_sand) * noise
    init_sand = init_sand.cuda()
    
    n_sandgrid = res[0]*res[1]*res[2]
    volume = length[0]*length[1]*length[2]
    mpm_traditional_vol = (volume / n_sandgrid) * torch.ones(n_sandgrid).float().cuda()
    return init_sand, mpm_traditional_vol

trans_xyz = lambda x, y, z : np.array([
    [1,0,0,x],
    [0,1,0,y],
    [0,0,1,z],
    [0,0,0,1]])

rot_phi = lambda phi : np.array([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]])

rot_theta = lambda th : np.array([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]])

def pose_spherical(theta, phi, x, y, z):
    c2w = trans_xyz(x, y, z)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    return c2w

def get_spherical_cam(cam, num_frames):
    w = cam.image_width
    h = cam.image_height
    k = np.array([[cam.fx, 0., 0.5*w], [0., cam.fy, 0.5*h], [0., 0., 1.]])
    render_c2w = np.stack([pose_spherical(angle, -10.0, 0.0, 1.1, 3.0) for angle in np.linspace(0,360,num_frames+1)[:-1]], 0)
    render_c2w[:, :3, 1:3] *= -1
    render_w2c = np.linalg.inv(render_c2w)
    return [Camera(camera_id="MovingCam", w=w, h=h, k=k, w2c=w2c) for w2c in render_w2c]

def get_extra_attr(chair_model, chair_color, sand_xyz):
    chair_xyz = chair_model["xyz"]
    chair_opacity = chair_model["opacity"]
    chair_rotation = chair_model["rotation"]
    chair_scale = chair_model["scale"]

    n_traditional = sand_xyz.shape[0]
    sand_color = (sand_xyz - sand_xyz.min(dim=0, keepdims=True)[0]) / (sand_xyz.max(dim=0, keepdims=True)[0] - sand_xyz.min(dim=0, keepdims=True)[0])
    sand_color = sand_color * 0.5 + 0.25
    sand_opacity = 1.0 * torch.ones((n_traditional, 1), dtype=torch.float, device="cuda")
    sand_scale = 0.3 * 0.2 / 50 * torch.ones((n_traditional, 3), device="cuda")
    sand_rotation = torch.zeros((n_traditional, 4), device="cuda")
    sand_rotation[:, 0] = 1
    
    extra_attr = [torch.cat([sand_xyz, chair_xyz], 0),
                  torch.cat([sand_color, chair_color], 0),
                  torch.cat([sand_opacity, chair_opacity], 0),
                  torch.cat([sand_scale, chair_scale], 0),
                  torch.cat([sand_rotation, chair_rotation], 0)]
    
    extra_chair = [chair_xyz,
                   chair_color,
                   chair_opacity,
                   chair_scale,
                   chair_rotation]
    
    return extra_attr, extra_chair, sand_color

def prune_faces(gaussians, f_idx_path):
    prune_f_idx = np.load(f_idx_path)
    prune_f_mask = torch.isin(gaussians.binding, torch.from_numpy(prune_f_idx).cuda())
    gaussians._opacity[prune_f_mask] = -100.