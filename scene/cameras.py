import math
import torch
from torch import nn

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

class Camera(nn.Module):
    def __init__(self, camera_id, w, h, k, w2c, near=1, far=10, data_device="cuda"):
        super(Camera, self).__init__()

        self.data_device = torch.device(data_device)

        fx, fy, cx, cy = k[0][0], k[1][1], k[0][2], k[1][2]
        self.fx, self.fy, self.cx, self.cy = fx, fy, cx, cy

        self.camera_id = camera_id
        self.image_width = w
        self.image_height = h
        self.FoVx = focal2fov(fx, w)
        self.FoVy = focal2fov(fy, h)

        self.world_view_transform = torch.tensor(w2c).float().transpose(0, 1).to(self.data_device)
        
        self.projection_matrix = torch.tensor([[2 * fx / w, 0.0, -(w - 2 * cx) / w, 0.0],
                                    [0.0, 2 * fy / h, -(h - 2 * cy) / h, 0.0],
                                    [0.0, 0.0, far / (far - near), -(far * near) / (far - near)],
                                    [0.0, 0.0, 1.0, 0.0]]).float().transpose(0, 1).to(self.data_device)
        
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(
                self.projection_matrix.unsqueeze(0)
            )
        ).squeeze(0)
        
        self.camera_center = self.world_view_transform.inverse()[3, :3]