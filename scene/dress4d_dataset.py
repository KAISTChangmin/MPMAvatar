import os
import numpy as np
import torch
from torch.utils.data import Dataset
from scene.cameras import Camera
import pickle
from PIL import Image
from tqdm import tqdm

class DRESS4DDataset(Dataset):
    """
    Multiview Video Dataset for Material Parameter Learning
    Output:
        camera, image
    """

    def __init__(
        self,
        data_dir,
        white_bkgd=True,
        downscale_ratio=1.0,
        test_camera_index=[],
        frame_index=[],
        train=True,
        return_type="image",
    ) -> None:
        super().__init__()

        self.data_dir = data_dir
        
        self.white_bkgd = white_bkgd
        self.downscale_ratio = downscale_ratio
        self.test_camera_index = test_camera_index
        self.frame_index = frame_index
        self.train = train
        self.return_type = return_type

        self._load_dataset()

    def _load_dataset(self):
        cameras, scene_radius = self._load_camera_dataset()

        self.scene_radius = scene_radius

        if self.train:
            self.camera_list = [cam for i, cam in enumerate(cameras)]
        else:
            self.camera_list = [cam for i, cam in enumerate(cameras) if i in self.test_camera_index]
        
        if self.return_type == "image":
            self.idx_list = [(camera_idx, frame_idx) for camera_idx in range(len(self.camera_list)) for frame_idx in range(len(self.frame_index))]
            self.rgb_path_list, self.msk_path_list = self._load_image_path_dataset()
        elif self.return_type == "video":
            self.rgb_list, self.msk_list = self._load_image_dataset()
    
    def _load_camera_dataset(self):
        with open(os.path.join(self.data_dir, "Capture/cameras.pkl"), "rb") as cam_pickle:
            cam_data = pickle.load(cam_pickle)
        cameras = []
        cam_centers = []

        for camera_id, cam_info in cam_data.items():
            w2c = cam_info["extrinsics"]
            w2c = np.concatenate([w2c, np.eye(4)[3:]], 0)
            c2w = np.linalg.inv(w2c)
            
            k = cam_info["intrinsics"]
            im = Image.open(os.path.join(self.data_dir, f"Capture/{camera_id}/images/capture-f{self.frame_index[0]:05d}.png"))
            
            w_raw, h_raw = im.size
            w, h = round(w_raw/self.downscale_ratio), round(h_raw/self.downscale_ratio)
            scale_x, scale_y = w/w_raw, h/h_raw
            k[0][0] *= scale_x
            k[0][2] *= scale_x
            k[1][1] *= scale_y
            k[1][2] *= scale_y
            
            cam = Camera(camera_id=camera_id, w=w, h=h, k=k, w2c=w2c, near=1, far=10, data_device="cuda")
            cameras.append(cam)
            cam_centers.append(c2w[:3, 3])

        cam_centers = np.array(cam_centers)
        scene_radius = 1.1 * np.max(np.linalg.norm(cam_centers - np.mean(cam_centers, 0)[None], axis=-1))

        return cameras, scene_radius
    
    def _load_image_path_dataset(self):
        rgb_list = []
        msk_list = []
        split = "train" if self.train else "test"
        for cam in tqdm(self.camera_list, desc=f"Reading {split} images..."):
            camera_id = cam.camera_id
            
            rgb_path_list = []
            msk_path_list = []
            for frame in self.frame_index:
                rgb_path = os.path.join(self.data_dir, f"Capture/{camera_id}/images/capture-f{frame:05d}.png")
                msk_path = os.path.join(self.data_dir, f"Capture/{camera_id}/masks/mask-f{frame:05d}.png")
                rgb_path_list.append(rgb_path)
                msk_path_list.append(msk_path)
            
            rgb_list.append(rgb_path_list)
            msk_list.append(msk_path_list)
        
        return rgb_list, msk_list
    
    def _load_image_dataset(self):
        rgb_list = []
        msk_list = []
        split = "train" if self.train else "test"
        for cam in tqdm(self.camera_list, desc=f"Reading {split} images..."):
            camera_id = cam.camera_id
            W, H = cam.image_width, cam.image_height

            rgb_numpy_list = []
            msk_numpy_list = []
            for frame in self.frame_index:
                rgb_path = os.path.join(self.data_dir, f"Capture/{camera_id}/images/capture-f{frame:05d}.png")
                msk_path = os.path.join(self.data_dir, f"Capture/{camera_id}/masks/mask-f{frame:05d}.png")
                
                rgb_numpy = np.array(Image.open(rgb_path).resize((W, H), Image.BILINEAR))
                msk_numpy = np.array(Image.open(msk_path).resize((W, H), Image.BILINEAR))

                rgb_numpy_list.append(rgb_numpy)
                msk_numpy_list.append(msk_numpy)
            rgb_numpy_list = np.stack(rgb_numpy_list, 0)
            msk_numpy_list = np.stack(msk_numpy_list, 0)
            
            rgb_list.append(rgb_numpy_list)
            msk_list.append(msk_numpy_list)
        
        return rgb_list, msk_list

    def __len__(self):
        if self.return_type == "image":
            return len(self.idx_list)
        elif self.return_type == "video":
            return len(self.camera_list)

    def __getitem__(self, idx):
        if self.return_type == "image":
            camera_idx, frame_idx = self.idx_list[idx]
            
            cam = self.camera_list[camera_idx]
            
            W, H = cam.image_width, cam.image_height

            rgb_path = self.rgb_path_list[camera_idx][frame_idx]
            msk_path = self.msk_path_list[camera_idx][frame_idx]

            rgb_numpy = np.array(Image.open(rgb_path).resize((W, H), Image.BILINEAR)).astype(np.float32) / 255.
            msk_numpy = np.array(Image.open(msk_path).resize((W, H), Image.BILINEAR)).astype(np.float32) / 255.

            rgb = torch.from_numpy(rgb_numpy).permute(2, 0, 1).contiguous()
            msk = torch.from_numpy(msk_numpy).unsqueeze(0).contiguous()
            
            ret = {
                "cam": cam,
                "camera_idx": camera_idx,
                "frame_idx": frame_idx,
                "rgb": rgb, # [3, H, W] value in [0, 1]
                "msk": msk, # [1, H, W] value in [0, 1]
            }
        
        elif self.return_type == "video":
            cam = self.camera_list[idx]

            rgb = torch.from_numpy(self.rgb_list[idx].astype(np.float32)).permute(0, 3, 1, 2).contiguous() / 255.
            msk = torch.from_numpy(self.msk_list[idx].astype(np.float32)).unsqueeze(1).contiguous() / 255.
            
            ret = {
                "cam": cam,
                "camera_idx": idx,
                "rgb": rgb, # [N_frames, 3, H, W] value in [0, 1]
                "msk": msk, # [N_frames, 1, H, W] value in [0, 1]
            }

        return ret

def collate_fn(batch):
    ret = {}
    
    for k in batch[0].keys():
        if k in ["cam", "camera_idx", "frame_idx"]:
            ret[k] = [item[k] for item in batch]
        else:
            ret[k] = torch.stack([item[k] for item in batch], dim=0)

    return ret