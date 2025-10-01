import os
import numpy as np
import torch
from lpipsPyTorch import lpips
from utils.image_utils import psnr
from utils.loss_utils import ssim
from utils.general_utils import read_obj
from metric import all_mesh_metrics
from tqdm import tqdm
import trimesh
import argparse
import pickle
import cv2
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--output_path", type=str, required=True)
parser.add_argument("--mesh_path", type=str, required=True)
parser.add_argument("--data_path", type=str, required=True)
parser.add_argument("--start_idx", type=int, default=660)
parser.add_argument("--num_timesteps", type=int, default=200)
parser.add_argument("--actor", type=int, default=1)
parser.add_argument("--dataset", type=str, default="actorshq", choices=["actorshq", "4ddress"])
args = parser.parse_args()

frames = list(range(args.start_idx, args.start_idx+args.num_timesteps))

metrics = {}

# Geometry
geo_metrics = {"CD": [], "F-Score": []}

mv, mf = read_obj(args.mesh_path)
mesh_pred = trimesh.Trimesh(mv, mf, process=False, maintain_order=True)

for idx, frame in enumerate(tqdm(frames, desc="Geometry")):
    vert_pred, _ = read_obj(os.path.join(args.output_path, "uvmesh", f"{idx:03d}.obj"))
    mesh_pred.vertices = vert_pred
    if args.dataset == "actorshq":
        mesh_gt = trimesh.load(os.path.join(args.data_path, f"meshes/Frame{frame:06d}.obj"), process=False, maintain_order=True)
    elif args.dataset == "4ddress":
        with open(os.path.join(args.data_path, f"Meshes_pkl/mesh-f{frame:05d}.pkl"), "rb") as f:
            mesh_pickle = pickle.load(f)
            mesh_gt = trimesh.Trimesh(vertices=mesh_pickle["vertices"], faces=mesh_pickle["faces"], process=False)       

    fscore_value, chamfer_distance = all_mesh_metrics(mesh_gt, mesh_pred)

    geo_metrics["CD"].append(chamfer_distance.item())
    geo_metrics["F-Score"].append(fscore_value.item())

for k, v in geo_metrics.items():
    v_mean = sum(v) / len(v)
    print(k, v_mean)

metrics.update(geo_metrics)
np.savez(os.path.join(args.output_path, "geo_metric.npz"), **geo_metrics)

# Appearance
app_metrics = {"LPIPS": [], "PSNR": [], "SSIM": []}

for frame in tqdm(frames, desc="Appearance"):
    if args.dataset == "actorshq":
        cams = ["Cam007", "Cam127"]
    elif args.dataset == "4ddress":
        cams = ["0004"]

    for cam in cams:
        img_pred = torch.from_numpy(np.array(Image.open(os.path.join(args.output_path, cam, "pred", f"{frame:04d}.png"))).astype(np.float32).transpose(2,0,1)).contiguous().cuda() / 255.
        img_gt = torch.from_numpy(np.array(Image.open(os.path.join(args.output_path, cam, "gt", f"{frame:04d}.png"))).astype(np.float32).transpose(2,0,1)).contiguous().cuda() / 255.
        
        if args.dataset == "actorshq":
            mask_gt = np.array(Image.open(os.path.join(args.data_path, f"masks/{cam}/{cam}_mask{frame:06d}.png"))).astype(np.float32) / 255.
            white_index = torch.where(img_pred.mean(axis=0) > 0.90)
            img_pred[:, white_index[0], white_index[1]] = 0
            white_index = torch.where(img_gt.mean(axis=0) > 0.90)
            img_gt[:, white_index[0], white_index[1]] = 0
        
        elif args.dataset == "4ddress":
            mask_gt = np.array(Image.open(os.path.join(args.data_path, f"Capture/{cam}/masks/mask-f{frame:05d}.png"))).astype(np.float32) / 255.
    
    kernel = np.ones((3, 3), np.uint8)
    mask_gt = cv2.erode(mask_gt, kernel, iterations=5)
    mask_gt = cv2.GaussianBlur(mask_gt, (5, 5), 0)
    msk_gt = torch.tensor(mask_gt, dtype=torch.float32, device="cuda")[None]

    img_pred = (img_pred * msk_gt).unsqueeze(0)
    img_gt = (img_gt * msk_gt).unsqueeze(0)

    app_metrics["LPIPS"].append(lpips(img_pred, img_gt, net_type='vgg').item())
    app_metrics["PSNR"].append(psnr(img_pred, img_gt).item())
    app_metrics["SSIM"].append(ssim(img_pred, img_gt).item())

for k, v in app_metrics.items():
    v_mean = sum(v) / len(v)
    print(k, v_mean)

metrics.update(app_metrics)
np.savez(os.path.join(args.output_path, "app_metric.npz"), **app_metrics)

np.savez(os.path.join(args.output_path, "metric.npz"), **metrics)