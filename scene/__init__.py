#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
from utils.system_utils import searchForMaxIteration
from scene.mesh_gaussian_model import MeshGaussianModel
from arguments import ModelParams

class Scene:

    gaussians : MeshGaussianModel

    def __init__(self, args : ModelParams, gaussians : MeshGaussianModel, return_type="image", device="cuda", load_timestep=None):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_timestep = None
        self.gaussians = gaussians

        if load_timestep:
            if load_timestep == -1:
                self.loaded_timestep = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_timestep = load_timestep
            print("Loading trained model at iteration {}".format(self.loaded_timestep))
        
        self.dataset_type = args.dataset_type
        self.dataset_dir = args.dataset_dir
        self.white_bkgd = args.white_bkgd
        self.image_downscale_ratio = args.image_downscale_ratio
        self.test_camera_index = args.test_camera_index
        self.train_frame_start, self.train_frame_num = args.train_frame_start_num
        self.test_frame_start, self.test_frame_num = args.test_frame_start_num
        self.train_frame_index = list(range(self.train_frame_start, self.train_frame_start+self.train_frame_num))
        self.test_frame_index = list(range(self.test_frame_start, self.test_frame_start+self.test_frame_num))
        self.uv_path = args.uv_path

        if self.dataset_type == "actorshq":
            from scene.actorshq_dataset import ActorsHQDataset, collate_fn
            
            self.actor = args.actor
            self.sequence = args.sequence
            
            self.train_dataset = ActorsHQDataset(
                self.dataset_dir,
                self.actor,
                self.sequence,
                white_bkgd=self.white_bkgd,
                downscale_ratio=self.image_downscale_ratio,
                test_camera_index=self.test_camera_index,
                frame_index=self.train_frame_index,
                train=True,
                return_type=return_type
            )
            self.test_dataset = ActorsHQDataset(
                self.dataset_dir,
                self.actor,
                self.sequence,
                white_bkgd=self.white_bkgd,
                downscale_ratio=self.image_downscale_ratio,
                test_camera_index=self.test_camera_index,
                frame_index=self.test_frame_index,
                train=False,
                return_type=return_type
            )
        elif self.dataset_type == "4ddress":
            from scene.dress4d_dataset import DRESS4DDataset, collate_fn

            self.subject = args.subject
            self.train_take = args.train_take
            self.test_take = args.test_take
    
            self.train_dataset = DRESS4DDataset(
                os.path.join(self.dataset_dir, f"4D-DRESS/{self.subject:05d}_Inner/Inner/Take{self.train_take}"),
                white_bkgd=self.white_bkgd,
                downscale_ratio=self.image_downscale_ratio,
                test_camera_index=self.test_camera_index,
                frame_index=self.train_frame_index,
                train=True,
                return_type=return_type
            )
            self.test_dataset = DRESS4DDataset(
                os.path.join(self.dataset_dir, f"4D-DRESS/{self.subject:05d}_Inner/Inner/Take{self.test_take}"),
                white_bkgd=self.white_bkgd,
                downscale_ratio=self.image_downscale_ratio,
                test_camera_index=self.test_camera_index,
                frame_index=self.test_frame_index,
                train=False,
                return_type=return_type
            )
        else:
            raise NotImplementedError(f"Undefined data type: {args.dataset_type}")
        
        self.collate_fn = collate_fn

        self.gaussians.init_from_trained_model(args.trained_model_path, self.train_dataset.scene_radius, self.uv_path, device=device)

        if self.loaded_timestep:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                 "point_cloud",
                                                 "timestep_" + str(self.loaded_timestep).zfill(6),
                                                 "point_cloud.ply"))

    def save(self, timestep):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/timestep_{}".format(timestep))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))