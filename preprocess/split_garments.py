import os
import numpy as np
import torch
import argparse

def read_obj(filename):
    vertices = []
    indices = []
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('v '):  # This line describes a vertex
                parts = line.strip().split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith('f '):  # This line describes a face
                parts = line.strip().split()
                face_indices = [int(p.split('/')[0]) - 1 for p in parts[1:]]  # OBJ indices start at 1
                indices.append(face_indices)
        vertices = np.array(vertices, dtype=np.float32)
        indices = np.array(indices, dtype=np.int32)

    return vertices, indices

def split_cloth_human(vertices: torch.Tensor, faces: torch.Tensor, is_cloth_faces: torch.Tensor, filename: str, fix_v: torch.Tensor, iterations: int = 20):

    v_idx = torch.arange(vertices.shape[0]).int().cuda()
    f_idx = torch.arange(faces.shape[0]).int().cuda()

    cloth_f_idx = f_idx[is_cloth_faces]

    if fix_v.shape[0] > 0:
        is_fix_faces = torch.isin(faces, fix_v).any(dim=1)
        fix_f_idx = f_idx[is_fix_faces]

        expanded_fix_f_idx = fix_f_idx.clone()
        for _ in range(iterations):
            expanded_faces = faces[expanded_fix_f_idx]
            expanded_fix_f_idx = torch.argwhere(torch.isin(faces, expanded_faces).sum(dim=1) > 1).squeeze()
        is_expanded_fix_faces = torch.zeros_like(is_fix_faces)
        is_expanded_fix_faces[expanded_fix_f_idx] = True
        
        is_human_faces = torch.logical_or(~is_cloth_faces, is_expanded_fix_faces)
    
    else:
        expanded_cloth_f_idx = cloth_f_idx.clone()
        for _ in range(iterations):
            expanded_faces = faces[expanded_cloth_f_idx]
            expanded_cloth_f_idx = torch.argwhere(torch.isin(faces, expanded_faces).sum(dim=1) > 1).squeeze()
        is_expanded_cloth_faces = torch.zeros_like(is_cloth_faces)
        is_expanded_cloth_faces[expanded_cloth_f_idx] = True
        
        is_human_faces = ~is_cloth_faces
        is_cloth_faces = is_expanded_cloth_faces

    cloth_faces = faces[is_cloth_faces]
    human_faces = faces[is_human_faces]
    
    is_joint_faces = torch.logical_and(is_cloth_faces, is_human_faces)
    
    is_cloth_vertices = torch.isin(v_idx, cloth_faces)
    is_human_vertices = torch.isin(v_idx, human_faces)
    is_human_vertices = torch.logical_or(is_human_vertices, ~is_cloth_vertices)
    is_joint_vertices = torch.logical_and(is_cloth_vertices, is_human_vertices)

    joint_v_idx = v_idx[is_joint_vertices]
    non_joint_cloth_v_idx = v_idx[~is_human_vertices]
    non_joint_human_v_idx = v_idx[~is_cloth_vertices]

    joint_f_idx = f_idx[is_joint_faces]
    non_joint_cloth_f_idx = f_idx[~is_human_faces]
    non_joint_human_f_idx = f_idx[~is_cloth_faces]

    reordered_cloth_v_idx = torch.cat([joint_v_idx, non_joint_cloth_v_idx], 0)
    reordered_human_v_idx = torch.cat([joint_v_idx, non_joint_human_v_idx], 0)

    reordered_cloth_f_idx = torch.cat([joint_f_idx, non_joint_cloth_f_idx], 0)
    reordered_human_f_idx = torch.cat([joint_f_idx, non_joint_human_f_idx], 0)

    cloth_v_mapping = {v.item(): i for i, v in enumerate(reordered_cloth_v_idx)}
    human_v_mapping = {v.item(): i for i, v in enumerate(reordered_human_v_idx)}

    new_cloth_faces = torch.tensor([[cloth_v_mapping[v.item()] for v in faces[f]] for f in reordered_cloth_f_idx]).int().cuda()
    new_human_faces = torch.tensor([[human_v_mapping[v.item()] for v in faces[f]] for f in reordered_human_f_idx]).int().cuda()

    ret = {}
    ret["num_joint_v"] = joint_v_idx.shape[0]
    ret["num_joint_f"] = joint_f_idx.shape[0]
    ret["reordered_cloth_v_idx"] = reordered_cloth_v_idx.cpu().numpy()
    ret["reordered_cloth_f_idx"] = reordered_cloth_f_idx.cpu().numpy()
    ret["reordered_human_v_idx"] = reordered_human_v_idx.cpu().numpy()
    ret["reordered_human_f_idx"] = reordered_human_f_idx.cpu().numpy()
    ret["new_cloth_faces"] = new_cloth_faces.cpu().numpy()
    ret["new_human_faces"] = new_human_faces.cpu().numpy()

    np.savez(filename, **ret)

    return reordered_cloth_v_idx

parser = argparse.ArgumentParser()
parser.add_argument("--mesh_path", type=str, required=True)
parser.add_argument("--cloth_obj", type=str, nargs='*', default=["./data/a1_s1/cloth_sim.obj"])
parser.add_argument("--cloth_npz", type=str, default="None")
parser.add_argument("--cloth_npy", type=str, default="None")
parser.add_argument("--labels", type=int, nargs="+", default=[3])
parser.add_argument("--fix_v", type=str, default="None")
parser.add_argument("--iteration", type=int, default=20)
parser.add_argument("--filename", type=str, default="../data/a1_s1/split_idx.npz")
args = parser.parse_args()

vertices, faces = read_obj(args.mesh_path)
vertices, faces = torch.tensor(vertices).float().cuda(), torch.tensor(faces).int().cuda()

if args.cloth_npz == "None" and args.cloth_npy == "None":
    cloth_f_list = []
    for cloth_obj in args.cloth_obj:
        _, cloth_f = read_obj(cloth_obj)
        cloth_f_list.append(cloth_f)
    cloth_f = np.concatenate(cloth_f_list, 0)
    cloth_f = torch.tensor(cloth_f).int().cuda()
    is_cloth_faces = torch.isin(faces, torch.tensor(cloth_f).int().cuda()).all(dim=1)
elif args.cloth_npz != "None":
    cloth_vertices = np.concatenate([v for k, v in np.load(args.cloth_npz).items() if int(k) in args.labels], 0)
    cloth_vertices = torch.tensor(cloth_vertices).float().cuda()
    is_cloth_faces = torch.isin(faces, cloth_vertices).all(dim=1)
else:
    cloth_vertices = np.load(args.cloth_npy)
    cloth_vertices = torch.tensor(cloth_vertices).float().cuda()
    is_cloth_faces = torch.isin(faces, cloth_vertices).all(dim=1)

if args.fix_v == "None":
    fix_v = torch.empty(0).int().cuda()
else:
    fix_v = torch.from_numpy(np.load(args.fix_v)).int().cuda()

split_cloth_human(vertices, faces, is_cloth_faces, filename=args.filename, fix_v=fix_v, iterations=args.iteration)