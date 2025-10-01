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

import torch
import sys
from datetime import datetime
import numpy as np
import random
from collections import defaultdict
from plyfile import PlyData

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)

def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper


def get_linear_noise_func(
        lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = lr_init * (1 - t) + lr_final * t
        return delay_rate * log_lerp

    return helper

def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert a unit quaternion to a standard form: one in which the real
    part is non negative.

    Args:
        quaternions: Quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    """
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)

def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret

def rot_to_quat_batch(rot: torch.Tensor):
    """
    Implementation based on pytorch3d implementation
    https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html#matrix_to_quaternion
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if rot.size(-1) != 3 or rot.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {rot.shape}.")
    
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        rot.reshape(-1, 9), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)
    out = quat_candidates[
        torch.nn.functional.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(-1, 4)

    return standardize_quaternion(out)

def rotation_activation(verts, faces):
    v1 = verts[faces[:, 0]]
    v2 = verts[faces[:, 1]]
    v3 = verts[faces[:, 2]]

    d1 = v2 - v1
    d2 = v3 - v1
    d3 = d1.cross(d2)
    z_axis = d3 / d3.norm(dim=1, keepdim=True)
    x_axis = d1 / d1.norm(dim=1, keepdim=True)
    y_axis = z_axis.cross(x_axis)
    rotations = torch.stack([x_axis, y_axis, z_axis], dim=2)
    
    quaternions = rot_to_quat_batch(rotations)

    return quaternions

def scaling_activation(verts, faces, scale):
    v1 = verts[faces[:, 0]]
    v2 = verts[faces[:, 1]]
    v3 = verts[faces[:, 2]]

    d1 = v2 - v1
    d2 = v3 - v1

    faces_x = d1.norm(dim=1)
    faces_shear = (d1*d2).sum(dim=1) / (faces_x + 1e-8)
    faces_y = (d2 - (faces_shear/ (faces_x + 1e-8))[:, None] * d1).norm(dim=1)

    scale_exp = torch.exp(scale)
    scaling = torch.stack([scale_exp[:, 0] * faces_x, scale_exp[:, 1] * faces_y, scale_exp[:, 2]], 1)

    return scaling

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

def safe_state(silent):
    old_f = sys.stdout
    class F:
        def __init__(self, silent):
            self.silent = silent

        def write(self, x):
            if not self.silent:
                if x.endswith("\n"):
                    old_f.write(x.replace("\n", " [{}]\n".format(str(datetime.now().strftime("%d/%m %H:%M:%S")))))
                else:
                    old_f.write(x)

        def flush(self):
            old_f.flush()

    sys.stdout = F(silent)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))

def find_adjacent_faces(faces, k=3):
    # Create a dictionary to store faces sharing the same edge
    edge_faces = defaultdict(list)

    # Populate the dictionary with edges and corresponding face indices
    for i, face in enumerate(faces):
        for j in range(3):
            # Create an edge as a tuple of vertex indices
            edge = tuple(sorted([face[j], face[(j + 1) % 3]]))
            edge_faces[edge].append(i)

    # Create a dictionary to store adjacent faces
    adjacent_faces = defaultdict(list)

    # Find adjacent faces by shared edges
    for edge, face_indices in edge_faces.items():
        for face_idx in face_indices:
            face_neighbors = face_indices.copy()
            face_neighbors.remove(face_idx)
            adjacent_faces[face_idx].extend(face_neighbors)

    # Remove each face from its own adjacency list
    for face_index, neighbors in adjacent_faces.items():
        while len(neighbors) != k:
            if len(neighbors) < k:
                adjacent_faces[face_index].append(face_index)
            elif len(neighbors) > k:
                adjacent_faces[face_index].pop()
    adjacent_faces = [adjacent_faces[i] for i in range(len(faces))]
    adjacent_faces = np.array(adjacent_faces)
    return adjacent_faces

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

def read_ply(filename):
    plydata = PlyData.read(filename)
    vertices = np.stack((np.asarray(plydata.elements[0]["x"]),
                      np.asarray(plydata.elements[0]["y"]),
                      np.asarray(plydata.elements[0]["z"])),  axis=1)
    faces = np.stack(plydata.elements[1]['vertex_indices'])
    return vertices, faces

def save_obj(filename, v, f):
    with open(filename, 'w') as fp:
        for vi in v:
            fp.write('v %f %f %f\n' % (vi[0], vi[1], vi[2]))
        for fi in f:
            fp.write('f %d %d %d\n' % (fi[0] + 1, fi[1] + 1, fi[2] + 1))
    fp.close()

def normalize(x):
    norms = torch.norm(x, dim=1, keepdim=True)
    return x / norms

def compute_vertex_normals(vertices, faces):
    normals = torch.zeros_like(vertices)
    triangles = vertices[faces]

    e1 = triangles[:, 0] - triangles[:, 1]
    e2 = triangles[:, 2] - triangles[:, 1]
    n = torch.cross(e1, e2)

    for i in range(faces.shape[1]):
        normals.index_add_(0, faces[:, i], n)

    return normalize(normals)