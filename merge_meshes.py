import os
import numpy as np
from glob import glob
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--seq", type=str, required=True)
parser.add_argument("--output_dir", type=str, default="./output/phys")
parser.add_argument("--data_dir", type=str, default="./data")
args = parser.parse_args()

split_idx_upper = np.load(f"{args.data_dir}/{args.seq}/split_idx_upper.npz")

mesh_upper_files = glob(f"./{args.output_dir}/{args.seq}_upper/seed0/uvmesh/*.obj")
mesh_lower_files = glob(f"./{args.output_dir}/{args.seq}_lower/seed0/uvmesh/*.obj")

os.makedirs(f"./{args.output_dir}/{args.seq}/seed0/uvmesh/", exist_ok=True)

for upper_file, lower_file in zip(tqdm(mesh_upper_files, desc="Merging meshes..."), mesh_lower_files):
    upper_v, lower_v, lines = [], [], []
    
    with open(upper_file, 'r') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.strip().split()
                upper_v.append([float(parts[1]), float(parts[2]), float(parts[3])])
            else:
                lines.append(line)

    with open(lower_file, 'r') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.strip().split()
                lower_v.append([float(parts[1]), float(parts[2]), float(parts[3])])
    
    upper_v = np.array(upper_v, dtype=np.float32)
    lower_v = np.array(lower_v, dtype=np.float32)
    cloth_v_idx_upper = split_idx_upper["reordered_cloth_v_idx"]
    lower_v[cloth_v_idx_upper] = upper_v[cloth_v_idx_upper]
    
    with open(lower_file.replace("_lower", ""), 'w') as f:
        f.writelines(['v %f %f %f\n' % (v[0], v[1], v[2]) for v in lower_v])
        f.writelines(lines)