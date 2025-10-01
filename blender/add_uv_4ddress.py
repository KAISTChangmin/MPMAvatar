import os
from tqdm import tqdm
from glob import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--uv_path", type=str, required=True)
parser.add_argument("--output_path", type=str, default="./output/exp1_cloth/a1_s1_460_200")

args = parser.parse_args()

uv_list, faces_list = [], []
with open(args.uv_path, "r") as f:
    for line in f:
        if line[:2] == "vt":
            uv_list.append(line)
        elif line[:2] == "f ":
            faces_list.append(line)

savedir = os.path.join(args.output_path, "uvmesh")
os.makedirs(savedir, exist_ok=True)

meshfiles = glob(os.path.join(args.output_path, "mesh_cloth_*.obj"))

for meshfile in tqdm(meshfiles):
    vertices_list = []
    with open(meshfile, "r") as f:
        for line in f:
            if line[:2] == "v ":
                vertices_list.append(line)

    write_obj_path = os.path.join(savedir, os.path.basename(meshfile))
    with open(write_obj_path, "w") as f:
        f.writelines(vertices_list)
        f.writelines(uv_list)
        f.writelines(faces_list)