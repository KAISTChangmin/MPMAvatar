import os
import bpy
from glob import glob
import sys
import argparse

class ArgumentParserForBlender(argparse.ArgumentParser):
    def _get_argv_after_doubledash(self):
        try:
            idx = sys.argv.index("--")
            return sys.argv[idx+1:]
        except ValueError as e:
            return []
    def parse_args(self):
        return super().parse_args(args=self._get_argv_after_doubledash())

parser = ArgumentParserForBlender()
parser.add_argument("--output_path", type=str, default="./output")
parser.add_argument("--ao_res", type=int, default=256)
args = parser.parse_args()

bpy.data.scenes[0].render.engine = "CYCLES"

# Set the device_type
bpy.context.preferences.addons[
    "cycles"
].preferences.compute_device_type = "CUDA" # or "OPENCL"

# Set the device and feature set
bpy.context.scene.cycles.device = "GPU"

# get_devices() to let Blender detects GPU device
bpy.context.preferences.addons["cycles"].preferences.get_devices()
for d in bpy.context.preferences.addons["cycles"].preferences.devices:
    d["use"] = 1 # Using all devices, include GPU and CPU

bpy.context.scene.render.bake.margin = args.ao_res // 256

meshdir = os.path.join(args.output_path, "uvmesh")
aomapdir = os.path.join(args.output_path, "aomap")
os.makedirs(aomapdir, exist_ok=True)

meshfiles = sorted(glob(os.path.join(meshdir, "*.obj")))

for idx, meshfile in enumerate(meshfiles):
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_all()
    bpy.ops.object.delete()

    bpy.ops.wm.obj_import(filepath=meshfile)

    current_mesh = bpy.context.scene.objects[0]
    bpy.context.view_layer.objects.active = current_mesh

    mat = bpy.data.materials.new(name="Material")
    current_mesh.data.materials.append(mat)

    mat = current_mesh.active_material
    mat.use_nodes = True
    matnodes = mat.node_tree.nodes

    bpy.ops.image.new(name="AO", width=args.ao_res, height=args.ao_res)
    image = bpy.data.images['AO']

    tex = matnodes.new('ShaderNodeTexImage')
    img = image
    tex.image = img

    disp = matnodes['Material Output'].inputs[2]
    mat.node_tree.links.new(disp, tex.outputs[0])

    # Bake the lightmap
    bpy.ops.object.bake(type='AO')

    # Save the baked image
    image.filepath_raw = os.path.join(aomapdir, os.path.basename(meshfile).replace("obj", "png"))
    image.file_format = "PNG"
    image.save()

    bpy.data.images.remove(image)