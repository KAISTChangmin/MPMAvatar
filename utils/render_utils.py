import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import numpy as np
import trimesh
import pyrender


def render_mesh(vertices, faces, cam):
    # scene
    bg_color = [0.0, 0.0, 0.0, 0.0]
    scene = pyrender.Scene(bg_color=bg_color, ambient_light=(0.5, 0.5, 0.5))

    # mesh
    mesh = trimesh.Trimesh(vertices, faces, process=False, maintain_order=True)
    mesh = pyrender.Mesh.from_trimesh(mesh, smooth=True)
    scene.add(mesh, 'mesh')
    
    # camera
    w2c = cam.world_view_transform.detach().cpu().numpy().T
    c2w = np.linalg.inv(w2c)
    c2w[:3, 1:3] *= -1
    camera = pyrender.IntrinsicsCamera(fx=cam.fx, fy=cam.fy, cx=cam.cx, cy=cam.cy, znear=1, zfar=10)
    scene.add(camera, pose=c2w)

    # light
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=0.8)
    light_pose = np.eye(4)
    light_pose[:3, 3] = np.array([0, -1, 1])
    scene.add(light, pose=light_pose)
    light_pose[:3, 3] = np.array([0, 1, 1])
    scene.add(light, pose=light_pose)
    light_pose[:3, 3] = np.array([1, 1, 2])
    scene.add(light, pose=light_pose)
    
    # render
    renderer = pyrender.OffscreenRenderer(viewport_width=cam.image_width, viewport_height=cam.image_height, point_size=1.0)
    img, msk = renderer.render(scene, flags=pyrender.RenderFlags.SKIP_CULL_FACES)
    img = np.concatenate([img, 255 * (msk>1).astype(np.uint8)[..., None]], -1)

    return img