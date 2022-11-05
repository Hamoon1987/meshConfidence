import os
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
import torch
from torchvision.utils import make_grid
import numpy as np
import pyrender
import trimesh
import json
from matplotlib import cm as mpl_cm, colors as mpl_colors



class Renderer:
    """
    Renderer used for visualizing the SMPL model
    Code adapted from https://github.com/vchoutas/smplify-x
    """
    def __init__(self, focal_length=5000, img_res=224, faces=None):
        self.renderer = pyrender.OffscreenRenderer(viewport_width=img_res,
                                       viewport_height=img_res,
                                       point_size=1.0)
        self.focal_length = focal_length
        self.camera_center = [img_res // 2, img_res // 2]
        self.faces = faces

    def visualize_tb(self, vertices, camera_translation, images):
        vertices = vertices.cpu().numpy()
        camera_translation = camera_translation.cpu().numpy()
        images = images.cpu()
        images_np = np.transpose(images.numpy(), (0,2,3,1))
        rend_imgs = []
        for i in range(vertices.shape[0]):
            rend_img = torch.from_numpy(np.transpose(self.__call__(vertices[i], camera_translation[i], images_np[i]), (2,0,1))).float()
            rend_imgs.append(images[i])
            rend_imgs.append(rend_img)
        rend_imgs = make_grid(rend_imgs, nrow=2)
        return rend_imgs

    def __call__(self, vertices, camera_translation, image, error_joint):
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.2,
            alphaMode='OPAQUE',
            baseColorFactor=(0.8, 0.3, 0.3, 1.0))

        camera_translation[0] *= -1.


        # Add segmentation rendering: Assign colore to body parts
        def part_segm_to_vertex_colors(part_segm, n_vertices, alpha=1.0):
            vertex_labels = np.zeros(n_vertices)
            vertex_labels_order = np.zeros(n_vertices)

            # Assign a number to the vertices of each part
            for part_idx, (k, v) in enumerate(part_segm.items()):
                # k is part name, v is vertices of that part
                vertex_labels[v] = part_idx
            
            # Each body part is influence by some joints based on the map
            """ Joint index    joint_names = ['0-Right Ankle','1-Right Knee', '2-Right Hip','3-Left Hip','4-Left Knee','5-Left Ankle','6-Right Wrist','7-Right Elbow',
                                                    '8-Right Shoulder', '9-Left Shoulder', '10-Left Elbow', '11-Left Wrist', '12-Neck', '13-Top of Head']"""
            part_joint_map = {
                0: [6], #rightHand
                1: [1, 2], #rightUpLeg
                2: [9, 10], #leftArm
                3: [4, 5], #leftLeg
                4: [5], #leftToeBase
                5: [5], #leftFoot
                6: [9, 8], #spine1
                7: [9, 8, 12], #spine2
                8: [9], #leftShoulder
                9: [8], #rightShoulder
                10: [0], #rightFoot
                11: [13], #head
                12: [8, 7], #rightArm
                13: [11], #leftHandIndex1
                14: [0, 1], #rightLeg
                15:[6], #rightHandIndex1
                16:[10, 11], #leftForeArm
                17:[6, 7], #rightForeArm
                18:[12], #neck
                19:[0], #rightToeBase
                20:[2, 3], #spine
                21:[3, 4], #leftUpLeg
                22:[11], #leftHand
                23:[2, 3], #hips
            }
            # Calculate each part error based on the joint error
            error_part = np.zeros(len(part_joint_map))
            for key , value in part_joint_map.items():
                for j in value:
                    error_part[key] += error_joint[j]/len(value)
            
            # Assign the error_part to the vertices of each part
            for part_idx, (k, v) in enumerate(part_segm.items()):
                # k is part name, v is vertices of that part
                vertex_labels[v] = error_part[part_idx]
            
            vertex_colors = np.ones((n_vertices, 4))
            vertex_colors[:, 3] = alpha
            cm = mpl_cm.get_cmap('jet')
            norm_gt = mpl_colors.Normalize()
            vertex_colors[:, :3] = cm(norm_gt(vertex_labels))[:, :3]

            return vertex_colors

        # We add the whole body mesh but with different colors for each part
        part_segm = json.load(open('/SPINH/data/smpl_vert_segmentation.json'))
        vertex_colors = part_segm_to_vertex_colors(part_segm, vertices.shape[0])
        mesh = trimesh.Trimesh(vertices, self.faces, vertex_colors=vertex_colors)
        # mesh = trimesh.Trimesh(vertices, self.faces)
        rot = trimesh.transformations.rotation_matrix(
            np.radians(180), [1, 0, 0])
        mesh.apply_transform(rot)
        mesh = pyrender.Mesh.from_trimesh(mesh)
        # mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

        scene = pyrender.Scene(ambient_light=(0.5, 0.5, 0.5))
        scene.add(mesh, 'mesh')

        camera_pose = np.eye(4)
        camera_pose[:3, 3] = camera_translation
        camera = pyrender.IntrinsicsCamera(fx=self.focal_length, fy=self.focal_length,
                                           cx=self.camera_center[0], cy=self.camera_center[1])
        scene.add(camera, pose=camera_pose)


        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1)
        light_pose = np.eye(4)

        light_pose[:3, 3] = np.array([0, -1, 1])
        scene.add(light, pose=light_pose)

        light_pose[:3, 3] = np.array([0, 1, 1])
        scene.add(light, pose=light_pose)

        light_pose[:3, 3] = np.array([1, 1, 2])
        scene.add(light, pose=light_pose)

        color, rend_depth = self.renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        color = color.astype(np.float32) / 255.0
        valid_mask = (rend_depth > 0)[:,:,None]
        output_img = (color[:, :, :3] * valid_mask +
                  (1 - valid_mask) * image)
        return output_img
