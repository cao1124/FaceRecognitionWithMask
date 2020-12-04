import os
import cv2
import json
import numpy as np
import face_alignment
# from apply_mask_to_faces import mask_generation_utils as utils
import apply_mask_to_faces.mask_generation_utils as utils


def Generate_landmark(fa, ori_image):
    landmarks = fa.get_landmarks_from_image(ori_image)
    landmarks = np.floor(landmarks[0]).astype(np.int32)
    target_points, s1, s2 = utils.extract_target_points_and_characteristic(landmarks)
    mask_rgba_crop, target_points = utils.extract_polygon(ori_image, target_points)
    print(s1, s2)

    mask_rgba_crop, target_points = utils.rotate_image_and_points(mask_rgba_crop, s1, target_points)
    triangles_indexes = utils.get_traingulation_mesh_points_indexes(target_points)

    mask_rgba_crop_vis = mask_rgba_crop[..., :3].copy().astype(np.uint8)
    for triangle in triangles_indexes:
        triangle_points = target_points[triangle]

        mask_rgba_crop_vis = cv2.line(mask_rgba_crop_vis, tuple(triangle_points[0]), tuple(triangle_points[1]), (0, 255, 0), 1)

        mask_rgba_crop_vis = cv2.line(mask_rgba_crop_vis, tuple(triangle_points[1]), tuple(triangle_points[2]), (0, 255, 0), 1)

        mask_rgba_crop_vis = cv2.line(mask_rgba_crop_vis, tuple(triangle_points[0]), tuple(triangle_points[2]), (0, 255, 0), 1)
    return mask_rgba_crop_vis, mask_rgba_crop, target_points, triangles_indexes


def apply_mask(target_image, fa, mask_rgba_crop, target_points, triangles_indexes):
    landmarks2 = fa.get_landmarks_from_image(target_image)
    landmarks2 = np.floor(landmarks2[0]).astype(np.int32)

    target_points2, _, _ = utils.extract_target_points_and_characteristic(landmarks2)

    target_image_with_mask = utils.warp_mask(mask_rgba_crop[..., :3], target_image, target_points, target_points2)

    # triangles_indexes = get_traingulation_mesh_points_indexes(target_points2)

    mask_rgba_crop_vis = target_image_with_mask.copy().astype(np.uint8)
    for triangle in triangles_indexes:
        triangle_points = target_points2[triangle]

        mask_rgba_crop_vis = cv2.line(mask_rgba_crop_vis, tuple(triangle_points[0]), tuple(triangle_points[1]), (0, 255, 0), 3)

        mask_rgba_crop_vis = cv2.line(mask_rgba_crop_vis, tuple(triangle_points[1]), tuple(triangle_points[2]), (0, 255, 0), 3)

        mask_rgba_crop_vis = cv2.line(mask_rgba_crop_vis, tuple(triangle_points[0]), tuple(triangle_points[2]), (0, 255, 0), 3)
    return mask_rgba_crop_vis


def apply_mask_to_face():
    path = 'apply_mask_to_faces/mask_data/nomask/'
    list_images = [x for x in os.listdir(os.path.join(path)) if 'raw' not in x]

    # Generate landmark estimator by face_alignment library
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cpu')
    # mask_rgba_crop_vis, mask_rgba_crop, target_points, triangles_indexes = Generate_landmark(fa, ori_image)
    # apply_mask(target_image, fa, mask_rgba_crop, target_points, triangles_indexes)

    with open('apply_mask_to_faces/mask_data/masks_base.json', 'r') as jf:
        masks_database = json.load(jf)

    for i in range(len(list_images)):
        img_path = os.path.join(path, list_images[i])
        out_path = img_path.split('nomask')[0] + "mask/" + list_images[i].split('2')[0] + '1' + list_images[i].split('2')[1]
        target_image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        target_image_with_mask = utils.end2end_mask_generation(target_image, masks_database, fa)
        cv2.imwrite(out_path, target_image_with_mask)
    print('done')


if __name__ == '__main__':
    apply_mask_to_face()
