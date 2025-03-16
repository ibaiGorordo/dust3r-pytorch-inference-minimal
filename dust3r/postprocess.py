# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# utilities for interpreting the DUST3R output
# --------------------------------------------------------
import torch
import numpy as np
import cv2

from .utils import Output


def parse_output(points: torch.Tensor,
                confidences: torch.Tensor,
                threshold: float = 3.0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    # Convert tensors to numpy arrays
    points = points.cpu().detach().numpy().squeeze()
    confidences = confidences.cpu().detach().numpy().squeeze()

    depth_map = points[..., 2]

    # Apply threshold
    mask = confidences > threshold
    points = points[mask, :]

    return points.reshape(-1, 3), confidences, depth_map, mask

def parse_output_with_color(points: torch.Tensor,
                           confidences: torch.Tensor,
                           img: np.ndarray,
                           threshold: float = 3.0,
                           ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    # Convert tensors to numpy arrays
    points, confidences, depth_map, mask = parse_output(points, confidences, threshold)
    colors = img[mask, :].reshape(-1, 3)

    return points, colors, confidences, depth_map, mask

def estimate_intrinsics(pts3d: np.ndarray,
                        mask: np.ndarray,
                        iterations: int = 10) -> np.ndarray:

    width, height = mask.shape[::-1]
    pixels = np.mgrid[-width//2:width//2, -height//2:height//2].T.astype(np.float32)
    pixels = pixels[mask, :].reshape(-1, 2)
    if len(pixels) == 0:
        return np.zeros((3, 3), dtype=np.float32)

    # Compute normalized image plane coordinates (x/z, y/z)
    xy_over_z = np.divide(pts3d[:, :2], pts3d[:, 2:3], where=pts3d[:, 2:3] != 0)
    xy_over_z[np.isnan(xy_over_z) | np.isinf(xy_over_z)] = 0  # Handle invalid values

    # Initial least squares estimate of focal length
    dot_xy_px = np.sum(xy_over_z * pixels, axis=-1)
    dot_xy_xy = np.sum(xy_over_z**2, axis=-1)
    focal = np.mean(dot_xy_px) / np.mean(dot_xy_xy)

    # Iterative re-weighted least squares refinement
    for _ in range(iterations):
        residuals = np.linalg.norm(pixels - focal * xy_over_z, axis=-1)
        weights = np.reciprocal(np.clip(residuals, 1e-8, None))  # Avoid division by zero
        focal = np.sum(weights * dot_xy_px) / np.sum(weights * dot_xy_xy)

    K = np.array([[focal, 0, width//2],
                  [0, focal, height//2],
                  [0, 0, 1]], dtype=np.float32)

    return K

def estimate_camera_pose(pts3d: np.ndarray,
                         K: np.ndarray,
                         mask: np.ndarray,
                         iterations=100,
                         reprojection_error=5):

    width, height = mask.shape[::-1]
    pixels = np.mgrid[:width, :height].T.astype(np.float32).reshape(-1, 2)
    pixels_valid = pixels[mask.flatten()]

    try:
        # Solve PnP using RANSAC
        success, R_vec, T, inliers = cv2.solvePnPRansac(
            pts3d, pixels_valid, K, None,
            iterationsCount=iterations, reprojectionError=reprojection_error,
            flags=cv2.SOLVEPNP_SQPNP
        )

        if not success:
            raise ValueError("PnP failed")

        # Convert rotation vector to rotation matrix
        R, _ = cv2.Rodrigues(R_vec)  # Converts Rodrigues rotation vector to matrix

        # Construct 4x4 transformation matrix (camera to world)
        pose = np.eye(4)
        pose[:3, :3] = R
        pose[:3, 3] = T.flatten()

        # Invert to get world-to-camera transform
        pose = np.linalg.inv(np.r_[np.c_[R, T], [(0, 0, 0, 1)]])  # cam to world
        # pose = np.linalg.inv(pose)

    except:
        # Return identity matrix if PnP fails
        pose = np.eye(4)

    return pose

def get_transformed_points(points3d: np.ndarray,
                            transform: np.ndarray) -> np.ndarray:

     # Transform points to world coordinates
     points3d = np.c_[points3d, np.ones(points3d.shape[0])]
     points3d = points3d @ np.linalg.inv(transform).T
     points3d = points3d[:, :3] / points3d[:, 3:]

     return points3d

def transform_points(pts3d: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """
    Transform Nx3 points by a 4x4 matrix.
    Returns the transformed Nx3 points.
    """
    ones = np.ones((pts3d.shape[0], 1), dtype=pts3d.dtype)
    pts_h = np.concatenate([pts3d, ones], axis=1)  # Nx4
    pts_trans = pts_h @ transform.T  # Nx4
    # convert back to 3D
    return pts_trans[:, :3] / pts_trans[:, 3:].clip(min=1e-8)

def invert_pose(pose: np.ndarray) -> np.ndarray:
    """Returns pose^-1 if pose is a 4x4 transform."""
    return np.linalg.inv(pose)

def get_transformed_depth(points3d: np.ndarray,
                          mask: np.ndarray,
                          transform: np.ndarray) -> np.ndarray:

    # Transform points to world coordinates
    points3d = get_transformed_points(points3d, transform)

    # Fill the depth map with transformed points
    depth_map = np.zeros_like(mask, dtype=np.float32)
    depth_map[mask] = points3d[:, 2]

    return depth_map

def postprocess(frame1: np.ndarray,
                 pt1: torch.Tensor,
                 cf1: torch.Tensor,
                 frame2: np.ndarray,
                 pt2: torch.Tensor,
                 cf2: torch.Tensor,
                 conf_threshold: float = 3.0,
                 width: int = 512,
                 height: int = 512,
                 ) -> tuple[Output, Output]:

    pts1, colors1, conf_map1, depth_map1, mask1 = parse_output_with_color(pt1, cf1, frame1, threshold=conf_threshold)
    pts2, colors2, conf_map2, depth_map2, mask2 = parse_output_with_color(pt2, cf2, frame2, threshold=conf_threshold)

    # Estimate intrinsics
    intrinsics1 = estimate_intrinsics(pts1, mask1)
    intrinsics2 = intrinsics1 # estimate_intrinsics(pts2, mask2)

    # Estimate camera pose (the first one is the origin)
    cam_pose1 = np.eye(4)
    cam_pose2 = estimate_camera_pose(pts2, intrinsics1, mask2)

    depth_map2 = get_transformed_depth(pts2, mask2, cam_pose2)

    output1 = Output(frame1, pts1, colors1, conf_map1, depth_map1, intrinsics1, cam_pose1, width, height)
    output2 = Output(frame2, pts2, colors2, conf_map2, depth_map2, intrinsics2, cam_pose2, width, height)

    return output1, output2

def postprocess_symmetric(frame1: np.ndarray,
                          pt1_1: torch.Tensor,
                          cf1_1: torch.Tensor,
                          pt1_2: torch.Tensor,
                          cf1_2: torch.Tensor,
                          frame2: np.ndarray,
                          pt2_1: torch.Tensor,
                          cf2_1: torch.Tensor,
                          pt2_2: torch.Tensor,
                          cf2_2: torch.Tensor,
                          conf_threshold: float = 3.0,
                          width: int = 512,
                          height: int = 512,
                          ) -> tuple[Output, Output]:

    pts1, colors1, conf_map1, depth_map1, mask1_1 = parse_output_with_color(pt1_1, cf1_1, frame1, threshold=conf_threshold)
    pts1_2, colors1_2, conf_map1_2, depth_map1_2, mask1_2 = parse_output_with_color(pt1_2, cf1_2, frame1, threshold=conf_threshold)
    pts2_1, colors2_1, conf_map2_1, depth_map2_1, mask2_1 = parse_output_with_color(pt2_1, cf2_1, frame2, threshold=conf_threshold)
    pts2, colors2, conf_map2, depth_map2, mask2_2 = parse_output_with_color(pt2_2, cf2_2, frame2, threshold=conf_threshold)

    # Estimate intrinsics
    intrinsics1 = estimate_intrinsics(pts1, mask1_1)
    intrinsics2 = estimate_intrinsics(pts2, mask2_2)

    conf1 = conf_map1.mean() * conf_map1_2.mean()
    conf2 = conf_map2_1.mean() * conf_map2.mean()

    # Always use the first frame as the origin
    cam_pose1 = np.eye(4)
    if conf1 > conf2:
        # Use i,j info
        cam_pose2 = estimate_camera_pose(pts2_1, intrinsics2, mask2_1)
        depth_map2 = get_transformed_depth(pts2_1, mask2_1, cam_pose2)
        conf_map2 = conf_map2_1
        colors2 = colors2_1
        pts2 = pts2_1
    else:
        # Use j,i info
        cam_pose1_to_2 = estimate_camera_pose(pts1_2, intrinsics1, mask1_2)
        cam_pose2 = np.linalg.inv(cam_pose1_to_2)

        pts1 = get_transformed_points(pts1_2, cam_pose1_to_2)
        pts2 = get_transformed_points(pts2, cam_pose1_to_2)
        colors1 = colors1_2
        conf_map1 = conf_map1_2
        depth_map1 = get_transformed_depth(pts1_2, mask1_2, cam_pose1_to_2)

    output1 = Output(frame1, pts1, colors1, conf_map1, depth_map1, intrinsics1, cam_pose1, width, height)
    output2 = Output(frame2, pts2, colors2, conf_map2, depth_map2, intrinsics2, cam_pose2, width, height)

    return output1, output2