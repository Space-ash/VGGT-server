# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random
import numpy as np
import glob
import os
import copy
import torch
import torch.nn.functional as F

# Configure CUDA settings
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

import argparse
from pathlib import Path
import trimesh
import pycolmap


from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images_square
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.helper import create_pixel_coordinate_grid, randomly_limit_trues
from vggt.dependency.track_predict import predict_tracks
from vggt.dependency.np_to_pycolmap import batch_np_matrix_to_pycolmap, batch_np_matrix_to_pycolmap_wo_track


# TODO: add support for masks
# TODO: add iterative BA
# TODO: add support for radial distortion, which needs extra_params
# TODO: test with more cases
# TODO: test different camera types


def parse_args():
    parser = argparse.ArgumentParser(description="VGGT Demo")
    parser.add_argument("--scene_dir", type=str, required=True, help="Directory containing the scene images")
    
    # --- 新增/修改的参数 start ---
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save the reconstruction results. If None, saves to scene_dir (will fail on Kaggle input).")
    parser.add_argument("--model_path", type=str, default="/kaggle/input/vggt-input-laojun/model.pt", help="Path to the local model weights file")
    # --- 新增/修改的参数 end ---

    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--use_ba", action="store_true", default=False, help="Use BA for reconstruction")
    ######### BA parameters #########
    parser.add_argument(
        "--max_reproj_error", type=float, default=8.0, help="Maximum reprojection error for reconstruction"
    )
    parser.add_argument("--shared_camera", action="store_true", default=False, help="Use shared camera for all images")
    parser.add_argument("--camera_type", type=str, default="SIMPLE_PINHOLE", help="Camera type for reconstruction")
    parser.add_argument("--vis_thresh", type=float, default=0.2, help="Visibility threshold for tracks")
    parser.add_argument("--query_frame_num", type=int, default=8, help="Number of frames to query")
    parser.add_argument("--max_query_pts", type=int, default=4096, help="Maximum number of query points")
    parser.add_argument(
        "--fine_tracking", action="store_true", default=True, help="Use fine tracking (slower but more accurate)"
    )
    parser.add_argument(
        "--conf_thres_value", type=float, default=5.0, help="Confidence threshold value for depth filtering (wo BA)"
    )
    return parser.parse_args()


def run_VGGT(model, images, dtype, resolution=518):
    # images: [B, 3, H, W]

    assert len(images.shape) == 4
    assert images.shape[1] == 3

    # hard-coded to use 518 for VGGT
    images = F.interpolate(images, size=(resolution, resolution), mode="bilinear", align_corners=False)

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            images = images[None]  # add batch dimension
            aggregated_tokens_list, ps_idx = model.aggregator(images)

        # Predict Cameras
        pose_enc = model.camera_head(aggregated_tokens_list)[-1]
        # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
        # Predict Depth Maps
        depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, ps_idx)

    extrinsic = extrinsic.squeeze(0).cpu().numpy()
    intrinsic = intrinsic.squeeze(0).cpu().numpy()
    depth_map = depth_map.squeeze(0).cpu().numpy()
    depth_conf = depth_conf.squeeze(0).cpu().numpy()
    return extrinsic, intrinsic, depth_map, depth_conf


def demo_fn(args):
    # Print configuration
    print("Arguments:", vars(args))

    # Set seed for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)  # for multi-GPU
    print(f"Setting seed as: {args.seed}")

    # Set device and dtype
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Using dtype: {dtype}")

    # Run VGGT for camera and depth estimation
    model = VGGT()
    
    # --- 修改部分：优先从本地 args.model_path 加载模型 ---
    model_loaded = False
    if hasattr(args, 'model_path') and args.model_path and os.path.exists(args.model_path):
        try:
            # 处理路径可能是目录也可能是文件的情况
            load_path = args.model_path
            if os.path.isdir(load_path):
                # 如果是目录，尝试寻找 .pt 文件
                pt_files = glob.glob(os.path.join(load_path, "*.pt"))
                if pt_files:
                    load_path = pt_files[0]
            
            if os.path.isfile(load_path):
                print(f"Loading weights from local path: {load_path}")
                # 使用 torch.load 加载本地权重
                state_dict = torch.load(load_path, map_location="cpu")
                model.load_state_dict(state_dict)
                model_loaded = True
            else:
                print(f"Warning: Model path {args.model_path} provided but file not found.")
        except Exception as e:
            print(f"Error loading local model: {e}. Fallback to URL.")

    if not model_loaded:
        print("Downloading weights from HuggingFace...")
        _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
        model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
    
    # -----------------------------------------------

    model.eval()
    model = model.to(device)
    print(f"Model loaded")

    # Get image paths and preprocess them
    # Check if images are in scene_dir or scene_dir/images
    if os.path.exists(os.path.join(args.scene_dir, "images")):
        image_dir = os.path.join(args.scene_dir, "images")
    else:
        image_dir = args.scene_dir
        
    image_path_list = glob.glob(os.path.join(image_dir, "*"))
    # Filter for image extensions to avoid errors
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_path_list = [p for p in image_path_list if os.path.splitext(p)[1].lower() in image_extensions]
    image_path_list.sort() # Ensure consistent order

    if len(image_path_list) == 0:
        raise ValueError(f"No images found in {image_dir}")
    base_image_path_list = [os.path.basename(path) for path in image_path_list]

    # Load images and original coordinates
    # Load Image in 1024, while running VGGT with 518
    vggt_fixed_resolution = 518
    img_load_resolution = 1024

    images, original_coords = load_and_preprocess_images_square(image_path_list, img_load_resolution)
    images = images.to(device)
    original_coords = original_coords.to(device)
    print(f"Loaded {len(images)} images from {image_dir}")

    # Run VGGT to estimate camera and depth
    # Run with 518x518 images
    extrinsic, intrinsic, depth_map, depth_conf = run_VGGT(model, images, dtype, vggt_fixed_resolution)
    points_3d = unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic)

    if args.use_ba:
        image_size = np.array(images.shape[-2:])
        scale = img_load_resolution / vggt_fixed_resolution
        shared_camera = args.shared_camera

        with torch.cuda.amp.autocast(dtype=dtype):
            # Predicting Tracks
            # Using VGGSfM tracker instead of VGGT tracker for efficiency
            # VGGT tracker requires multiple backbone runs to query different frames (this is a problem caused by the training process)
            # Will be fixed in VGGT v2

            # You can also change the pred_tracks to tracks from any other methods
            # e.g., from COLMAP, from CoTracker, or by chaining 2D matches from Lightglue/LoFTR.
            pred_tracks, pred_vis_scores, pred_confs, points_3d, points_rgb = predict_tracks(
                images,
                conf=depth_conf,
                points_3d=points_3d,
                masks=None,
                max_query_pts=args.max_query_pts,
                query_frame_num=args.query_frame_num,
                keypoint_extractor="aliked+sp",
                fine_tracking=args.fine_tracking,
            )

            torch.cuda.empty_cache()

        # rescale the intrinsic matrix from 518 to 1024
        intrinsic[:, :2, :] *= scale
        track_mask = pred_vis_scores > args.vis_thresh

        # TODO: radial distortion, iterative BA, masks
        reconstruction, valid_track_mask = batch_np_matrix_to_pycolmap(
            points_3d,
            extrinsic,
            intrinsic,
            pred_tracks,
            image_size,
            masks=track_mask,
            max_reproj_error=args.max_reproj_error,
            shared_camera=shared_camera,
            camera_type=args.camera_type,
            points_rgb=points_rgb,
        )

        if reconstruction is None:
            raise ValueError("No reconstruction can be built with BA")

        # Bundle Adjustment
        ba_options = pycolmap.BundleAdjustmentOptions()
        pycolmap.bundle_adjustment(reconstruction, ba_options)

        reconstruction_resolution = img_load_resolution
    else:
        conf_thres_value = args.conf_thres_value
        max_points_for_colmap = 100000  # randomly sample 3D points
        shared_camera = False  # in the feedforward manner, we do not support shared camera
        camera_type = "PINHOLE"  # in the feedforward manner, we only support PINHOLE camera

        image_size = np.array([vggt_fixed_resolution, vggt_fixed_resolution])
        num_frames, height, width, _ = points_3d.shape

        points_rgb = F.interpolate(
            images, size=(vggt_fixed_resolution, vggt_fixed_resolution), mode="bilinear", align_corners=False
        )
        points_rgb = (points_rgb.cpu().numpy() * 255).astype(np.uint8)
        points_rgb = points_rgb.transpose(0, 2, 3, 1)

        # (S, H, W, 3), with x, y coordinates and frame indices
        points_xyf = create_pixel_coordinate_grid(num_frames, height, width)

        conf_mask = depth_conf >= conf_thres_value
        # at most writing 100000 3d points to colmap reconstruction object
        conf_mask = randomly_limit_trues(conf_mask, max_points_for_colmap)

        points_3d = points_3d[conf_mask]
        points_xyf = points_xyf[conf_mask]
        points_rgb = points_rgb[conf_mask]

        print("Converting to COLMAP format")
        reconstruction = batch_np_matrix_to_pycolmap_wo_track(
            points_3d,
            points_xyf,
            points_rgb,
            extrinsic,
            intrinsic,
            image_size,
            shared_camera=shared_camera,
            camera_type=camera_type,
        )

        reconstruction_resolution = vggt_fixed_resolution

    reconstruction = rename_colmap_recons_and_rescale_camera(
        reconstruction,
        base_image_path_list,
        original_coords.cpu().numpy(),
        img_size=reconstruction_resolution,
        shift_point2d_to_original_res=True,
        shared_camera=shared_camera,
    )

    # --- 修改部分：处理输出路径 ---
    # 如果指定了 output_dir，则使用该路径；否则使用 scene_dir（如果 scene_dir 只读则会报错）
    if hasattr(args, 'output_dir') and args.output_dir:
        output_base = args.output_dir
    else:
        output_base = args.scene_dir
    
    print(f"Saving reconstruction to {output_base}/sparse")
    sparse_reconstruction_dir = os.path.join(output_base, "sparse")
    os.makedirs(sparse_reconstruction_dir, exist_ok=True)
    reconstruction.write(sparse_reconstruction_dir)

    # Save point cloud for fast visualization
    trimesh.PointCloud(points_3d, colors=points_rgb).export(os.path.join(sparse_reconstruction_dir, "points.ply"))
    # ---------------------------

    return True


def rename_colmap_recons_and_rescale_camera(
    reconstruction, image_paths, original_coords, img_size, shift_point2d_to_original_res=False, shared_camera=False
):
    rescale_camera = True

    for pyimageid in reconstruction.images:
        # Reshaped the padded&resized image to the original size
        # Rename the images to the original names
        pyimage = reconstruction.images[pyimageid]
        pycamera = reconstruction.cameras[pyimage.camera_id]
        pyimage.name = image_paths[pyimageid - 1]

        if rescale_camera:
            # Rescale the camera parameters
            pred_params = copy.deepcopy(pycamera.params)

            real_image_size = original_coords[pyimageid - 1, -2:]
            resize_ratio = max(real_image_size) / img_size
            pred_params = pred_params * resize_ratio
            real_pp = real_image_size / 2
            pred_params[-2:] = real_pp  # center of the image

            pycamera.params = pred_params
            pycamera.width = real_image_size[0]
            pycamera.height = real_image_size[1]

        if shift_point2d_to_original_res:
            # Also shift the point2D to original resolution
            top_left = original_coords[pyimageid - 1, :2]

            for point2D in pyimage.points2D:
                point2D.xy = (point2D.xy - top_left) * resize_ratio

        if shared_camera:
            # If shared_camera, all images share the same camera
            # no need to rescale any more
            rescale_camera = False

    return reconstruction


if __name__ == "__main__":
    args = parse_args()
    with torch.no_grad():
        demo_fn(args)


# Work in Progress (WIP)

"""
VGGT Runner Script (Modified for Kaggle)
=======================================

使用说明 (Usage Guide):

在 Kaggle Notebook 中使用时，你既可以在命令行运行，也可以在 Notebook Cell 中直接调用。
最方便的方法是直接定义一个参数类，如下所示：

```python
# 1. 定义参数 (模拟 argparse 的参数)
class Args:
    # 必须修改的路径
    scene_dir = "/kaggle/input/your-dataset/scene_name"       # 输入图片路径 (只读)
    output_dir = "/kaggle/working/output_scene_name"          # 输出结果路径 (可写)
    model_path = "/kaggle/input/vggt-input-laojun/model.pt"   # 你的本地模型路径
    
    # 可选参数 (保持默认即可)
    seed = 42
    use_ba = False   # 是否使用 Bundle Adjustment
    max_reproj_error = 8.0
    shared_camera = False
    camera_type = "SIMPLE_PINHOLE"
    vis_thresh = 0.2
    query_frame_num = 8
    max_query_pts = 4096
    fine_tracking = True
    conf_thres_value = 5.0
    
    class Args:
    # --- 基础设置 ---
    seed = 42                       # 随机种子：固定这个数字可以保证每次运行结果一致（复现性）
    
    # --- 核心模式选择 ---
    use_ba = False                  # 是否使用 Bundle Adjustment (光束法平差)：
                                    # False = 前馈模式 (Feed-forward)。速度极快，直接输出预测结果。
                                    # True  = 优化模式。速度慢，但会优化相机位姿和点云，精度更高。

    # --- 优化模式参数 (当 use_ba = True 时生效) ---
    max_reproj_error = 8.0          # 最大重投影误差 (像素)：
                                    # 在优化过程中，如果一个3D点投影回图像的误差超过8像素，就会被认为是"坏点"并剔除。
                                    # 调小(如 2.0)会让点云更干净但点数变少；调大(如 10.0)点数多但杂讯多。
    
    vis_thresh = 0.2                # 轨迹可见性阈值：
                                    # 过滤掉置信度低于 0.2 的特征点轨迹（Track）。
    
    query_frame_num = 8             # 查询帧数：
                                    # 在追踪特征点时，每一帧会参考前后多少帧来寻找匹配。
                                    # 数值越大，匹配越鲁棒（能处理大动作），但显存消耗和计算时间显著增加。
    
    max_query_pts = 4096            # 最大查询点数：
                                    # 每一帧最多处理多少个关键点。显存不够时可以调小这个值 (例如 2048)。
    
    fine_tracking = True            # 开启精细追踪：
                                    # True = 使用更耗时的算法进行亚像素级别的点位修正，精度更高。
    
    # --- 前馈模式参数 (当 use_ba = False 时生效) ---
    conf_thres_value = 5.0          # 深度置信度阈值：
                                    # 模型输出的深度图包含置信度。只有置信度 > 5.0 的像素才会被转换成 3D 点。
                                    # 调高此值可以减少空中的噪点，但可能会导致物体变稀疏。

    # --- 相机模型设置 ---
    shared_camera = False           # 是否共享相机参数：
                                    # True  = 假设所有图片是用同一台相机、同一个焦距拍摄的（适合视频）。
                                    # False = 假设每张图片可能有不同的焦距（适合网上的混合图片集）。
    
    camera_type = "SIMPLE_PINHOLE"  # 相机模型类型：
                                    # "SIMPLE_PINHOLE": 只有 1 个焦距参数 (f)，光心 (cx, cy) 固定在图像中心。
                                    # "PINHOLE": 包含焦距 (fx, fy) 和光心 (cx, cy)。

# 2. 运行函数
import torch
import os

# 确保输出目录存在 (虽然脚本里会创建，但为了安全可以先检查)
if not os.path.exists(Args.output_dir):
    os.makedirs(Args.output_dir)

print(f"Start processing {Args.scene_dir}...")

with torch.no_grad():
    # 直接调用主函数，传入参数对象
    demo_fn(Args())

print(f"Done! Results saved to {Args.output_dir}")
"""