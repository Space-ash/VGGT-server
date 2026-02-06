import os
import struct
import argparse
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as R

# ================= 配置参数区域 (默认值) =================
# 你可以在这里修改默认路径，也可以通过命令行参数覆盖
DEFAULT_NPZ_PATH = "./input_images/predictions.npz"
DEFAULT_GLB_PATH = "./input_images/scene.glb"
DEFAULT_IMAGES_FOLDER = "./input_images/images"
DEFAULT_OUTPUT_DIR = "./input_images/sparse/0"
# ========================================================

def get_transformation_matrix(r_mat, t_vec):
    """构建 4x4 变换矩阵"""
    mat = np.eye(4)
    mat[:3, :3] = r_mat
    mat[:3, 3] = t_vec.flatten()
    return mat

def opengl_to_opencv_pose(w2c_opengl):
    """
    坐标系转换: OpenGL (Y上, -Z前) -> OpenCV (Y下, +Z前)
    这是 VGGT/NeRF 类数据常见的转换需求。
    """
    flup_mat = np.array([
        [1,  0,  0,  0],
        [0, -1,  0,  0],
        [0,  0, -1,  0],
        [0,  0,  0,  1]
    ])
    return np.matmul(flup_mat, w2c_opengl)

# ================= COLMAP 二进制写入函数 =================

def write_cameras_binary(cameras, path):
    """
    写入 cameras.bin
    Format:
        NUM_CAMERAS (uint64)
        [CAMERA_ID (int32), MODEL_ID (int32), WIDTH (uint64), HEIGHT (uint64), PARAMS (double*N)]
    """
    print(f"正在写入 {path} ...")
    with open(path, "wb") as fid:
        # 写入相机数量
        fid.write(struct.pack("<Q", len(cameras)))
        for cam in cameras:
            cam_id = cam['id']
            model_id = 1  # PINHOLE
            width = cam['width']
            height = cam['height']
            params = cam['params'] # [fx, fy, cx, cy]
            
            fid.write(struct.pack("<iiQQ", cam_id, model_id, width, height))
            for p in params:
                fid.write(struct.pack("<d", p))

def write_images_binary(images, path):
    """
    写入 images.bin
    Format:
        NUM_IMAGES (uint64)
        [IMAGE_ID (int32), QW, QX, QY, QZ, TX, TY, TZ (double), CAMERA_ID (int32), NAME (string\0), NUM_POINTS2D (uint64), POINTS2D...]
    """
    print(f"正在写入 {path} ...")
    with open(path, "wb") as fid:
        fid.write(struct.pack("<Q", len(images)))
        for img in images:
            img_id = img['id']
            qw, qx, qy, qz = img['qvec']
            tx, ty, tz = img['tvec']
            cam_id = img['camera_id']
            name = img['name']
            
            fid.write(struct.pack("<idddddddi", img_id, qw, qx, qy, qz, tx, ty, tz, cam_id))
            
            # 写入文件名 (以 \0 结尾)
            name_bytes = name.encode("utf-8") + b"\x00"
            fid.write(name_bytes)
            
            # 写入 2D 点数量 (此处为 0，因为我们没有特征点匹配数据)
            fid.write(struct.pack("<Q", 0))

def write_points3D_binary(points, path):
    """
    写入 points3D.bin
    Format:
        NUM_POINTS (uint64)
        [POINT3D_ID (uint64), X, Y, Z (double), R, G, B (uint8), ERROR (double), TRACK_LENGTH (uint64), TRACKS...]
    """
    print(f"正在写入 {path} ...")
    with open(path, "wb") as fid:
        fid.write(struct.pack("<Q", len(points)))
        for pt in points:
            pt_id = pt['id']
            x, y, z = pt['xyz']
            r, g, b = pt['rgb']
            error = 0.0
            track_length = 0
            
            fid.write(struct.pack("<QdddBBBdQ", pt_id, x, y, z, r, g, b, error, track_length))

# ================= 主转换逻辑 =================

def convert(npz_path, glb_path, images_folder, output_dir):
    os.makedirs(output_dir, exist_ok=True)

# ---------------- 1. 处理 GLB 点云 ----------------
    print(f"加载 GLB 文件: {glb_path}")
    if not os.path.exists(glb_path):
        print(f"错误: 找不到文件 {glb_path}")
        return

    # force='mesh' 确保即使全是点也被读取
    mesh = trimesh.load(glb_path, force='mesh')
    
    # 检查是否为 Scene 对象，如果是，则合并所有几何体
    if isinstance(mesh, trimesh.Scene):
        print("检测到 GLB 包含多个子对象 (Scene)，正在合并几何体...")
        # dump(concatenate=True) 会将场景中所有几何体合并为一个 Trimesh 对象
        # 这样我们就能拿到所有的顶点
        mesh = mesh.dump(concatenate=True)
    
    vertices = mesh.vertices
    print(f"点云总点数: {len(vertices)}")
    
    # 先获取初始颜色，因为增密时需要用到它
    if hasattr(mesh.visual, 'vertex_colors') and len(mesh.visual.vertex_colors) > 0:
        # trimesh 颜色通常是 RGBA (0-255)，只取前3位 RGB
        colors = mesh.visual.vertex_colors[:, :3].astype(int)
    else:
        print("警告: .glb 文件中没有颜色信息，使用随机颜色代替。")
        colors = np.random.randint(0, 255, size=(len(vertices), 3))

    # 封装数据
    points_data = []
    for i in range(len(vertices)):
        points_data.append({
            'id': i + 1,
            'xyz': vertices[i],
            'rgb': colors[i]
        })

    write_points3D_binary(points_data, os.path.join(output_dir, "points3D.bin"))
    
    # ---------------- 2. 处理 NPZ 相机数据 ----------------
    print(f"加载 NPZ 文件: {npz_path}")
    if not os.path.exists(npz_path):
        print(f"错误: 找不到文件 {npz_path}")
        return
        
    data = np.load(npz_path, allow_pickle=True)
    
    try:
        extrinsics = data['extrinsic'] # Pose (Camera-to-World)
        intrinsics_K = data['intrinsic']
        
        # 尝试检测图片尺寸
        if 'images' in data:
            img_shape = data['images'][0].shape
            img_h, img_w = img_shape[0], img_shape[1]
            print(f"检测到图片尺寸: {img_w}x{img_h}")
        else:
            img_h, img_w = 1080, 1920
            print(f"使用默认图片尺寸: {img_w}x{img_h}")

    except KeyError as e:
        print(f"错误: NPZ 中缺少 Key {e}")
        return

    # 获取图片文件名列表
    if os.path.exists(images_folder):
        image_files = sorted(os.listdir(images_folder))
    else:
        print(f"警告: 图片文件夹 {images_folder} 不存在，将生成虚拟文件名。")
        image_files = [f"{i:05d}.jpg" for i in range(len(extrinsics))]

    num_cameras = min(len(image_files), len(extrinsics))
    
    cameras_data = []
    images_data = []

    for i in range(num_cameras):
        # --- 准备 Camera 数据 ---
        cam_id = i + 1
        K = intrinsics_K[i]
        fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
        
        cameras_data.append({
            'id': cam_id,
            'width': img_w,
            'height': img_h,
            'params': [fx, fy, cx, cy]
        })

        # --- 准备 Image 数据 ---
        # 1. 补全 3x4 到 4x4
        mat = extrinsics[i]
        if mat.shape == (3, 4):
            mat = np.vstack([mat, [0, 0, 0, 1]])
        
        # 2. 求逆 (Pose -> View / Cam2World -> World2Cam)
        try:
            mat_w2c = np.linalg.inv(mat)
        except np.linalg.LinAlgError:
            print(f"跳过无法求逆的帧: {i}")
            continue
        
        # 3. 坐标系转换 (OpenGL -> OpenCV)
        mat_colmap = opengl_to_opencv_pose(mat_w2c)
        
        r_mat = mat_colmap[:3, :3]
        t_vec = mat_colmap[:3, 3]
        
        # 4. 旋转矩阵 -> 四元数 (scipy 返回 x,y,z,w)
        rot = R.from_matrix(r_mat)
        quat = rot.as_quat() 
        # COLMAP 需要 w,x,y,z
        qw, qx, qy, qz = quat[3], quat[0], quat[1], quat[2]
        
        images_data.append({
            'id': i + 1,
            'qvec': [qw, qx, qy, qz],
            'tvec': t_vec,
            'camera_id': cam_id,
            'name': image_files[i]
        })

    write_cameras_binary(cameras_data, os.path.join(output_dir, "cameras.bin"))
    write_images_binary(images_data, os.path.join(output_dir, "images.bin"))

    print(f"\n✅ 转换完成！COLMAP 格式文件已保存至: {output_dir}")
    print("现在可以直接将此文件夹用于 3DGS 训练。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert VGGT .npz and .glb to COLMAP binary format.")
    parser.add_argument("--npz_path", default=DEFAULT_NPZ_PATH, help="Path to predictions.npz")
    parser.add_argument("--glb_path", default=DEFAULT_GLB_PATH, help="Path to scene .glb file")
    parser.add_argument("--images_folder", default=DEFAULT_IMAGES_FOLDER, help="Path to input images folder")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR, help="Output directory for COLMAP files")
    
    args = parser.parse_args()
    
    convert(args.npz_path, args.glb_path, args.images_folder, args.output_dir)
    
# python convert.py --npz_path ./data/predictions.npz --glb_path ./data/scene.glb --images_folder ./data/images --output_dir ./data/sparse/0