import numpy as np
import os
from scipy.spatial.transform import Rotation as R

# ==================== 配置参数区域 ====================
# 请确认 input_images_... 的路径是否正确
NPZ_PATH = "./input_images_20260105_102300_260674/predictions.npz" 
IMAGES_FOLDER = "./input_images_20260105_102300_260674/images" 
OUTPUT_DIR = "./input_images_20260105_102300_260674/sparse/0" 
# =====================================================

def get_transformation_matrix(r_mat, t_vec):
    """构建 4x4 变换矩阵"""
    mat = np.eye(4)
    mat[:3, :3] = r_mat
    mat[:3, 3] = t_vec.flatten()
    return mat

def opengl_to_opencv_pose(w2c_opengl):
    """
    坐标系转换:
    如果你的重建结果在 COLMAP 中相机朝向是反的(Z轴相反)或者上下颠倒(Y轴相反)，
    可能需要注释掉或修改这个函数。
    """
    flup_mat = np.array([
        [1,  0,  0,  0],
        [0, -1,  0,  0],
        [0,  0, -1,  0],
        [0,  0,  0,  1]
    ])
    return np.matmul(flup_mat, w2c_opengl)

def convert_npz_to_colmap(npz_path, images_folder, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 加载数据
    print(f"正在加载 {npz_path} ...")
    if not os.path.exists(npz_path):
        print(f"错误: 文件不存在 {npz_path}")
        return

    data = np.load(npz_path, allow_pickle=True)
    
    # --- [修改部分] 适配你的 .npz keys ---
    try:
        # 你的 npz key 是 'extrinsic'，不是 'pred_cam_T'
        extrinsics = data['extrinsic'] 
        
        # 你的 npz key 是 'intrinsic'，不是 'pred_K'
        intrinsics_K = data['intrinsic'] 
        
        # 尝试自动获取图片尺寸
        if 'images' in data:
            # images 通常形状为 (N, H, W, 3)
            img_shape = data['images'][0].shape
            img_h, img_w = img_shape[0], img_shape[1]
            print(f"自动检测到图片尺寸: {img_w}x{img_h}")
        else:
            # 如果没有 images 数据，回退到默认值
            img_h = 1080
            img_w = 1920
            print(f"未在数据中找到图片，使用默认尺寸: {img_w}x{img_h}")

    except KeyError as e:
        print(f"错误: 找不到 Key {e}。请检查 .npz 结构。可用 keys: {data.files}")
        return
    # ------------------------------------------------

    image_files = sorted(os.listdir(images_folder))
    
    # 简单的数量检查
    if len(image_files) != len(extrinsics):
        print(f"警告: 文件夹图片数量 ({len(image_files)}) 与 .npz 中相机数量 ({len(extrinsics)}) 不匹配！")
        # 以较小者为准，防止越界
        num_cameras = min(len(image_files), len(extrinsics))
    else:
        num_cameras = len(extrinsics)

    # 2. 写入 cameras.txt (内参)
    print("正在生成 cameras.txt ...")
    with open(os.path.join(output_dir, "cameras.txt"), "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        
        for i in range(num_cameras):
            cam_id = i + 1
            K = intrinsics_K[i]
            fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
            f.write(f"{cam_id} PINHOLE {img_w} {img_h} {fx} {fy} {cx} {cy}\n")

    # 3. 写入 images.txt (外参)
    print("正在生成 images.txt ...")
    with open(os.path.join(output_dir, "images.txt"), "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        
        for i in range(num_cameras):
            img_name = image_files[i]
            image_id = i + 1
            cam_id = image_id
            
            # 获取外参矩阵
            mat = extrinsics[i] # 通常是 (4,4) 或 (3,4)
            
            # [兼容性处理] 确保矩阵是 4x4
            if mat.shape == (3, 4):
                mat = np.vstack([mat, [0, 0, 0, 1]])
            
            # [关键步骤] 坐标系处理
            # 这里的 'extrinsic' 通常代表 Camera-to-World (Pose)
            # COLMAP 需要 World-to-Camera (View Transform)
            # 所以通常需要求逆
            try:
                mat_w2c = np.linalg.inv(mat)
            except np.linalg.LinAlgError:
                print(f"警告: 第 {i} 帧矩阵无法求逆，跳过。")
                continue

            # [可选] OpenGL 到 OpenCV 转换
            # 如果你的 .npz 输出已经是 OpenCV 标准坐标系，这一步可能会导致相机颠倒。
            # 如果结果不对，请尝试注释掉下面这一行，直接使用 mat_w2c
            mat_colmap = opengl_to_opencv_pose(mat_w2c)
            # mat_colmap = mat_w2c # <--- 如果方向反了，用这行代替上一行
            
            r_mat = mat_colmap[:3, :3]
            t_vec = mat_colmap[:3, 3]
            
            # 旋转矩阵 -> 四元数
            rot = R.from_matrix(r_mat)
            quat = rot.as_quat() # (x, y, z, w)
            qw, qx, qy, qz = quat[3], quat[0], quat[1], quat[2]
            
            f.write(f"{image_id} {qw} {qx} {qy} {qz} {t_vec[0]} {t_vec[1]} {t_vec[2]} {cam_id} {img_name}\n")
            f.write("\n") # 空行代表没有 2D 特征点

    print(f"转换完成！输出目录: {output_dir}")
    print("提示: 如果 COLMAP 中相机看起来是红色的或者是反的，请检查代码中 'opengl_to_opencv_pose' 的部分。")

if __name__ == "__main__":
    convert_npz_to_colmap(NPZ_PATH, IMAGES_FOLDER, OUTPUT_DIR)
# 使用方法：
# 1. 在文件顶部的配置区域修改参数
# 2. 直接运行: python npz2txt.py