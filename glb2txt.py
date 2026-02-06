import trimesh
import numpy as np
import os

# ==================== 配置参数区域 ====================
# 在这里直接设置参数，无需命令行传参
GLB_PATH = "input_images_20260105_102300_260674/glbscene_20_All_maskbFalse_maskwFalse_camTrue_skyFalse_predPointmap_Branch.glb"  # VGGT .glb 文件路径
OUTPUT_DIR = "./input_images_20260105_102300_260674/sparse/0"  # 输出 sparse/0 文件夹路径
# =====================================================

def convert_glb_to_points3d(glb_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "points3D.txt")
    
    print(f"正在加载 {glb_path} ...")
    # force='mesh' 即使文件只包含点，也作为 mesh 结构加载
    mesh = trimesh.load(glb_path, force='mesh')
    
    vertices = mesh.vertices
    
    # 处理颜色
    if hasattr(mesh.visual, 'vertex_colors') and len(mesh.visual.vertex_colors) > 0:
        # trimesh 颜色通常是 RGBA (0-255)
        colors = mesh.visual.vertex_colors[:, :3].astype(int)
    else:
        print("警告: .glb 文件中没有颜色信息，使用随机颜色代替。")
        colors = np.random.randint(0, 255, size=(len(vertices), 3))

    print(f"正在写入 {output_file} (包含 {len(vertices)} 个点)...")
    
    with open(output_file, "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        
        for i in range(len(vertices)):
            point_id = i + 1
            x, y, z = vertices[i]
            r, g, b = colors[i]
            # ERROR 设为 0，TRACK 留空
            f.write(f"{point_id} {x} {y} {z} {r} {g} {b} 0.0\n")

    print("点云转换完成！")

if __name__ == "__main__":
    convert_glb_to_points3d(GLB_PATH, OUTPUT_DIR)
# 使用方法：
# 1. 在文件顶部的配置区域修改参数
# 2. 直接运行: python glb2txt.py