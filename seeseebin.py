import numpy as np
import struct
import open3d as o3d


def read_points3D_binary(path_to_model_file):
    """
    从COLMAP的points3D.bin文件读取3D点云数据
    
    参数:
        path_to_model_file: points3D.bin文件的路径
    
    返回:
        points3D: 字典，包含点ID和点数据
    """
    points3D = {}
    
    with open(path_to_model_file, "rb") as fid:
        num_points = struct.unpack("Q", fid.read(8))[0]
        
        for _ in range(num_points):
            binary_point_line_properties = fid.read(43)
            
            point3D_id = struct.unpack("Q", binary_point_line_properties[0:8])[0]
            xyz = struct.unpack("ddd", binary_point_line_properties[8:32])
            rgb = struct.unpack("BBB", binary_point_line_properties[32:35])
            error = struct.unpack("d", binary_point_line_properties[35:43])[0]
            
            track_length = struct.unpack("Q", fid.read(8))[0]
            track_elems = fid.read(8 * track_length)
            
            points3D[point3D_id] = {
                'xyz': np.array(xyz),
                'rgb': np.array(rgb) / 255.0,  # 归一化到[0,1]
                'error': error,
                'track_length': track_length
            }
    
    return points3D


def visualize_point_cloud(points3D, max_points=None):
    """
    使用Open3D可视化3D点云
    
    参数:
        points3D: 点云字典
        max_points: 最大显示点数（用于大规模点云的降采样）
    """
    # 提取xyz坐标和RGB颜色
    xyz = np.array([p['xyz'] for p in points3D.values()])
    rgb = np.array([p['rgb'] for p in points3D.values()])
    
    # 如果点太多，进行降采样
    if max_points and len(xyz) > max_points:
        indices = np.random.choice(len(xyz), max_points, replace=False)
        xyz = xyz[indices]
        rgb = rgb[indices]
    
    # 创建Open3D点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    
    print(f"点云信息:")
    print(f"  总点数: {len(xyz)}")
    print(f"  边界框: min={xyz.min(axis=0)}, max={xyz.max(axis=0)}")
    
    # 可视化
    print("\n正在打开3D可视化窗口...")
    print("操作提示:")
    print("  - 鼠标左键拖动: 旋转视角")
    print("  - 鼠标滚轮: 缩放")
    print("  - 鼠标右键拖动: 平移")
    print("  - 按 H 键: 查看更多帮助")
    
    o3d.visualization.draw_geometries(
        [pcd],
        window_name="3D Point Cloud Visualization",
        width=1024,
        height=768,
        point_show_normal=False
    )


def save_point_cloud_as_ply(points3D, output_path):
    """
    将点云保存为PLY格式
    
    参数:
        points3D: 点云字典
        output_path: 输出PLY文件路径
    """
    xyz = np.array([p['xyz'] for p in points3D.values()])
    rgb = np.array([p['rgb'] for p in points3D.values()])
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    
    o3d.io.write_point_cloud(output_path, pcd)
    print(f"点云已保存到: {output_path}")


if __name__ == "__main__":
    # 点云文件路径
    points3D_path = r"C:\Users\26092\Downloads\kaggle-VGGT-output\house\sparse\0\points3D.bin"
    
    print("正在读取points3D.bin文件...")
    points3D = read_points3D_binary(points3D_path)
    print(f"成功读取 {len(points3D)} 个3D点")
    
    # 可视化点云
    visualize_point_cloud(points3D, max_points=100000)  # 限制最多显示10万个点
    
    # 可选：保存为PLY格式
    # save_point_cloud_as_ply(points3D, "output_pointcloud.ply")
