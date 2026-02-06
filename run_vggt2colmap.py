import os
import torch

# 定义参数类
class Args:
    # 必须修改的路径
    scene_dir = "./input/images"    # 输入图片路径 (只读)
    output_dir = "./output"         # 输出结果路径 (可写)
    vggt_model_path = "./model/vggt_model.pt"       # VGGT 主模型路径
    vggsfm_model_path = "./model/vggsfm_v2_tracker.pt"  # VGGSfM tracker 模型路径
    
    # 核心参数
    seed = 42        # 随机种子
    max_points_for_colmap = 100000  # 用于 COLMAP 的最大点数
    use_ba = False   # 是否使用 Bundle Adjustment

    # 相机参数
    shared_camera = False
    camera_type = "SIMPLE_PINHOLE"

    # BA-True
    max_reproj_error = 8.0
    vis_thresh = 0.02
    query_frame_num = 4
    max_query_pts = 2048
    fine_tracking = True

    # BA-False
    conf_thres_value = 2


if __name__ == "__main__":
    # 尝试导入
    try:
        import demo_colmap
        from demo_colmap import demo_fn
        
        print("成功导入 demo_colmap 模块")
        
        # 创建参数实例
        args = Args()
        
        # 开始运行
        print(f"开始处理: {args.scene_dir}")

        # 确保输出目录存在
        if hasattr(args, 'output_dir') and args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)

        # 调用函数
        with torch.no_grad():
            # 传递参数实例
            demo_fn(args)

        print(f"处理完成! 结果保存在: {args.output_dir}")
        
    except ImportError as e:
        print(f"导入失败: {e}")
        print(f"请检查路径下是否有 demo_colmap.py")
    except Exception as e:
        print(f"执行出错: {e}")
        import traceback
        traceback.print_exc()