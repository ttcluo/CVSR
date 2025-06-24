import os
import argparse
from glob import glob

def generate_meta_file(gt_folder, output_file):
    """
    生成meta_info文件

    参数:
        gt_folder: GT文件夹路径
        output_file: 输出文件路径
    """
    # 获取所有子目录
    subdirs = sorted([d for d in os.listdir(gt_folder)
                     if os.path.isdir(os.path.join(gt_folder, d))])

    with open(output_file, 'w') as f:
        for subdir in subdirs:
            subdir_path = os.path.join(gt_folder, subdir)

            # 计算该目录下的PNG文件数量
            png_files = glob(os.path.join(subdir_path, '*.png'))
            num_files = len(png_files)

            # 写入格式: 子目录名 文件数量 (640,640,3)
            line = f"{subdir} {num_files} (640,640,3)\n"
            f.write(line)

            print(f"处理完成: {subdir} - 共 {num_files} 个文件")

if __name__ == "__main__":
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='生成数据集meta文件')
    parser.add_argument('--gt', type=str, required=True,
                       help='GT文件夹路径')
    parser.add_argument('--output', type=str, default='meta_info_jilin189_GT.txt',
                       help='输出文件路径 (默认: meta_info_jilin189_GT.txt)')

    args = parser.parse_args()

    # 检查GT文件夹是否存在
    if not os.path.isdir(args.gt):
        raise ValueError(f"指定的GT文件夹不存在: {args.gt}")

    # 生成meta文件
    generate_meta_file(args.gt, args.output)
    print(f"Meta文件已生成: {args.output}")