import argparse
import os
import time
from pathlib import Path
import rasterio
import numpy as np
import math
import os
import random

import cv2
import torch
import torch.backends.cudnn as cudnn

from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, set_logging, \
    increment_path, scale_coords
from utils.plots import plot_one_box
from utils.torch_utils import select_device, time_synchronized


def crop_image_with_geo_info(src_path, out_folder, size=(4096, 4096)):
    """
    按照指定尺寸裁剪图像，并保持地理信息
    :param src_path: 源影像的路径 (例如: .tif)
    :param out_folder: 输出裁剪后影像的文件夹
    :param size: 裁剪块的尺寸 (宽, 高)
    """
    with rasterio.open(src_path) as src:
        profile = src.profile
        height, width = profile['height'], profile['width']

        x_blocks = math.ceil(width / size[0])
        y_blocks = math.ceil(height / size[1])

        # 获取源图像文件名（不带扩展名）
        base_name = os.path.basename(src_path).replace(".tif", "")

        # 为该图像创建一个文件夹（如果文件夹不存在）
        image_output_folder = os.path.join(out_folder, base_name)
        os.makedirs(image_output_folder, exist_ok=True)

        # 逐块裁剪并保存
        for y in range(y_blocks):
            for x in range(x_blocks):
                window = rasterio.windows.Window(
                    x * size[0],
                    y * size[1],
                    min(size[0], width - x * size[0]),
                    min(size[1], height - y * size[1])
                )

                # 读取块
                block_data = src.read(window=window)

                # 生成裁剪后影像的文件名
                block_filename = os.path.join(image_output_folder, f'{base_name}_block_{y}_{x}.tif')

                # 定义新的影像的 Profile
                transform = src.window_transform(window)
                new_profile = profile.copy()
                new_profile.update({
                    'height': block_data.shape[1],
                    'width': block_data.shape[2],
                    'transform': transform
                })

                # 保存裁剪后的块
                with rasterio.open(block_filename, 'w', **new_profile) as dst:
                    dst.write(block_data)
def crop_images_in_folder(src_folder, out_folder, size=(4096, 4096)):
    """
    裁剪文件夹中的所有影像
    :param src_folder: 源影像的文件夹路径
    :param out_folder: 输出裁剪后影像的文件夹路径
    :param size: 裁剪块的尺寸 (宽, 高)
    """
    # 确保输出文件夹存在
    os.makedirs(out_folder, exist_ok=True)

    # 获取源文件夹中的所有.tif文件
    for file_name in os.listdir(src_folder):
        if file_name.endswith('.tif'):
            src_path = os.path.join(src_folder, file_name)
            crop_image_with_geo_info(src_path, out_folder, size)
            print(f"裁剪 {file_name} 完成！")


def detect_images_in_folder(source, weights, imgsz, conf_thres, iou_thres, device, view_img, save_txt, save_img, project, name, exist_ok):
    """
    对文件夹中的图像进行检测
    """
    save_dir = Path(increment_path(Path(project) / name, exist_ok=exist_ok))  # 加载目录
    (save_dir / 'labels').mkdir(parents=True, exist_ok=True)  # 创建目录

    # 初始化日志
    set_logging()
    device = select_device(device)  # 选择设备
    half = device.type != 'cpu'  # 仅CUDA支持半精度

    # 加载模型
    model = attempt_load(weights, map_location=device)  # 加载FP32模型
    stride = int(model.stride.max())  # 模型步幅
    imgsz = check_img_size(imgsz, s=stride)  # 检查img_size

    if half:
        model.half()  # 转为FP16

    # 设置数据加载
    dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # 获取类别名称和颜色
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # 执行推理
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # 推理过程
        t1 = time_synchronized()
        with torch.no_grad():
            pred = model(img)[0]
        t2 = time_synchronized()

        # 应用NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, agnostic=False)
        t3 = time_synchronized()

        # 处理检测结果
        for i, det in enumerate(pred):  # 每张图像的检测结果
            p, s, im0 = path, '', im0s
            p = Path(p)  # 转换为Path对象

            # 保存路径设置
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + '.txt'  # label.txt
            s += f"{det.shape[0]}对象被探测到"  # 打印检测结果

            if det is not None and len(det):
                # 转换为xyxy格式
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # 打印预测
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # 每个类别的数量
                    s += f"类别 {names[int(c)]}，数量: {n} "

                # 结果保存
                if save_txt:  # 如果需要保存txt结果
                    with open(txt_path, 'a') as f:
                        for *xyxy, conf, cls in det:
                            f.write(('%g ' * 5 + '\n') % (*xyxy, conf, cls))  # 写入文件

                if save_img or view_img:  # 保存图片或显示图片
                    for *xyxy, conf, cls in det:
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

            # 保存结果
            if save_img:
                cv2.imwrite(save_path, im0)
            print(f'{s}，保存于 {save_path}')

    print(f'完成。({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 新增命令行参数
    parser.add_argument('--weights', type=str, default='F:/yolov7-main/runs/train/exp21/weights/best.pt', help='model.pt路径')
    parser.add_argument('--source', type=str, help='源文件夹路径')  # 源文件夹路径不设默认值
    parser.add_argument('--img-size', type=int, default=640, help='推理图像尺寸（像素）')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='对象置信度阈值')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS的IOU阈值')
    parser.add_argument('--device', default='0', help='CUDA设备，例如0或0,1,2,3或cpu')
    parser.add_argument('--view-img', action='store_true', help='显示结果')
    parser.add_argument('--save-txt', action='store_true', help='结果保存到*.txt')
    parser.add_argument('--save-conf', action='store_true', help='将置信度保存到--save-txt标签中')
    parser.add_argument('--nosave', action='store_true', help='不保存图像/视频')
    parser.add_argument('--project', default='runs/detect', help='将结果保存到项目/名称')
    parser.add_argument('--name', default='exp', help='将结果保存到项目/名称')
    parser.add_argument('--exist-ok', action='store_true', help='现有项目/名称可以，避免自增')
    parser.add_argument('--size', type=int, nargs=2, default=[4096, 4096], help='裁剪块尺寸（宽, 高）')

    opt = parser.parse_args()
    print(opt)

    # Perform cropping
    crop_folder = "F:/yolov7-main/inference/images"  # 源文件夹路径
    out_folder = f"F:/yolov7-main/inference/block"  # 设定裁剪后的文件夹路径
    os.makedirs( out_folder, exist_ok=True)  # 确保裁剪文件夹存在
    crop_images_in_folder(crop_folder, out_folder, size=tuple(opt.size))  # 确保传递的是元组

    # 更新source为裁剪后的文件夹路径
    opt.source = out_folder  # 将source的值设置为裁剪后的文件夹路径

    # Perform detection on cropped images
    detect_images_in_folder(opt.source, opt.weights, opt.img_size, opt.conf_thres, opt.iou_thres, opt.device, opt.view_img, opt.save_txt, True, opt.project, opt.name, opt.exist_ok)

