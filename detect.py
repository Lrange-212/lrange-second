# -*- coding:utf-8 -*-
from ultralytics import YOLOv10
import argparse

# 解析命令行参数
parser = argparse.ArgumentParser(description='Train or validate YOLO model.')
# train用于训练原始模型  val 用于得到精度指标
parser.add_argument('--mode', type=str, default='train', help='Mode of operation.')
# 预训练模型
#parser.add_argument('--weights', type=str, default='F:/yolov10-main/runs/detect/train107/weights/best.pt', help='Path to model file.')
#parser.add_argument('--weights', type=str, default='F:/yolov10-main/yolov10n.pt', help='Path to model file.')
# 是否使用nwdloss
parser.add_argument('--usenwd', type=str, default=False, help='Whether to use NWDLoss or not (True/False)')
# iou使用比例
parser.add_argument('--iou_ratio', type=float, default=0.5, help='Intersection over Union (IoU) threshold for NWDLoss')
#parser.add_argument('--cfg', type=str, default='ultralytics/cfg/models/v10/yolov10s_ECACA.yaml', help='model.yaml path')

# 新增模型配置文件路径参数
#parser.add_argument('--config', type=str, default='ultralytics/cfg/models/v10/yolov10s_ECACA.yaml', help='Path to model configuration file.')

# 数据集存放路径
parser.add_argument('--data', type=str, default='E:/PPFFYOLO/FF-YOLOv10/datasets/Data/data.yaml', help='Path to data file.')
parser.add_argument('--epoch', type=int, default=300, help='Number of epochs.')
parser.add_argument('--batch', type=int, default=32, help='Batch size.')
parser.add_argument('--workers', type=int, default=8, help='Number of workers.')
parser.add_argument('--device', type=str, default='0', help='Device to use.')
parser.add_argument('--name', type=str, default='', help='Name data file.')
args = parser.parse_args()


def train(model, data, epoch, batch, workers, device, name, usenwd, iou_ratio):
    model.train(data=data, epochs=epoch, batch=batch, workers=workers, device=device, name=name,
                nwdloss=usenwd, iou_ratio=iou_ratio,val_period=1)


def validate(model, data, batch, workers, device, name):
    model.val(data=data, batch=batch, workers=workers, device=device, name=name)

#F:/yolov10-main/ultralytics/cfg/models/v10/yolov10n_fasterNetend.yaml

def main():
    model =YOLOv10('E:/PPFFYOLO/FF-YOLOv10/ultralytics/cfg/models/v10/FFyolov10.yaml').load('yolov10n.pt')
    if args.mode == 'train':
        train(model, args.data, args.epoch, args.batch, args.workers, args.device, args.name, args.usenwd,
              args.iou_ratio)
    else:
        validate(model, args.data, args.batch, args.workers, args.device, args.name)


if __name__ == '__main__':
    main()

