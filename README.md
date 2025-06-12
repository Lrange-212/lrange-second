#数据集网盘链接
通过网盘分享的文件：PPFFYOLO.zip
链接: https://pan.baidu.com/s/1PQhmD-Sr3UEA5t6dWQpm_g 提取码: hnsv 

#相关模型的代码分别在对应的分支中

0.Installation
创建训练环境
conda create -n yolo python=3.8
conda activate yolo
pip install -r requirements.txt


# PP-YOLOv7
Implementation of paper - YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors.https://doi.org/10.48550/arXiv.2207.02696

1.数据集的准备
训练前需要自己制作好数据集，并划分为train和val，其中标签文件（txt格式）放在label文件夹下，图片在image文件夹下。
创建一个新的data.yaml文件，包括train and val data as 1) directory: path/images/, 2) file: path/images.txt, or 3) list: [path1/images/, path2/images/]
# number of classes
nc: 1
# class names
names: ['Phoca largha']

2.网络训练
修改train.py文件中的相关信息后即可运行
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov7.pt', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='PPyolov7.yaml', help='model.yaml path')
    parser.add_argument('--data', type=str, default='data/data.yaml', help='data.yaml path')
    parser.add_argument('--hyp', type=str, default='data/hyp.scratch.p5.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=32, help='total batch size for all GPUs')
    parser.add_argument('--img-size', nargs='+', type=int, default=[480, 480], help='[train, test] image sizes')
   
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    
    parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')

    opt = parser.parse_args()


3.训练结果预测
使用自己训练好的模型进行预测一定要修改detect.py文件中的weights，同时sources是需要进行预测的图片或文件夹路径
  if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='G:/yolov7-main/runs/train/exp21/weights/best.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='G:/img', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    
    opt = parser.parse_args()
    print(opt)
4.验证
需修改test.py文件中的weights和data，修改为自己的权重文件和数据集文件路径。



#FF-YOLOv10
Implementation of paper - YOLOv10: Real-Time End-to-End Object Detection. https://arxiv.org/abs/2405.14458

1.训练
yolo detect train data=coco.yaml model=yolov10n/s/m/b/l/x.yaml epochs=500 batch=256 imgsz=640 device=0,1,2,3,4,5,6,7
或直接运行文件detect.py，并修改相应的参数。

2.预测
直接运行文件infer.py，并修改相应的参数。
from ultralytics import YOLOv10
if __name__ == '__main__':
    model = YOLOv10('runs/detect/train48/weights/best.pt') # select your model.pt path
    model.predict(source='inference/images',
                  imgsz=640,
                  project='runs/detect',
                  name='exp',
                  save=True,
                  # conf=0.2,
                  # iou=0.7,
                  # agnostic_nms=True,
                  # visualize=True, # visualize model features maps
                  # line_width=2, # line width of the bounding boxes
                  # show_conf=False, # do not show prediction confidence
                  # show_labels=False, # do not show prediction labels
                  # save_txt=True, # save results as .txt file
                  # save_crop=True, # save cropped images with results
                )










