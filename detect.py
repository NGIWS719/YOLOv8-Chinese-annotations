# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.
（说明了yolov5可以检测哪些类型的资源）
Usage - sources:（使用pytorch权重检测不同类型的资源的命令）
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:（使用不同版本的权重进行检测）
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""

import argparse  # 用于命令项选项与参数解析
import os  # 通用的、基本的操作系统交互功能
import platform  # 用于访问底层平台——即操作系统
import sys  # 供对一些Python解释器使用或维护的变量和函数的访问
from pathlib import Path  # 提供了面向对象的文件系统路径操作

import torch  # PyTorch库的主要模块

FILE = Path(__file__).resolve()  # 获取当前Python脚本文件的绝对路径
ROOT = FILE.parents[0]  # 当前脚本文件绝对路径的上一级（即项目的根路径）
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # 将脚本所在的目录添加到Python的模块搜索路径中
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # 更新ROOT变量为相对于当前工作目录的路径

# 系列导包
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode


# 根据Pytorch版本选择PyTorch的自动梯度计算（这个装饰器的作用）
@smart_inference_mode()
def run(
        weights=ROOT / 'yolov5s.pt',  # 模型权重路径
        source=ROOT / 'data/images',  # 检测的资源路径
        data=ROOT / 'data/coco128.yaml',  # 数据的配置文件路径
        imgsz=(640, 640),  # 推理的图片尺寸 (height, width)
        conf_thres=0.25,  # 置信度阈值
        iou_thres=0.45,  # NMS IOU 阈值
        max_det=1000,  # 预测图片的最大检测目标数
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu（推理设备）
        view_img=False,  # 是否在屏幕上展示结果
        save_txt=False,  # 保存 *.txt类型的结果文件
        save_conf=False,  # 将置信度（confidences）保存在文本标签（labels）中
        save_crop=False,  # 保存裁剪下来的预测框图片
        nosave=False,  # 不保存预测后的图片或视频
        classes=None,  # （去除某类别的框）: --class 0, or --class 0 2 3
        agnostic_nms=False,  # 是否进行class-agnostic NMS
        augment=False,  # 数据增强
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # 保存检测结果的目录
        name='exp',  # 预测结果的子目录 project/name
        exist_ok=False,  # 保存结果子目录增量命名
        line_thickness=3,  # 画框的线条粗细
        hide_labels=False,  # 是否隐藏标签
        hide_conf=False,  # 是否隐藏置信度
        half=False,  # 是否使用 FP16 半精度推理
        dnn=False,  # 是否使用 OpenCV DNN for ONNX 推理
        vid_stride=1,  # 视频帧率步长
):
    source = str(source)  # 将检测资源路径转为字符串
    save_img = not nosave and not source.endswith('.txt')  # 保存检测的图片
    # 判断检测资源的后缀名是否存在于图片/视频后缀名列表中
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    # 判断检测资源的前缀是否含有以下元组中的字符串，要是有则是网络资源路径
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    # 判断资源路径是否是数字，是否以.txt结尾的文件，是否是一个URL，但不是一个文件，意味着源是一个网络摄像头的URL
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    # 判断资源路径开头，是否是从屏幕截图中加载图像数据
    screenshot = source.lower().startswith('screen')
    # 既包含网络资源开头后包含图片或视频结尾，说明是一个网络上的图片或视频
    if is_url and is_file:
        source = check_file(source)  # 下载这个网络资源

    # 文件目录处理
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # 保存检测后的文件采用增量目录（exp1,exp2...）
    # 如果save_txt为真，就创建save_dir/labels目录；否则就创建save_dir目录。
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)

    # 加载模型
    device = select_device(device)  # 选择检测的设备（gpu或cpu）
    # 根据weights后缀选择用于执行推理的模型类
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    # 模型步长，模型名称，模型权重
    stride, names, pt = model.stride, model.names, model.pt
    # 检查图像的尺寸是否是步长stride的倍数。如果不是，调整图像的尺寸
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    # 网络摄像头的URL
    if webcam:
        # 检查系统是否支持显示图像
        view_img = check_imshow(warn=True)
        # 加载视频流数据
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        # 视频流的数量
        bs = len(dataset)
    # 是否是从屏幕截图中加载图像数据
    elif screenshot:
        # 从屏幕截图中加载图像
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        # 加载图片或视频数据
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # 运行推理
    # 预热的目的是为了在实际运行模型之前，让模型进入一个就绪的状态，这样可以避免在实际运行模型时由于初始化等操作导致的延迟
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    # seen用于记录已经处理过的帧的数量
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    # 遍历dataset中的每一帧图像
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            # 将图像从NumPy数组转换为PyTorch张量，然后将其移动到模型所在的设备
            im = torch.from_numpy(im).to(model.device)
            # 根据模型是否使用半精度浮点数（FP16），将图像的数据类型转换为FP16或者FP32
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0  归一化
            # 如果图像只有三个维度（也就是说，它不是一个批次的图像，而是一个单独的图像），那么就在批次维度上添加一个维度
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # 推理预测
        with dt[1]:
            # 如果visualize参数为True，那么就在保存目录下创建一个与当前文件名相同的目录，用于保存可视化的结果
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            # 使用非极大值抑制（NMS）处理预测的结果
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # 处理预测结果
        for i, det in enumerate(pred):  # 遍历每个预测结果中的检测到的目标
            seen += 1  # 更新已处理的帧数seen
            if webcam:  # batch_size >= 1
                # 获取当前帧的路径\图像\编号
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:  # 图片或视频
                # 尝试从dataset获取当前帧的编号，如果dataset没有frame属性(说明检测的是图片)，那么就使用0作为默认值
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # 转为Path对象
            # 构造图像的保存路径
            save_path = str(save_dir / p.name)  # im.jpg
            # 构造文本文件的保存路径,根据dataset.mode决定是否添加帧编号frame
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            # 图像的高度和宽度添加到字符串s的末尾
            s += '%gx%g ' % im.shape[2:]  # print string
            # 创建一个表示归一化因子的张量gn
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            # 如果设置了保存裁剪的图像，那么就复制图像im0，否则就直接使用im
            imc = im0.copy() if save_crop else im0  # for save_crop
            # 用于在图像上添加边界框和标签
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):  # 有检测目标
                # Rescale boxes from img_size to im0 size
                # 将预测的边界框的坐标从模型的输入尺寸转换到原始图像的尺寸
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # 打印结果
                for c in det[:, 5].unique():  # 遍历预测结果det中的每一个唯一的类别
                    n = (det[:, 5] == c).sum()  # 统计当前这个类别的数量
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # 写结果
                for *xyxy, conf, cls in reversed(det):  # 对于每一个检测目标获取坐标,置信度和类别
                    # 如果设置了保存文本
                    if save_txt:  # Write to file
                        # 将边界框的坐标和类别写入到文本文件中。这里，边界框的坐标被转换为了归一化的中心坐标和宽高形式
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    # 如果设置了保存图像或者查看图像，那么就将边界框和对应的标签添加到图像上
                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    # 如果设置了保存裁剪的图像，那么就将边界框对应的图像区域保存为一个新的图像
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # 视频流结果
            im0 = annotator.result()  # 获取添加了边界框和标签的图像
            # 如果设置了查看图像
            if view_img:
                # 当前的操作系统是Linux，并且还没有为当前的文件创建窗口，那么就创建一个新的窗口，然后调整窗口的大小以适应图像的尺寸
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                # 在窗口中显示图像
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # 保存结果 (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    # 将检测后的图像保存至save_path
                    cv2.imwrite(save_path, im0)
                else:  # 如果是视频或者视频流
                    if vid_path[i] != save_path:  # 新视频
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # 释放之前的视频写入器
                        if vid_cap:  # 视频
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)  # 帧率
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))  #髋
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 高
                        else:  # 视频流
                            fps, w, h = 30, im0.shape[1], im0.shape[0]  # 帧率,宽,高
                        # 生成的检测视频保存路径
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    # 写入视频
                    vid_writer[i].write(im0)

        # 打印推理时间
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # 打印结果
    t = tuple(x.t / seen * 1E3 for x in dt)  # 计算每个阶段的平均处理速度，单位是毫秒/帧
    # 打印每个阶段的处理速度和图像的尺寸
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    # 如果设置了保存文本或者保存图像，那么就打印保存的结果。如果设置了保存文本，那么就打印保存的标签的数量和保存的目录
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    # 如果设置了更新模型,为了修复PyTorch在加载模型时可能出现的源代码改变的警告
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


def parse_opt():
    """
        opt参数解析
        weights: 模型的权重地址 默认 weights/best.pt
        source: 测试数据文件(图片或视频)的保存路径 默认data/images
        imgsz: 网络输入图片的大小 默认640
        conf-thres: object置信度阈值 默认0.25
        iou-thres: 做nms的iou阈值 默认0.45
        max-det: 每张图片最大的目标个数 默认1000
        device: 设置代码执行的设备 cuda device, i.e. 0 or 0,1,2,3 or cpu
        view-img: 是否展示预测之后的图片或视频 默认False
        save-txt: 是否将预测的框坐标以txt文件格式保存 默认True 会在runs/detect/expn/labels下生成每张图片预测的txt文件
        save-conf: 是否保存预测每个目标的置信度到预测tx文件中 默认True
        save-crop: 是否需要将预测到的目标从原图中扣出来 剪切好 并保存 会在runs/detect/expn下生成crops文件，将剪切的图片保存在里面  默认False
        nosave: 是否不要保存预测后的图片  默认False 就是默认要保存预测后的图片
        classes: 在nms中是否是只保留某些特定的类 默认是None 就是所有类只要满足条件都可以保留
        agnostic-nms: 进行nms是否也除去不同类别之间的框 默认False
        augment: 预测是否也要采用数据增强 TTA
        update: 是否将optimizer从ckpt中删除  更新模型  默认False
        project: 当前测试结果放在哪个主文件夹下 默认runs/detect
        name: 当前测试结果放在run/detect下的文件名  默认是exp
        exist-ok: 是否存在当前文件 默认False 一般是 no exist-ok 连用  所以一般都要重新创建文件夹
        line-thickness: 画框的框框的线宽  默认是 3
        hide-labels: 画出的框框是否需要隐藏label信息 默认False
        hide-conf: 画出的框框是否需要隐藏conf信息 默认False
        half: 是否使用半精度 Float16 推理 可以缩短推理时间 但是默认是False
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()  # 解析命令行参数，得到一个命名空间对象opt
    # 如果opt.imgsz只有一个元素，那么就将它扩展为两个元素,因为图像的尺寸需要两个元素，分别表示高度和宽度
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    # 将opt转换为字典，然后使用print_args函数打印所有的参数
    print_args(vars(opt))
    return opt


def main(opt):
    # 检查除了'tensorboard'和'thop'之外的所有依赖项是否已经安装
    check_requirements(exclude=('tensorboard', 'thop'))
    # 调用run函数，并将opt转换为字典作为参数
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()  # 命令行参数解析
    main(opt)
