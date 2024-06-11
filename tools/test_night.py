import argparse
import time
from pathlib import Path
import sys

import cv2
#import torch
#import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
from tqdm import tqdm
from numpy import random

import numpy

sys.path.insert(0, './yolov7')
sys.path.append('.')

#from yolov7.models.experimental import attempt_load
from yolov7.utils.datasets import LoadStreams, LoadImages
from yolov7.utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, \
    apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from yolov7.utils.plots import plot_one_box,plot_one_box_det
from yolov7.utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

from tracker.mc_bot_sort import BoTSORT
from tracker.mc_bot_sort_rule import BoTSORT_TEST
from tracker.tracking_utils.timer import Timer

import logging
log = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setStream(tqdm) # <-- important
handler = log.addHandler(handler)

def tlbr_to_tlwh(tlbr):
        ret = numpy.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, opt.trace
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    #model = attempt_load(weights, map_location=device)  # load FP32 model
    #stride = int(model.stride.max())  # model stride
    #imgsz = check_img_size(imgsz, s=stride)  # check img_size

 
    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        #cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=32)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=32)

    # Get names and colors
    #names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(100)]

    # Create tracker
    #tracker = BoTSORT_TEST(opt, frame_rate=10.0) #frame_rate=30.0)
    #
    #tracker = BoTSORT(opt, frame_rate=30.0) #frame_rate=30.0)
    frameID = 0
    cnt=0
    cnt2=0
    for path, img, im0s, vid_cap in tqdm(dataset, desc=f'tracking {opt.name}'):
        #cv2.imwrite("img.jpg",img)
        #cv2.imwrite("im0s.jpg",im0s)
        frameID += 1
        if frameID%30 != 0:
            continue
        print(path)
        id = path.split("\\")[-1]
        print(id)
        gray_img = cv2.cvtColor(im0s,cv2.COLOR_BGR2GRAY)
        hsv_image = cv2.cvtColor(im0s, cv2.COLOR_BGR2HSV)
        brightness = hsv_image[:,:,2]
        #获取灰度图矩阵的行数和列数
        r,c = gray_img.shape[:2]
        piexs_sum=r*c #整个图的像素个数
        #遍历灰度图的所有像素
        #灰度值小于60被认为是黑
        dark_points = (gray_img < 60)
        print(frameID)        
        hist = cv2.calcHist([gray_img],[0],None,[256],[0,256])
        #cv2.imwrite("hist/" + id,hist)
        filename = "hist/" + id

        #write histogram
        
        #fig = plt.figure()
        '''
        plt.subplot(121)
        plt.imshow(gray_img,'gray')
        plt.xticks([])
        plt.yticks([])
        plt.title("Original")
        plt.subplot(122)
        plt.hist(brightness.ravel(),256,[0,256])
        plt.show()
        plt.savefig(filename)
        plt.cla()        
        f, axarr = plt.subplots(1,2)
        plt.figure()
        axarr[0].imshow(gray_img)
        axarr[1].hist(gray_img.ravel(),256,[0,256])
        #axarr.show()
        plt.savefig(filename)
        '''
        target_array = gray_img[dark_points]
        dark_sum = target_array.size #偏暗的像素
        dark_prop=dark_sum/(piexs_sum) #偏暗像素所占比例
        print("dark_sum {}".format(dark_sum))
        print("dark_prop {}".format(dark_prop))
        if dark_prop <=0.2: #若偏暗像素所占比例超过0.6,认为为整体环境黑暗的图片
            cnt += 1
        # 计算亮度的平均值
        avg_brightness = numpy.mean(brightness)
        print("avg_brightness {}".format(avg_brightness))
        if avg_brightness > 110:
            cnt2 += 1

        filename = "hist/test/" + id

        if avg_brightness > 105 and dark_prop < 0.2:
            day =1

        if day :
            lightness = 'day'
        else:
            lightness = 'night'

        cv2.putText(im0s, lightness, (15, 15 ), 0, 3, [225, 0, 0], thickness=2, lineType=cv2.LINE_AA)
        cv2.imwrite(filename,im0s)
    print(cnt/frameID)    
    print(cnt2/frameID)        



    # Run inference
    #if device.type != 'cpu':
    #    model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        
    t0 = time.time()
    
    # Process detections
    results = []
    frameID = 0


    #plot_one_box(tlbr, im0, label=label, color=colors[int(tid) % len(colors)], line_thickness=2)
    if save_txt or save_img:
        with open(save_dir / f"{opt.name}.txt", 'w') as f:
            f.writelines(results)
            
        print(f"Results saved to {save_dir}")

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=1920, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.09, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.1, help='IOU threshold for NMS') #0.7
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--trace', action='store_true', help='trace model')
    parser.add_argument('--hide-labels-name', default=False, action='store_true', help='hide labels')

    # tracking args
    parser.add_argument("--track_high_thresh", type=float, default=0.1, help="tracking confidence threshold") #0.3
    parser.add_argument("--track_low_thresh", default=0.05, type=float, help="lowest detection threshold")
    parser.add_argument("--new_track_thresh", default=0.05, type=float, help="new track thresh") #0.4
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking") #0.35
    parser.add_argument("--aspect_ratio_thresh", type=float, default=1.6,
                        help="threshold for filtering out boxes of which aspect ratio are above the given value.")
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--fuse-score", dest="mot20", default=False, action="store_true",
                        help="fuse score and iou for association")

    # CMC
    parser.add_argument("--cmc-method", default="sparseOptFlow", type=str, help="cmc method: sparseOptFlow | files (Vidstab GMC) | orb | ecc")

    # ReID
    parser.add_argument("--with-reid", dest="with_reid", default=False, action="store_true", help="with ReID module.")
    parser.add_argument("--fast-reid-config", dest="fast_reid_config", default=r"fast_reid/configs/MOT17/sbs_S50.yml",
                        type=str, help="reid config file path")
    parser.add_argument("--fast-reid-weights", dest="fast_reid_weights", default=r"pretrained/mot17_sbs_S50.pth",
                        type=str, help="reid config file path")
    parser.add_argument('--proximity_thresh', type=float, default=0.5,
                        help='threshold for rejecting low overlap reid matches')
    parser.add_argument('--appearance_thresh', type=float, default=0.5,
                        help='threshold for rejecting low appearance similarity reid matches')

    opt = parser.parse_args()

    opt.jde = False
    opt.ablation = False

    print(opt)
    # check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
