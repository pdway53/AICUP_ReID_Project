import argparse
import time
from pathlib import Path
import sys

import cv2
import torch
import torch.backends.cudnn as cudnn

from tqdm import tqdm
from numpy import random

import numpy

sys.path.insert(0, './yolov7')
sys.path.append('.')

from yolov7.models.experimental import attempt_load
from yolov7.utils.datasets import LoadStreams, LoadImages
from yolov7.utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, \
    apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from yolov7.utils.plots import plot_one_box,plot_one_box_det,plot_one_box_debug
from yolov7.utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

from tracker.mc_bot_sort_day_night import BoTSORT_ADJUST
from tracker.tracking_utils.timer import Timer


import logging
log = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setStream(tqdm) # <-- important
handler = log.addHandler(handler)



def day_classifier(im0s):

    gray_img = cv2.cvtColor(im0s,cv2.COLOR_BGR2GRAY)
    hsv_image = cv2.cvtColor(im0s, cv2.COLOR_BGR2HSV)
    brightness = hsv_image[:,:,2]
    #获取灰度图矩阵的行数和列数
    r,c = gray_img.shape[:2]
    piexs_sum=r*c #整个图的像素个数
    #遍历灰度图的所有像素
    #灰度值小于60被认为是黑
    dark_points = (gray_img < 60)    
    #hist = cv2.calcHist([gray_img],[0],None,[256],[0,256])
    #cv2.imwrite("hist/" + id,hist)
    #filename = "hist/" + id
        #write histogram
    '''
    #fig = plt.figure()
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
    '''
    target_array = gray_img[dark_points]
    dark_sum = target_array.size #偏暗的像素
    dark_prop=dark_sum/(piexs_sum) #偏暗像素所占比例
    #print("dark_sum {}".format(dark_sum))
    #print("dark_prop {}".format(dark_prop))
    #if dark_prop <=0.2: #若偏暗像素所占比例超过0.6,认为为整体环境黑暗的图片
        #cnt += 1
    # 计算亮度的平均值
    avg_brightness = numpy.mean(brightness)
    #print("avg_brightness {}".format(avg_brightness))
    #if avg_brightness > 108:
        #cnt2 += 1
        
    if avg_brightness > 108 and dark_prop < 0.25:
        return 1
    else:
        return 0
        


def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, opt.trace
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    opt.project ="runs/submit/ori"
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(100)]

    with open('cnt2.txt', 'r') as f:
        lines = f.readlines()
    
    if len(lines) == 0:
        prev_cnt = 0
    else:
        prev_cnt = int(lines[0])
    #tracker = BoTSORT(opt, frame_rate, prev_cnt)
    '''
    if day :
        tracker = BoTSORT_TEST(opt, frame_rate=10.0) #frame_rate=30.0)
    else:
        tracker = BoTSORT_NIGHT(opt, frame_rate=10.0) #frame_rate=30.0)
    '''
    frame_rate =30
    tracker = BoTSORT_NIGHT(opt, frame_rate,prev_cnt) #frame_rate=30.0)

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        
    t0 = time.time()
    
    # Process detections
    results = []
    frameID = 0

    for path, img, im0s, vid_cap in tqdm(dataset, desc=f'tracking {opt.name}'):
        frameID += 1
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

         # day or night
        #day = 1   night = 0
        day = day_classifier(im0s)

        # Create tracker
        '''
        if day :
            tracker = BoTSORT_TEST(opt, frame_rate=10.0) #frame_rate=30.0)
        else:
            tracker = BoTSORT_NIGHT(opt, frame_rate=10.0,day) #frame_rate=30.0)
        '''
        if frameID == 5:
            print("test")
        #    continue
        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            # Run tracker
            detections = []
            if len(det):
                boxes = scale_coords(img.shape[2:], det[:, :4], im0.shape)                
                boxes = boxes.cpu().numpy()
                detections = det.cpu().numpy()
                detections[:, :4] = boxes

            detections_car = []

            for detect in detections:
                cls = detect[5]
                label = f' { names[int(cls)] }'
                if 'car' in label:   
                    detections_car.append(detect)


            detections_car = numpy.array(detections_car)
            online_targets = tracker.update(detections_car, im0, frameID, day,opt)
            #add
            for detect in detections:
                cls = detect[5]
                label = f' { names[int(cls)] }'
                if 'car' in label:   
                    #detections_car.append(detect)
                    scores = detect[4]
                    score = float("{:.2f}".format(scores))
                    label = f'{str(score)}'
                    tlbr = bboxes = detect[:4]
                    plot_one_box_det(tlbr, im0, label=label, color=[255,255,255], line_thickness=4) 
            
            


            online_tlwhs = []
            online_ids = []
            online_scores = []
            online_cls = []
            for t in online_targets:
                tlwh = t.tlwh
                tlbr = t.tlbr
                tid = t.track_id
                tcls = t.cls
                #tscore = t.score
                if tlwh[2] * tlwh[3] > opt.min_box_area:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)
                    online_cls.append(t.cls)

                    if save_img or view_img:  # Add bbox to image
                        if opt.hide_labels_name:
                            label = f'{tid}, {int(tcls)}'
                        else:
                            label = f'{tid}, {names[int(tcls)]}'
                        
                        if 'car' in label: # AICUP only have one cls: car
                            # save results
                            results.append(
                                f"{frameID},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                            )
                            score = float("{:.2f}".format(t.score))
                            label = f'{tid}, {str(score)}'
                            match_score =f'{ t.match_score:.2f}'
                            #plot_one_box(tlbr, im0, label=label, color=colors[int(tid) % len(colors)], line_thickness=2)
                            plot_one_box_debug(tlbr, im0, label=label, color=colors[int(tid) % len(colors)], line_thickness=2,score = match_score)
                            #plot_one_box(tlbr, im0, label=label, color=colors[int(tid) % len(colors)], line_thickness=2)

                            
            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg

            # Print time (inference + NMS)
            # print(f'{s}Done. ({t2 - t1:.3f}s)')
            if day :
                lightness = 'day'
            else:
                lightness = 'night'
            cv2.putText(im0, lightness, (20, 20 ), 0, 2, [225, 0, 0], thickness=2, lineType=cv2.LINE_AA)
            # Stream results
            if view_img:
                cv2.imshow('BoT-SORT', im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    cnt = tracker.id_cnt
    print("cnt  : ")
    print(cnt)
    with open("cnt2.txt", 'w') as f:
        f.writelines(str(cnt))

    if save_txt or save_img:
        save_dir1 = 'runs/submit/ori/'
        with open(save_dir1 + f"{opt.name}.txt", 'w') as f:
            f.writelines(results)
        print(f"Results saved to {save_dir}")

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=1920, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.09, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.7, help='IOU threshold for NMS')
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
    parser.add_argument("--track_high_thresh", type=float, default=0.2, help="tracking confidence threshold") #0.4
    parser.add_argument("--track_low_thresh", default=0.05, type=float, help="lowest detection threshold")
    parser.add_argument("--new_track_thresh", default=0.18, type=float, help="new track thresh")  #0.4
    
    #night
    parser.add_argument("--track_high_thresh_night", type=float, default=0.12, help="tracking confidence threshold") #0.4
    parser.add_argument("--track_low_thresh_night", default=0.05, type=float, help="lowest detection threshold")
    parser.add_argument("--new_track_thresh_night", default=0.12, type=float, help="new track thresh")  #0.4
    parser.add_argument("--match_thresh_night", type=float, default=0.45, help="matching threshold for tracking") #default=0.35
    parser.add_argument("--match_thresh2_night", type=float, default=0.8, help="matching threshold for tracking") #default=0.3
    #add  match_thresh_2
    parser.add_argument("--match_thresh2", type=float, default=0.7, help="matching threshold for tracking") #default=0.3


    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.32, help="matching threshold for tracking") #default=0.35
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