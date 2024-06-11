import sys
sys.path.append('tools/py_motmetrics')

import os
import argparse
#import py_motmetrics.motmetrics as mm

from loguru import logger
'''
def evaluate(gt_dir, ts_dir, mode):
    metrics = list(mm.metrics.motchallenge_metrics)
    mh = mm.metrics.create()

    if mode == 'multi_cam':
        accs = []
        names = []

        gt_files = sorted(os.listdir(gt_dir))
        ts_files = sorted(os.listdir(ts_dir))

        for gt_file, ts_file in zip(gt_files, ts_files):
            gt_path = os.path.join(gt_dir, gt_file)
            ts_path = os.path.join(ts_dir, ts_file)

            # compare the same title files
            if os.path.splitext(gt_file)[0] == os.path.splitext(ts_file)[0]:
                gt = mm.io.loadtxt(gt_path, fmt="mot15-2D", min_confidence=1)
                ts = mm.io.loadtxt(ts_path, fmt="mot15-2D")
                names.append(os.path.splitext(os.path.basename(ts_path))[0])
                accs.append(mm.utils.compare_to_groundtruth(gt, ts, 'iou', distth=0.5)) # ground truth 覆蓋率 > 0.5 才是對的

        summary = mh.compute_many(accs, metrics=metrics, generate_overall=True)
        print("Score for total file: IDF1 ", summary.idf1.OVERALL, " + MOTA ", summary.mota.OVERALL, " = ", summary.idf1.OVERALL+summary.mota.OVERALL)
        # 完整的 motmetrics 各項評分成果 V
        logger.info(f'\n{mm.io.render_summary(summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names)}')

    elif mode == 'single_cam':
        gt_files = sorted(os.listdir(gt_dir))
        ts_files = sorted(os.listdir(ts_dir))

        for gt_file, ts_file in zip(gt_files, ts_files):
            gt_path = os.path.join(gt_dir, gt_file)
            ts_path = os.path.join(ts_dir, ts_file)

            # compare the same title files
            if os.path.splitext(gt_file)[0] == os.path.splitext(ts_file)[0]:
                gt = mm.io.loadtxt(gt_path, fmt="mot15-2D", min_confidence=1)
                ts = mm.io.loadtxt(ts_path, fmt="mot15-2D")
                name = os.path.splitext(os.path.basename(ts_path))[0]
                acc = mm.utils.compare_to_groundtruth(gt, ts, 'iou', distth=0.5) # ground truth 覆蓋率 > 0.5 才是對的

                summary = mh.compute_many([acc], metrics=metrics, generate_overall=True)
                print(f"Score for {name}: IDF1 {summary.idf1.OVERALL}, MOTA {summary.mota.OVERALL}")
'''

#
def convert_to_gt():

    label_path = "../../train/labels/"
    glabel_files = sorted(os.listdir(label_path))
    outputpath = "./gt/"
    img_width = 1280
    img_width = 720
    frame_ID= 0
    for file in glabel_files:
        name = file.split(".")[0]
        print(name)
        out_filename = outputpath + name + '.txt'
        f = open(out_filename , 'w')
        gt_path = os.path.join(label_path, file)      

        gt_files_all = sorted(os.listdir(gt_path))

        for gt_file in gt_files_all:
            frame_ID= 0
            file1 = open(gt_path + '/' + gt_file, 'r')
            name_id = gt_file.split("_")[1].lstrip("0")
            Lines = file1.readlines()
            frame_ID =  name_id.split(".")[0]
            for line in Lines:
                center_x = float(line.strip().split(" ")[1])
                center_y = float(line.strip().split(" ")[2])
                width_scale = float(line.strip().split(" ")[3])
                height_scale = float(line.strip().split(" ")[4])
                track_id = line.strip().split(" ")[5]
                bbox_w = width_scale*1280
                bbox_h = height_scale*720
                bbox_left = center_x - bbox_w/2 
                bbox_top = center_y - bbox_h/2 
                conf = 1

                outline = [frame_ID,",",str(track_id),",", str(bbox_left),",",str(bbox_top),",",str(bbox_w),",",str(bbox_h),",",str(conf),",",str(-1) ,",",str(-1),",",str(-1),"\n",]
                f.writelines(outline)


        f.close()










if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate multiple object tracking results.')
    #parser.add_argument('--gt_dir', type=str, help='Path to the ground truth directory')
    #parser.add_argument('--ts_dir', type=str, help='Path to the tracking result directory')
    #parser.add_argument('--mode', type=str, choices=['multi_cam', 'single_cam'], default='multi_cam', help='Evaluation mode')

    #args = parser.parse_args()

    #evaluate(args.gt_dir, args.ts_dir, args.mode)
    convert_to_gt()