## [AICUP 2024] Competition-2024-跨相機多目標車輛追蹤競賽
https://tbrain.trendmicro.com.tw/Competitions/Details/33 <br>
Extremely low frame-rate (1 fps) video object tracking challenge
## 2nd place in 跨相機多目標車輛追蹤競賽模型組 and 執行效能獎

https://github.com/user-attachments/assets/20fc4543-e6ec-4d56-9a64-8a7bcaa828c3

https://github.com/user-attachments/assets/a8ea9150-8dc6-4d99-8f0b-ba2ed34229f0


## Technique HighLight
<br>
![image](https://github.com/pdway53/AICUP_ReID_Project/blob/main/BOTSORT.png)

-Tracking Framework : BOTSORT<br>
-Detector : YoloV7<br>
-Day/Night classifier<br>
-ReID : Bag of Tricks(BoT) , Backbone model: Resnet101<br>
-Add vihicle motion predict score on linear assignment<br>

## Setup with Conda
**Step 1.** Create Conda environment and install pytorch.
```shell
conda create -n botsort python=3.7
conda activate botsort
```
**Step 2.** Install torch and matched torchvision from [pytorch.org](https://pytorch.org/get-started/locally/).<br>
The code was tested using torch 1.10.1+cu102 and torchvision==0.11.2

**Step 3.** Fork this Repository and clone your Repository to your device

**Step 4.** **Install numpy first!!**
```shell
pip install numpy
```

**Step 5.** Install `requirement.txt`
```shell
pip install -r requirement.txt
```

**Step 6.** Install [pycocotools](https://github.com/cocodataset/cocoapi).
```shell
pip install cython; pip3 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```

**Step 7.** Others
```shell
# Cython-bbox
pip install cython_bbox

# faiss cpu / gpu
pip install faiss-cpu
pip install faiss-gpu
```




### ReID Model 

For training the ReID, detection patches must be generated as follows:   
```shell
python fast_reid/datasets/generate_AICUP_patches.py --data_path <dataets_dir>/AI_CUP_MCMOT_dataset/train
```

#Train ReID MODEL
```shell
python fast_reid/tools/train_net.py --config-file fast_reid/configs/AICUP/bagtricks_R50-ibn.yml MODEL.DEVICE "cuda:0"
```

The training results are stored by default in ```logs/AICUP/bagtricks_R50-ibn```. The storage location and model hyperparameters can be modified in ```fast_reid/configs/AICUP/bagtricks_R50-ibn.yml```.
You can refer to `fast_reid/fastreid/config/defaults.py` to find out which hyperparameters can be modified.




### YOLOv7 Model

run the `yolov7/tools/AICUP_to_YOLOv7.py` by the following command:
```
cd <BoT-SORT_dir>
python yolov7/tools/AICUP_to_YOLOv7.py --AICUP_dir datasets/AI_CUP_MCMOT_dataset/train --YOLOv7_dir datasets/AI_CUP_MCMOT_dataset/yolo
```
The file tree after conversion by `AICUP_to_YOLOv7.py` is as follows:

```python
/datasets/AI_CUP_MCMOT_dataset/yolo
    ├── train
    │   ├── images
    │   │   ├── 0902_150000_151900_0_00001.jpg (Date_StartTime_EndTime_CamID_FrameNum)
    │   │   ├── 0902_150000_151900_0_00002.jpg
    │   │   ├── ...
    │   │   ├── 0902_150000_151900_7_00001.jpg
    │   │   ├── 0902_150000_151900_7_00002.jpg
    │   │   ├── ...
    │   └── labels
    │   │   ├── 0902_150000_151900_0_00001.txt (Date_StartTime_EndTime_CamID_FrameNum)
    │   │   ├── 0902_150000_151900_0_00002.txt
    │   │   ├── ...
    │   │   ├── 0902_150000_151900_7_00001.txt
    │   │   ├── 0902_150000_151900_7_00002.txt
    │   │   ├── ...
    ├── valid
    │   ├── images
    │   │   ├── 1015_190000_191900_0_00001.jpg (Date_StartTime_EndTime_CamID_FrameNum)
    │   │   ├── 1015_190000_191900_0_00002.jpg
    │   │   ├── ...
    │   │   ├── 1015_190000_191900_7_00001.jpg
    │   │   ├── 1015_190000_191900_7_00002.jpg
    │   │   ├── ...
    │   └── labels
    │   │   ├── 1015_190000_191900_0_00001.txt (Date_StartTime_EndTime_CamID_FrameNum)
    │   │   ├── 1015_190000_191900_0_00002.txt
    │   │   ├── ...
    │   │   ├── 1015_190000_191900_7_00001.txt
    │   │   ├── 1015_190000_191900_7_00002.txt
    │   │   ├── ...
```



the trained models in 'pretrained' folder as follows:
```
<BoT-SORT_dir>/pretrained
```
Final yolov7 trained weight : pretrained/yolov7-w6-AICUP7_049.pt`.
Yolov7 pretrained model : pretrained/yolov7-w6_training.pt `.




#Fine-tune YOLOv7 for AICUP

- The dataset path is configured in `yolov7/data/AICUP.yaml`.
- The model architecture can be configured in `yolov7/cfg/training/yolov7-AICUP.yaml`.
- Training hyperparameters are configured in `yolov7/data/hyp.scratch.custom.yaml` (default is `yolov7/data/hyp.scratch.p5.yaml`).
- pretrained model : pretrained/yolov7-w6_training.pt  from (https://github.com/WongKinYiu/yolov7/releases/download/v0.1/)

```shell
python yolov7/train_aux.py --device "0" --batch-size 4 --epochs 50 --data yolov7/data/AICUP.yaml --img 1280 1280 --cfg yolov7/cfg/training/yolov7-w6-AICUP.yaml --weights 'pretrained/yolov7-w6_training.pt' --name yolov7-w6-AICUP --hyp data/hyp.scratch.custom.yaml
```
The training results will be saved by default at `runs/train`.



### Tracking and creating the submission file for AICUP 
If you want to track all `<timestamps>`testdata the same as submit, you can execute the bash file we provided and clean cnt.txt file to 0. Make sure the ID start from 0
```shell
sh tools/track_all_timestamps2.sh --weights pretrained/yolov7-w6-AICUP7_049.pt --source-dir ./<test_data_dir>/32_33_AI_CUP_testdataset/AI_CUP_testdata/images --device "0" --fast-reid-config "fast_reid/configs/AICUP/bagtricks_R50-ibn.yml" --fast-reid-weights "logs/AICUP_BOT_resnet/bagtricks_R50-ibn/model_0058.pth"
```
The submission file and visualized images will be saved by default at `runs/submit/<timestamp>`.


### Track the spesific timestamp
If you want to track specific `<timestamps>` video data, you can execute the bash as following
```shell
python tools/mc_demo_yolov7_day_night_submit.py --weights "$WEIGHTS" --source "$folder" --device "$DEVICE" --name "$timestamp" --fuse-score --agnostic-nms --with-reid --fast-reid-config "$FAST_REID_CONFIG" --fast-reid-weights "$FAST_REID_WEIGHTS"
```










