## OpenPose Toyproject
1. Clone openpose

```
git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose.git
```
* clone 후 models 파일에 getModels 파일 실행
* 코드 실행 시엔 models -> pose -> mpi 의 'pose_deploy_linevec_faster_4_stages.prototxt' 와 'pose_iter_160000.caffemodel' 만 필요

2. OpenCV Ver 3.4.1 이상 다운
```
pip install opencv-python
```

3. OpenPose.py 실행

## YOLO Object Detection 구현
* Object detection with YOLOv3

## ToFNetRGB
* ToF 카메라 이미지로 Net 모델 test
* Data augmentation, transform 과정 O

## mnistNet
#### VGG & ResNet으로 구현

## dog VS cat classfication
1. 간단한 keras cnn model로 binary classfication 구현
2. train & test

## Fashion mnist
#### 옷 이미지 분류 신경망 모델 train
* tensorflow에서 fashion mnist dataset import
* dataset에 class 지정 안 되어 있음 -> 변수 만들어 저장
* image pixel 값 전처리 (0-1 사이로 값 조정)
* train & test
