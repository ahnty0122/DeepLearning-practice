## OpenPose Toyproject
* __Clone openpose__

```
git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose.git
```
1. clone 후 models 파일에 getModels 파일 실행
2. 코드 실행 시엔 models -> pose -> mpi 의 'pose_deploy_linevec_faster_4_stages.prototxt' 와 'pose_iter_160000.caffemodel' 만 필요

* __OpenCV Ver 3.4.1 이상 다운__
```
pip install opencv-python
```

* __OpenPose.py 실행__

## YOLO Object Detection 구현
* Object detection with YOLOv3

## ToFNetRGB
* ToF 카메라 이미지로 Net 모델 test
* Data augmentation, transform 과정 O

## mnistNet
#### VGG & ResNet으로 구현

## dog VS cat classfication
* 간단한 keras cnn model로 binary classfication 구현
* train & test

## Fashion mnist
#### 옷 이미지 분류 신경망 모델 train
* tensorflow에서 fashion mnist dataset import
* dataset에 class 지정 안 되어 있음 -> 변수 만들어 저장
* image pixel 값 전처리 (0-1 사이로 값 조정)
* train & test

## SRCNN
#### 저해상도 image 입력하면 고해상도 image로 출력
* hyperparameter tuning
* PSNR(Peak Signal to Noise Ratio)과 구조적 유사성 SSIM(Structural Similarith)으로 성능 평가
