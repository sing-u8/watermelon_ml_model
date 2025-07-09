# 🍉 수박 당도 예측 ONNX 모델

## 📝 개요
이 모델은 수박 타격음에서 추출한 멜-스펙트로그램을 분석하여 수박의 당도(Brix)를 예측합니다.

## 📊 성능
- **RMSE**: 0.75
- **당도 정확도 (±1.0 Brix)**: 85.2%
- **모델 크기**: 4.8 MB

## 🚀 사용 방법

### 1. 필요 패키지 설치
```bash
pip install onnxruntime pillow torchvision
```

### 2. 모델 로드 및 예측
```python
import onnxruntime as ort
import numpy as np

# 모델 로드
session = ort.InferenceSession("WatermelonBrixPredictor.onnx")

# 이미지 전처리 (224x224 RGB, ImageNet 정규화)
# input_data = preprocess_your_image()

# 예측 실행
input_dict = {"melspectrogram_image": input_data}
output = session.run(None, input_dict)
predicted_brix = output[0][0][0]
```

## 📱 배포 플랫폼
- Windows, Linux, macOS
- 모바일 (Android, iOS via ONNX Runtime)
- 웹 (ONNX.js)
- 임베디드 시스템

## 📋 입력 요구사항
- **형식**: 멜-스펙트로그램 이미지
- **크기**: 224×224 픽셀
- **채널**: RGB (3채널)
- **정규화**: ImageNet 표준 (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])

## 📈 출력
- **형식**: Float 값
- **범위**: 일반적으로 8.0-13.0 Brix
- **의미**: 수박의 당도 (Brix 단위)

## 📧 문의
WatermelonAI Team
