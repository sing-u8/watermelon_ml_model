"""
수박 당도 예측 ONNX 모델 사용 예제

이 스크립트는 ONNX 런타임을 사용하여 수박 당도를 예측하는 방법을 보여줍니다.
"""

import numpy as np
import onnxruntime as ort
from PIL import Image
import torchvision.transforms as transforms

def preprocess_image(image_path):
    """이미지 전처리"""
    # 이미지 로드 및 RGB 변환
    image = Image.open(image_path).convert('RGB')
    
    # 전처리 파이프라인
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # 배치 차원 추가
    input_tensor = preprocess(image).unsqueeze(0).numpy()
    return input_tensor

def predict_brix(model_path, image_path):
    """당도 예측"""
    # ONNX 세션 생성
    session = ort.InferenceSession(model_path)
    
    # 이미지 전처리
    input_data = preprocess_image(image_path)
    
    # 예측 실행
    input_dict = {'melspectrogram_image': input_data}
    output = session.run(None, input_dict)
    
    # 결과 반환
    predicted_brix = output[0][0][0]
    return predicted_brix

# 사용 예제
if __name__ == "__main__":
    model_path = "WatermelonBrixPredictor.onnx"
    image_path = "sample_melspectrogram.png"
    
    try:
        brix_value = predict_brix(model_path, image_path)
        print(f"예측된 당도: {brix_value:.2f} Brix")
    except Exception as e:
        print(f"예측 실패: {e}")
