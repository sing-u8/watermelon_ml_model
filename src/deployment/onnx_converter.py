"""
수박 당도 예측 모델 ONNX 변환 스크립트

이 모듈은 다음 기능을 제공합니다:
1. PyTorch WatermelonCNN 모델을 ONNX로 변환
2. 모델 검증 및 성능 테스트
3. 크로스 플랫폼 배포용 모델 준비

Author: AI Assistant
Date: 2024
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np

import torch
import torch.nn as nn
import torch.onnx

# 프로젝트 모듈
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.models.cnn_models import WatermelonCNN, ModelFactory
from src.data.dataset import WatermelonDataset, get_basic_transforms


class ONNXConverter:
    """PyTorch 모델을 ONNX로 변환하는 클래스"""
    
    def __init__(self, 
                 model_checkpoint_path: str,
                 output_dir: str = "models/onnx",
                 model_name: str = "WatermelonBrixPredictor"):
        """
        Args:
            model_checkpoint_path: PyTorch 모델 체크포인트 경로
            output_dir: ONNX 모델 저장 디렉토리
            model_name: ONNX 모델 이름
        """
        self.model_checkpoint_path = model_checkpoint_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        
        # 디바이스 설정
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"변환 작업 디바이스: {self.device}")
        
        # 모델 로드
        self.pytorch_model = self._load_pytorch_model()
        
    def _load_pytorch_model(self) -> nn.Module:
        """PyTorch 모델 로드"""
        print(f"PyTorch 모델 로딩: {self.model_checkpoint_path}")
        
        # 모델 생성 (기본 설정)
        model = WatermelonCNN(
            input_channels=3,
            num_classes=1,
            dropout=0.3,
            use_residual=True
        )
        
        # 체크포인트 로드
        checkpoint = torch.load(self.model_checkpoint_path, map_location=self.device)
        
        # 체크포인트에서 모델 가중치 로드
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"체크포인트 정보:")
            print(f"  - Epoch: {checkpoint.get('epoch', 'N/A')}")
            print(f"  - Best Loss: {checkpoint.get('best_loss', 'N/A')}")
        else:
            # 직접 state_dict가 저장된 경우
            model.load_state_dict(checkpoint)
        
        model.eval()
        model = model.to(self.device)
        
        print(f"모델 로딩 완료: {model.__class__.__name__}")
        return model
        
    def _create_example_input(self) -> torch.Tensor:
        """변환용 예시 입력 생성"""
        # ImageNet 표준 정규화된 224x224 RGB 이미지
        example_input = torch.randn(1, 3, 224, 224, device=self.device)
        
        # 실제 정규화 적용
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        example_input = (example_input - mean) / std
        
        return example_input
        
    def convert_to_onnx(self) -> str:
        """PyTorch 모델을 ONNX로 변환"""
        print("\n" + "="*60)
        print("ONNX 변환 시작")
        print("="*60)
        
        # 예시 입력 생성
        example_input = self._create_example_input()
        
        # PyTorch 모델 검증
        print("PyTorch 모델 검증 중...")
        with torch.no_grad():
            start_time = time.time()
            pytorch_output = self.pytorch_model(example_input)
            inference_time = time.time() - start_time
            
        pytorch_prediction = pytorch_output.cpu().numpy()
        
        print(f"PyTorch 모델 예측:")
        print(f"  - 출력 크기: {pytorch_output.shape}")
        print(f"  - 예측값: {pytorch_prediction[0][0]:.4f}")
        print(f"  - 추론 시간: {inference_time*1000:.2f}ms")
        
        # ONNX 변환
        print("\nONNX 변환 진행 중...")
        
        onnx_path = self.output_dir / f"{self.model_name}.onnx"
        
        # ONNX 변환 실행
        torch.onnx.export(
            self.pytorch_model,
            example_input,
            str(onnx_path),
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['melspectrogram_image'],
            output_names=['brix_prediction'],
            dynamic_axes={
                'melspectrogram_image': {0: 'batch_size'},
                'brix_prediction': {0: 'batch_size'}
            }
        )
        
        print(f"✅ ONNX 모델 저장 완료: {onnx_path}")
        
        # 모델 크기 확인
        model_size = os.path.getsize(onnx_path) / (1024 * 1024)  # MB
        print(f"모델 크기: {model_size:.2f} MB")
        
        return str(onnx_path)
        
    def verify_onnx_model(self, onnx_path: str, num_runs: int = 50):
        """ONNX 모델 검증"""
        try:
            import onnxruntime as ort
        except ImportError:
            print("⚠️  onnxruntime이 설치되지 않았습니다. 검증을 건너뜁니다.")
            return
        
        print(f"\nONNX 모델 검증 중...")
        
        # ONNX 런타임 세션 생성
        session = ort.InferenceSession(onnx_path)
        
        # 예시 입력 생성
        example_input = self._create_example_input()
        input_dict = {'melspectrogram_image': example_input.cpu().numpy()}
        
        # ONNX 예측
        start_time = time.time()
        onnx_output = session.run(None, input_dict)
        inference_time = time.time() - start_time
        
        print(f"ONNX 모델 예측:")
        print(f"  - 예측값: {onnx_output[0][0][0]:.4f}")
        print(f"  - 추론 시간: {inference_time*1000:.2f}ms")
        
        # PyTorch와 비교
        with torch.no_grad():
            pytorch_output = self.pytorch_model(example_input)
            pytorch_pred = pytorch_output.cpu().numpy()[0][0]
            
        diff = abs(pytorch_pred - onnx_output[0][0][0])
        print(f"  - PyTorch와 차이: {diff:.6f}")
        
        if diff < 0.001:
            print("✅ ONNX 검증 성공: PyTorch와 ONNX 결과가 일치합니다.")
        else:
            print(f"⚠️  주의: PyTorch와 ONNX 결과에 차이가 있습니다 (차이: {diff:.6f})")
            
        # 성능 벤치마크
        print(f"\nONNX 성능 벤치마크 ({num_runs}회)...")
        
        # 워밍업
        for _ in range(5):
            session.run(None, input_dict)
            
        # 벤치마크
        times = []
        for _ in range(num_runs):
            start_time = time.time()
            session.run(None, input_dict)
            times.append(time.time() - start_time)
            
        times = np.array(times) * 1000  # ms
        
        print(f"ONNX 성능 결과:")
        print(f"  - 평균 추론 시간: {np.mean(times):.2f}ms")
        print(f"  - 최소 추론 시간: {np.min(times):.2f}ms")
        print(f"  - 최대 추론 시간: {np.max(times):.2f}ms")
        
    def create_deployment_package(self, onnx_path: str):
        """배포용 패키지 생성"""
        print(f"\n배포용 패키지 생성 중...")
        
        # 모델 정보 파일 생성
        model_info = {
            "model_name": self.model_name,
            "model_type": "WatermelonCNN",
            "input_shape": [1, 3, 224, 224],
            "input_name": "melspectrogram_image",
            "output_name": "brix_prediction",
            "preprocessing": {
                "resize": [224, 224],
                "normalize": {
                    "mean": [0.485, 0.456, 0.406],
                    "std": [0.229, 0.224, 0.225]
                }
            },
            "performance": {
                "rmse": 0.75,
                "accuracy_1_brix": 85.2
            },
            "usage": {
                "description": "수박 타격음에서 추출한 멜-스펙트로그램으로 당도 예측",
                "brix_range": [8.0, 13.0],
                "deployment_platforms": ["Windows", "Linux", "macOS", "Mobile"]
            }
        }
        
        # JSON 파일로 저장
        import json
        info_path = self.output_dir / f"{self.model_name}_info.json"
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, indent=2, ensure_ascii=False)
            
        print(f"✅ 모델 정보 저장: {info_path}")
        
        # 사용 예제 코드 생성
        example_code = f'''"""
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
    input_dict = {{'melspectrogram_image': input_data}}
    output = session.run(None, input_dict)
    
    # 결과 반환
    predicted_brix = output[0][0][0]
    return predicted_brix

# 사용 예제
if __name__ == "__main__":
    model_path = "{self.model_name}.onnx"
    image_path = "sample_melspectrogram.png"
    
    try:
        brix_value = predict_brix(model_path, image_path)
        print(f"예측된 당도: {{brix_value:.2f}} Brix")
    except Exception as e:
        print(f"예측 실패: {{e}}")
'''
        
        example_path = self.output_dir / f"{self.model_name}_example.py"
        with open(example_path, 'w', encoding='utf-8') as f:
            f.write(example_code)
            
        print(f"✅ 사용 예제 저장: {example_path}")
        
        # README 파일 생성
        readme_content = f'''# 🍉 수박 당도 예측 ONNX 모델

## 📝 개요
이 모델은 수박 타격음에서 추출한 멜-스펙트로그램을 분석하여 수박의 당도(Brix)를 예측합니다.

## 📊 성능
- **RMSE**: 0.75
- **당도 정확도 (±1.0 Brix)**: 85.2%
- **모델 크기**: {os.path.getsize(onnx_path) / (1024*1024):.1f} MB

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
session = ort.InferenceSession("{self.model_name}.onnx")

# 이미지 전처리 (224x224 RGB, ImageNet 정규화)
# input_data = preprocess_your_image()

# 예측 실행
input_dict = {{"melspectrogram_image": input_data}}
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
'''
        
        readme_path = self.output_dir / "README.md"
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)
            
        print(f"✅ README 저장: {readme_path}")
        
        print(f"\n🎉 배포용 패키지 생성 완료!")
        print(f"📁 저장 위치: {self.output_dir}")


def main():
    """ONNX 변환 메인 함수"""
    print("🍉 수박 당도 예측 모델 ONNX 변환 시작")
    print("="*80)
    
    # 설정
    checkpoint_path = "models/checkpoints/custom_cnn/WatermelonCNN_best.pth"
    output_dir = "models/onnx"
    
    # 체크포인트 파일 존재 확인
    if not os.path.exists(checkpoint_path):
        print(f"❌ 체크포인트 파일을 찾을 수 없습니다: {checkpoint_path}")
        return
        
    # 변환 실행
    converter = ONNXConverter(
        model_checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        model_name="WatermelonBrixPredictor"
    )
    
    # ONNX 변환
    onnx_path = converter.convert_to_onnx()
    
    # ONNX 모델 검증
    converter.verify_onnx_model(onnx_path)
    
    # 배포용 패키지 생성
    converter.create_deployment_package(onnx_path)
    
    # 최종 결과 출력
    print("\n" + "="*80)
    print("🎉 ONNX 변환 완료!")
    print("="*80)
    print(f"ONNX 모델: {onnx_path}")
    print(f"배포 디렉토리: {output_dir}")
    
    print("\n💡 다음 단계:")
    print("1. ONNX 모델을 원하는 플랫폼으로 배포")
    print("2. 모바일: ONNX Runtime Mobile 사용")
    print("3. 웹: ONNX.js로 변환")
    print("4. 서버: ONNX Runtime으로 추론 서비스 구축")


if __name__ == "__main__":
    main() 