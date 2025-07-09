"""
간단한 Core ML 변환 스크립트
PyTorch 모델을 Core ML로 변환하고 .mlmodel 형식으로 저장

Author: AI Assistant
Date: 2024
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

# Core ML 변환 도구
import coremltools as ct

# 프로젝트 모듈
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.models.cnn_models import WatermelonCNN


def load_pytorch_model(checkpoint_path: str) -> nn.Module:
    """PyTorch 모델 로드"""
    print(f"PyTorch 모델 로딩: {checkpoint_path}")
    
    # 모델 생성
    model = WatermelonCNN(
        input_channels=3,
        num_classes=1,
        dropout=0.3,
        use_residual=True
    )
    
    # 체크포인트 로드
    device = torch.device("cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"체크포인트 정보: Epoch {checkpoint.get('epoch', 'N/A')}")
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    print("✅ PyTorch 모델 로딩 완료")
    return model


def convert_to_coreml(pytorch_model: nn.Module) -> ct.models.MLModel:
    """PyTorch 모델을 Core ML로 변환"""
    print("\n🔄 Core ML 변환 시작...")
    
    # 예시 입력 생성
    example_input = torch.randn(1, 3, 224, 224)
    
    # PyTorch 모델 테스트
    with torch.no_grad():
        pytorch_output = pytorch_model(example_input)
        print(f"PyTorch 예측값: {pytorch_output.item():.4f}")
    
    # TorchScript로 변환
    print("TorchScript 변환 중...")
    traced_model = torch.jit.trace(pytorch_model, example_input)
    
    # Core ML 변환 (간단한 설정)
    print("Core ML 변환 중...")
    try:
        coreml_model = ct.convert(
            traced_model,
            inputs=[ct.TensorType(name="melspectrogram_image", shape=example_input.shape)],
            outputs=[ct.TensorType(name="brix_prediction")],
            convert_to="neuralnetwork",  # 호환성을 위해 neuralnetwork 형식 사용
            minimum_deployment_target=ct.target.iOS13
        )
        print("✅ Core ML 변환 성공!")
        return coreml_model
        
    except Exception as e:
        print(f"❌ Core ML 변환 실패: {e}")
        print("다른 설정으로 재시도...")
        
        # 더 기본적인 설정으로 재시도
        coreml_model = ct.convert(traced_model)
        print("✅ 기본 설정으로 Core ML 변환 성공!")
        return coreml_model


def add_metadata(coreml_model: ct.models.MLModel):
    """메타데이터 추가"""
    print("메타데이터 추가 중...")
    
    coreml_model.short_description = "수박 당도(Brix) 예측 AI 모델"
    coreml_model.author = "WatermelonAI Team"
    coreml_model.license = "MIT License"
    coreml_model.version = "1.0.0"
    
    # 입력/출력 설명
    spec = coreml_model.get_spec()
    if len(spec.description.input) > 0:
        input_name = spec.description.input[0].name
        coreml_model.input_description[input_name] = (
            "수박 타격음에서 추출한 멜-스펙트로그램 이미지 (224x224 RGB). "
            "ImageNet 표준으로 정규화된 상태여야 함."
        )
    
    if len(spec.description.output) > 0:
        output_name = spec.description.output[0].name
        coreml_model.output_description[output_name] = (
            "예측된 수박 당도 값 (Brix). 일반적으로 8.0-13.0 범위의 값을 가짐."
        )
    
    print("✅ 메타데이터 추가 완료")


def save_coreml_model(coreml_model: ct.models.MLModel, output_path: str) -> bool:
    """Core ML 모델 저장"""
    print(f"\n💾 Core ML 모델 저장 중: {output_path}")
    
    try:
        # .mlmodel 형식으로 저장 (더 호환성이 좋음)
        if not output_path.endswith('.mlmodel'):
            output_path = output_path.replace('.mlpackage', '.mlmodel')
        
        coreml_model.save(output_path)
        
        # 파일 크기 확인
        file_size = Path(output_path).stat().st_size / (1024 * 1024)  # MB
        print(f"✅ 모델 저장 완료!")
        print(f"   파일 경로: {output_path}")
        print(f"   파일 크기: {file_size:.2f} MB")
        return True
        
    except Exception as e:
        print(f"❌ 저장 실패: {e}")
        
        # 대안: 다른 경로로 시도
        try:
            alt_path = output_path.replace('.mlmodel', '_fallback.mlmodel')
            coreml_model.save(alt_path)
            print(f"✅ 대안 경로로 저장 성공: {alt_path}")
            return True
        except Exception as e2:
            print(f"❌ 대안 저장도 실패: {e2}")
            return False


def test_coreml_model(coreml_model: ct.models.MLModel):
    """Core ML 모델 테스트"""
    print("\n🧪 Core ML 모델 테스트 중...")
    
    try:
        # 테스트 입력 생성
        test_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
        
        # 입력 이름 가져오기
        spec = coreml_model.get_spec()
        input_name = spec.description.input[0].name
        
        # 예측 실행
        prediction = coreml_model.predict({input_name: test_input})
        pred_value = list(prediction.values())[0]
        
        if isinstance(pred_value, np.ndarray):
            pred_value = pred_value.flatten()[0]
            
        print(f"✅ Core ML 예측 성공: {pred_value:.4f} Brix")
        return True
        
    except Exception as e:
        print(f"⚠️ Core ML 테스트 실패 (환경 문제일 수 있음): {e}")
        print("iOS/macOS 기기에서 테스트하세요.")
        return False


def main():
    """메인 함수"""
    print("🍉 수박 당도 예측 모델 → Core ML 변환")
    print("=" * 50)
    
    # 설정
    checkpoint_path = "models/checkpoints/custom_cnn/WatermelonCNN_best.pth"
    output_dir = Path("models/coreml")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "WatermelonBrixPredictor.mlmodel"
    
    try:
        # 1. PyTorch 모델 로드
        pytorch_model = load_pytorch_model(checkpoint_path)
        
        # 2. Core ML 변환
        coreml_model = convert_to_coreml(pytorch_model)
        
        # 3. 메타데이터 추가
        add_metadata(coreml_model)
        
        # 4. 모델 저장
        if save_coreml_model(coreml_model, str(output_path)):
            # 5. 모델 테스트
            test_coreml_model(coreml_model)
            
            print(f"\n🎉 변환 완료!")
            print(f"Core ML 모델: {output_path}")
            print(f"\n📱 이제 Swift 프로젝트에서 사용할 수 있습니다!")
            
            # Swift 사용법 안내
            print(f"\n📋 Swift에서 사용하는 방법:")
            print(f"1. {output_path} 파일을 Xcode 프로젝트에 추가")
            print(f"2. import CoreML")
            print(f"3. let model = try! WatermelonBrixPredictor(configuration: MLModelConfiguration())")
            print(f"4. let prediction = try! model.prediction(melspectrogram_image: pixelBuffer)")
            
        else:
            print("❌ 변환 실패")
            
    except Exception as e:
        print(f"❌ 전체 프로세스 실패: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 