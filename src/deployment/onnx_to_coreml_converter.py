"""
ONNX 모델을 Core ML로 변환하는 스크립트

이 모듈은 ONNX 형식의 수박 당도 예측 모델을 Apple Core ML 형식으로 변환합니다.
iOS/macOS 앱에서 직접 사용할 수 있도록 최적화됩니다.

Author: AI Assistant
Date: 2024
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np

# Core ML 변환 도구
import coremltools as ct
from coremltools.models import MLModel

# ONNX 관련
import onnx
from onnx import numpy_helper

# 이미지 처리
from PIL import Image
import cv2


class ONNXToCoreMLConverter:
    """ONNX 모델을 Core ML로 변환하는 클래스"""
    
    def __init__(self, 
                 onnx_model_path: str,
                 output_dir: str = "models/coreml",
                 model_name: str = "WatermelonBrixPredictor"):
        """
        Args:
            onnx_model_path: ONNX 모델 파일 경로
            output_dir: Core ML 모델 저장 디렉토리  
            model_name: Core ML 모델 이름
        """
        self.onnx_model_path = Path(onnx_model_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        
        # 모델 정보 로드
        self.model_info = self._load_model_info()
        
        print(f"ONNX 모델 경로: {self.onnx_model_path}")
        print(f"출력 디렉토리: {self.output_dir}")
        print(f"모델 이름: {self.model_name}")
        
    def _load_model_info(self) -> Dict[str, Any]:
        """모델 정보 JSON 파일 로드"""
        info_path = self.onnx_model_path.parent / f"{self.onnx_model_path.stem}_info.json"
        
        if info_path.exists():
            with open(info_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            print(f"모델 정보 파일을 찾을 수 없습니다: {info_path}")
            # 기본값 반환
            return {
                "input_shape": [1, 3, 224, 224],
                "input_name": "melspectrogram_image",
                "output_name": "brix_prediction",
                "preprocessing": {
                    "normalize": {
                        "mean": [0.485, 0.456, 0.406],
                        "std": [0.229, 0.224, 0.225]
                    }
                }
            }
    
    def _verify_onnx_model(self) -> onnx.ModelProto:
        """ONNX 모델 검증 및 로드"""
        print("\nONNX 모델 검증 중...")
        
        if not self.onnx_model_path.exists():
            raise FileNotFoundError(f"ONNX 모델 파일을 찾을 수 없습니다: {self.onnx_model_path}")
        
        # ONNX 모델 로드
        onnx_model = onnx.load(str(self.onnx_model_path))
        
        # 모델 검증
        try:
            onnx.checker.check_model(onnx_model)
            print("✅ ONNX 모델 검증 통과")
        except Exception as e:
            print(f"⚠️ ONNX 모델 검증 실패: {e}")
            print("변환을 계속 시도합니다...")
        
        # 모델 정보 출력
        print(f"ONNX 모델 정보:")
        print(f"  - IR 버전: {onnx_model.ir_version}")
        print(f"  - 프로듀서: {onnx_model.producer_name} {onnx_model.producer_version}")
        
        # 입력 정보
        for inp in onnx_model.graph.input:
            shape = [dim.dim_value for dim in inp.type.tensor_type.shape.dim]
            print(f"  - 입력: {inp.name}, 형태: {shape}")
        
        # 출력 정보  
        for out in onnx_model.graph.output:
            shape = [dim.dim_value for dim in out.type.tensor_type.shape.dim]
            print(f"  - 출력: {out.name}, 형태: {shape}")
        
        return onnx_model
    
    def convert_to_coreml(self, 
                         compute_precision: str = "FLOAT16",
                         minimum_deployment_target: str = "iOS13") -> MLModel:
        """ONNX 모델을 Core ML로 변환"""
        print("\n" + "="*60)
        print("ONNX → Core ML 변환 시작")
        print("="*60)
        
        # ONNX 모델 검증
        onnx_model = self._verify_onnx_model()
        
        # Core ML 변환
        print("\nCore ML 변환 진행 중...")
        
        try:
            # 배포 타겟 설정
            if minimum_deployment_target == "iOS13":
                target = ct.target.iOS13
            elif minimum_deployment_target == "iOS14":
                target = ct.target.iOS14
            elif minimum_deployment_target == "iOS15": 
                target = ct.target.iOS15
            elif minimum_deployment_target == "iOS16":
                target = ct.target.iOS16
            else:
                target = ct.target.iOS13
            
            # ONNX에서 Core ML로 변환
            coreml_model = ct.convert(
                str(self.onnx_model_path),
                minimum_deployment_target=target,
                compute_precision=ct.precision.FLOAT16 if compute_precision == "FLOAT16" else ct.precision.FLOAT32
            )
            
            print(f"✅ Core ML 변환 성공!")
            
        except Exception as e:
            print(f"❌ Core ML 변환 실패: {e}")
            print("다른 설정으로 재시도 중...")
            
            # 대안 방법: 더 기본적인 설정으로 시도
            try:
                coreml_model = ct.convert(str(self.onnx_model_path))
                print(f"✅ 기본 설정으로 Core ML 변환 성공!")
            except Exception as e2:
                raise Exception(f"모든 변환 방법 실패: {e2}")
        
        # 메타데이터 추가
        self._add_metadata(coreml_model)
        
        # 모델 검증
        self._validate_coreml_model(coreml_model)
        
        return coreml_model
    
    def _add_metadata(self, coreml_model: MLModel):
        """Core ML 모델에 메타데이터 추가"""
        print("\n메타데이터 추가 중...")
        
        # 기본 정보
        coreml_model.short_description = "수박 당도(Brix) 예측 AI 모델 - ONNX 변환"
        coreml_model.author = "WatermelonAI Team"
        coreml_model.license = "MIT License"
        coreml_model.version = "1.0.0"
        
        # 입력/출력 이름 가져오기
        input_name = None
        output_name = None
        
        for inp in coreml_model.get_spec().description.input:
            input_name = inp.name
            break
            
        for out in coreml_model.get_spec().description.output:
            output_name = out.name
            break
        
        # 입력 설명
        if input_name:
            coreml_model.input_description[input_name] = (
                "수박 타격음에서 추출한 멜-스펙트로그램 이미지 (224x224 RGB). "
                "ImageNet 표준으로 정규화된 상태여야 함."
            )
        
        # 출력 설명
        if output_name:
            coreml_model.output_description[output_name] = (
                "예측된 수박 당도 값 (Brix). "
                "일반적으로 8.0-13.0 범위의 값을 가짐."
            )
        
        # 사용법 메타데이터
        coreml_model.user_defined_metadata["preprocessing_info"] = (
            "1. 이미지 크기를 224x224로 조정 "
            "2. RGB 채널로 변환 "
            "3. [0,1]로 정규화 후 ImageNet 평균/표준편차로 정규화: "
            f"mean={self.model_info['preprocessing']['normalize']['mean']}, "
            f"std={self.model_info['preprocessing']['normalize']['std']}"
        )
        
        coreml_model.user_defined_metadata["model_source"] = "ONNX 모델에서 변환됨"
        coreml_model.user_defined_metadata["original_model"] = str(self.onnx_model_path)
        
        # 성능 정보 (JSON에서 가져오기)
        if "performance" in self.model_info:
            perf = self.model_info["performance"]
            coreml_model.user_defined_metadata["performance"] = (
                f"RMSE: {perf.get('rmse', 'N/A')}, "
                f"당도 정확도(±1.0 Brix): {perf.get('accuracy_1_brix', 'N/A')}%"
            )
        
        print("✅ 메타데이터 추가 완료")
    
    def _validate_coreml_model(self, coreml_model: MLModel):
        """Core ML 모델 검증"""
        print("\nCore ML 모델 검증 중...")
        
        try:
            # 모델 스펙 가져오기
            spec = coreml_model.get_spec()
            
            print(f"모델 유형: {spec.WhichOneof('Type')}")
            print(f"입력 개수: {len(spec.description.input)}")
            print(f"출력 개수: {len(spec.description.output)}")
            
            # 입력 정보 출력
            for inp in spec.description.input:
                print(f"  입력: {inp.name}")
                if inp.type.WhichOneof('Type') == 'multiArrayType':
                    print(f"    - 타입: MultiArray")
                    print(f"    - 형태: {list(inp.type.multiArrayType.shape)}")
                    print(f"    - 데이터 타입: {inp.type.multiArrayType.dataType}")
                elif inp.type.WhichOneof('Type') == 'imageType':
                    print(f"    - 타입: Image")
                    print(f"    - 폭: {inp.type.imageType.width}")
                    print(f"    - 높이: {inp.type.imageType.height}")
            
            # 출력 정보 출력
            for out in spec.description.output:
                print(f"  출력: {out.name}")
                if out.type.WhichOneof('Type') == 'multiArrayType':
                    print(f"    - 타입: MultiArray")
                    print(f"    - 형태: {list(out.type.multiArrayType.shape)}")
                    print(f"    - 데이터 타입: {out.type.multiArrayType.dataType}")
            
            print("✅ Core ML 모델 검증 완료")
            
        except Exception as e:
            print(f"⚠️ Core ML 모델 검증 중 오류: {e}")
    
    def save_model(self, coreml_model: MLModel, filename: Optional[str] = None) -> str:
        """Core ML 모델 저장"""
        if filename is None:
            filename = f"{self.model_name}.mlmodel"
        
        output_path = self.output_dir / filename
        
        print(f"\nCore ML 모델 저장 중: {output_path}")
        
        try:
            coreml_model.save(str(output_path))
            
            # 파일 크기 확인
            file_size = output_path.stat().st_size / (1024 * 1024)  # MB
            print(f"✅ 모델 저장 완료!")
            print(f"   파일 경로: {output_path}")
            print(f"   파일 크기: {file_size:.2f} MB")
            
            return str(output_path)
            
        except Exception as e:
            print(f"❌ 모델 저장 실패: {e}")
            raise
    
    def test_model_inference(self, coreml_model: MLModel, num_tests: int = 3):
        """Core ML 모델 추론 테스트"""
        print(f"\nCore ML 모델 추론 테스트 ({num_tests}회)...")
        
        # 테스트 입력 생성
        input_shape = self.model_info["input_shape"]
        test_input = np.random.randn(*input_shape).astype(np.float32)
        
        # ImageNet 정규화 적용
        mean = np.array(self.model_info["preprocessing"]["normalize"]["mean"])
        std = np.array(self.model_info["preprocessing"]["normalize"]["std"])
        
        # 정규화 (C, H, W 형식)
        for c in range(3):
            test_input[0, c] = (test_input[0, c] - mean[c]) / std[c]
        
        inference_times = []
        predictions = []
        
        try:
            # 입력 이름 가져오기
            input_name = None
            for inp in coreml_model.get_spec().description.input:
                input_name = inp.name
                break
            
            if input_name is None:
                print("❌ 입력 이름을 찾을 수 없습니다.")
                return
            
            for i in range(num_tests):
                start_time = time.time()
                
                # Core ML 예측
                prediction = coreml_model.predict({input_name: test_input})
                
                inference_time = (time.time() - start_time) * 1000  # ms
                inference_times.append(inference_time)
                
                # 예측값 추출
                pred_value = list(prediction.values())[0]
                if isinstance(pred_value, np.ndarray):
                    pred_value = pred_value.flatten()[0]
                predictions.append(pred_value)
                
                print(f"  테스트 {i+1}: {pred_value:.4f} Brix, {inference_time:.2f}ms")
            
            # 통계 출력
            avg_time = np.mean(inference_times)
            avg_pred = np.mean(predictions)
            std_pred = np.std(predictions)
            
            print(f"\n📊 추론 성능 통계:")
            print(f"   평균 추론 시간: {avg_time:.2f}ms")
            print(f"   평균 예측값: {avg_pred:.4f} Brix")
            print(f"   예측값 표준편차: {std_pred:.4f}")
            
        except Exception as e:
            print(f"❌ 추론 테스트 실패: {e}")


def main():
    """메인 실행 함수"""
    print("🍉 ONNX → Core ML 변환 도구")
    print("="*50)
    
    # ONNX 모델 경로
    onnx_model_path = "models/onnx/WatermelonBrixPredictor.onnx"
    
    # 변환기 생성
    converter = ONNXToCoreMLConverter(
        onnx_model_path=onnx_model_path,
        output_dir="models/coreml",
        model_name="WatermelonBrixPredictor"
    )
    
    try:
        # Core ML 변환
        coreml_model = converter.convert_to_coreml(
            compute_precision="FLOAT16",
            minimum_deployment_target="iOS13"
        )
        
        # 모델 저장
        output_path = converter.save_model(coreml_model)
        
        # 추론 테스트
        converter.test_model_inference(coreml_model)
        
        print(f"\n🎉 변환 완료!")
        print(f"Core ML 모델: {output_path}")
        print(f"\n📱 iOS/macOS 앱에서 사용할 수 있습니다.")
        
    except Exception as e:
        print(f"\n❌ 변환 실패: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 