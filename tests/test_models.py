"""
수박 당도 예측 모델 통합 테스트

이 스크립트는 데이터셋과 모델의 통합 동작을 검증합니다:
1. 데이터 로딩 및 전처리 테스트
2. 모델 순전파 테스트
3. 손실함수 및 옵티마이저 테스트
4. 간단한 훈련 루프 테스트

Author: AI Assistant
Date: 2024
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.utils.font_utils import setup_korean_font

# 한글 폰트 설정
setup_korean_font()
from typing import Dict, Any, Tuple

# 로컬 모듈 import
from src.data.dataset import WatermelonDataset, get_basic_transforms, create_stratified_split, create_dataloaders
from src.models.cnn_models import ModelFactory, LossManager, OptimizerManager, print_model_summary


def test_dataset_loading():
    """데이터셋 로딩 테스트"""
    print("📊 데이터셋 로딩 테스트")
    print("-" * 40)
    
    try:
        # 기본 변환 설정
        transforms = get_basic_transforms()
        
        # 데이터셋 생성 (새로운 경로)
        dataset = WatermelonDataset(
            root_dir="data/features/melspectrogram_data",
            transform=transforms['train'],
            normalize_targets=True
        )
        
        print(f"✅ 데이터셋 로딩 성공")
        print(f"  - 총 이미지 수: {len(dataset)}")
        print(f"  - Brix 통계: {dataset.get_brix_statistics()}")
        
        # 샘플 데이터 확인
        sample_image, sample_label = dataset[0]
        print(f"  - 샘플 이미지 크기: {sample_image.shape}")
        print(f"  - 샘플 라벨 타입: {type(sample_label)}, 값: {sample_label}")
        
        return dataset
    
    except Exception as e:
        print(f"❌ 데이터셋 로딩 실패: {e}")
        return None


def test_data_splitting(dataset: WatermelonDataset):
    """데이터 분할 테스트"""
    print("\n🔀 데이터 분할 테스트")
    print("-" * 40)
    
    try:
        # 데이터 분할
        train_indices, val_indices, test_indices = create_stratified_split(
            dataset, 
            train_ratio=0.7, 
            val_ratio=0.15, 
            test_ratio=0.15
        )
        
        print(f"✅ 데이터 분할 성공")
        print(f"  - 훈련 데이터: {len(train_indices)}개")
        print(f"  - 검증 데이터: {len(val_indices)}개")
        print(f"  - 테스트 데이터: {len(test_indices)}개")
        
        return train_indices, val_indices, test_indices
    
    except Exception as e:
        print(f"❌ 데이터 분할 실패: {e}")
        return None, None, None


def test_dataloader_creation(dataset: WatermelonDataset, train_indices, val_indices, test_indices):
    """DataLoader 생성 테스트"""
    print("\n📦 DataLoader 생성 테스트")
    print("-" * 40)
    
    try:
        # 변환 설정
        transforms = get_basic_transforms()
        
        # DataLoader 생성
        dataloaders = create_dataloaders(
            dataset=dataset,
            train_indices=train_indices,
            val_indices=val_indices,
            test_indices=test_indices,
            batch_size=4,  # 작은 배치 크기로 테스트
            num_workers=0,  # 멀티프로세싱 비활성화
            train_transforms=transforms['train'],
            val_transforms=transforms['val']
        )
        
        print(f"✅ DataLoader 생성 성공")
        
        # 각 DataLoader 테스트
        for split, dataloader in dataloaders.items():
            print(f"  - {split}: {len(dataloader)}개 배치")
            
            # 첫 번째 배치 테스트
            try:
                batch_images, batch_labels = next(iter(dataloader))
                print(f"    배치 이미지 크기: {batch_images.shape}")
                print(f"    배치 라벨 크기: {batch_labels.shape}")
            except Exception as e:
                print(f"    ❌ 배치 로딩 실패: {e}")
        
        return dataloaders
    
    except Exception as e:
        print(f"❌ DataLoader 생성 실패: {e}")
        return None


def test_model_forward_pass(dataloaders: Dict[str, DataLoader]):
    """모델 순전파 테스트"""
    print("\n🧠 모델 순전파 테스트")
    print("-" * 40)
    
    # 테스트할 모델들
    models_to_test = [
        {"name": "VGG-16", "type": "vgg16", "kwargs": {"dropout": 0.5, "use_pretrained": False}},
        {"name": "커스텀 CNN", "type": "custom_cnn", "kwargs": {"dropout": 0.3, "use_residual": True}}
    ]
    
    test_results = {}
    
    # 테스트 데이터 준비
    train_dataloader = dataloaders['train']
    test_images, test_labels = next(iter(train_dataloader))
    
    for model_info in models_to_test:
        try:
            print(f"\n🔍 {model_info['name']} 테스트")
            
            # 모델 생성
            model = ModelFactory.create_model(model_info['type'], **model_info['kwargs'])
            model.eval()
            
            # 순전파
            with torch.no_grad():
                outputs = model(test_images)
            
            print(f"  ✅ 순전파 성공")
            print(f"    입력 크기: {test_images.shape}")
            print(f"    출력 크기: {outputs.shape}")
            print(f"    예측값 범위: {outputs.min().item():.3f} ~ {outputs.max().item():.3f}")
            
            test_results[model_info['name']] = {
                'model': model,
                'success': True,
                'output_shape': outputs.shape
            }
            
        except Exception as e:
            print(f"  ❌ {model_info['name']} 순전파 실패: {e}")
            test_results[model_info['name']] = {
                'model': None,
                'success': False,
                'error': str(e)
            }
    
    return test_results, (test_images, test_labels)


def test_loss_functions_and_optimizers(model_results: Dict, test_data: Tuple):
    """손실함수 및 옵티마이저 테스트"""
    print("\n⚖️ 손실함수 및 옵티마이저 테스트")
    print("-" * 40)
    
    test_images, test_labels = test_data
    
    # 테스트할 손실함수들
    loss_functions = ['mse', 'mae', 'huber', 'rmse']
    optimizers = ['adam', 'adamw', 'sgd']
    
    # 성공한 모델 중 하나 선택
    successful_model = None
    for name, result in model_results.items():
        if result['success']:
            successful_model = result['model']
            model_name = name
            break
    
    if successful_model is None:
        print("❌ 테스트할 수 있는 모델이 없습니다.")
        return
    
    print(f"🔧 {model_name} 모델로 테스트")
    
    # 손실함수 테스트
    print("\n📉 손실함수 테스트:")
    for loss_type in loss_functions:
        try:
            loss_fn = LossManager.get_loss_function(loss_type)
            
            # 모델 예측
            successful_model.eval()
            with torch.no_grad():
                predictions = successful_model(test_images)
            
            # 손실 계산
            loss = loss_fn(predictions, test_labels.unsqueeze(1))
            print(f"  ✅ {loss_type.upper()}: {loss.item():.4f}")
            
        except Exception as e:
            print(f"  ❌ {loss_type.upper()} 실패: {e}")
    
    # 옵티마이저 테스트
    print("\n🔧 옵티마이저 테스트:")
    for opt_type in optimizers:
        try:
            optimizer = OptimizerManager.get_optimizer(
                successful_model, 
                opt_type, 
                lr=0.001
            )
            print(f"  ✅ {opt_type.upper()}: {optimizer.__class__.__name__}")
            
        except Exception as e:
            print(f"  ❌ {opt_type.upper()} 실패: {e}")


def test_simple_training_loop(dataloaders: Dict[str, DataLoader]):
    """간단한 훈련 루프 테스트"""
    print("\n🏃 간단한 훈련 루프 테스트")
    print("-" * 40)
    
    try:
        # 간단한 모델 생성
        model = ModelFactory.create_model('custom_cnn', dropout=0.3, use_residual=False)
        
        # 손실함수 및 옵티마이저
        criterion = LossManager.get_loss_function('mse')
        optimizer = OptimizerManager.get_optimizer(model, 'adam', lr=0.001)
        
        # 훈련 모드로 설정
        model.train()
        
        # 몇 개 배치로 테스트
        train_dataloader = dataloaders['train']
        num_test_batches = min(3, len(train_dataloader))
        
        print(f"🔄 {num_test_batches}개 배치로 훈련 테스트")
        
        total_loss = 0.0
        for batch_idx, (images, labels) in enumerate(train_dataloader):
            if batch_idx >= num_test_batches:
                break
            
            # 순전파
            outputs = model(images)
            loss = criterion(outputs, labels.unsqueeze(1))
            
            # 역전파
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            print(f"  배치 {batch_idx + 1}: 손실 = {loss.item():.4f}")
        
        avg_loss = total_loss / num_test_batches
        print(f"✅ 훈련 루프 테스트 성공")
        print(f"  평균 손실: {avg_loss:.4f}")
        
    except Exception as e:
        print(f"❌ 훈련 루프 테스트 실패: {e}")


def run_comprehensive_test():
    """종합 테스트 실행"""
    print("🚀 수박 당도 예측 모델 종합 테스트")
    print("=" * 50)
    
    # 1. 데이터셋 로딩 테스트
    dataset = test_dataset_loading()
    if dataset is None:
        print("❌ 데이터셋 로딩 실패로 테스트 중단")
        return
    
    # 2. 데이터 분할 테스트
    train_indices, val_indices, test_indices = test_data_splitting(dataset)
    if train_indices is None:
        print("❌ 데이터 분할 실패로 테스트 중단")
        return
    
    # 3. DataLoader 생성 테스트
    dataloaders = test_dataloader_creation(dataset, train_indices, val_indices, test_indices)
    if dataloaders is None:
        print("❌ DataLoader 생성 실패로 테스트 중단")
        return
    
    # 4. 모델 순전파 테스트
    model_results, test_data = test_model_forward_pass(dataloaders)
    
    # 5. 손실함수 및 옵티마이저 테스트
    test_loss_functions_and_optimizers(model_results, test_data)
    
    # 6. 간단한 훈련 루프 테스트
    test_simple_training_loop(dataloaders)
    
    print("\n" + "=" * 50)
    print("🎉 종합 테스트 완료!")
    
    # 결과 요약
    print("\n📋 테스트 결과 요약:")
    successful_models = [name for name, result in model_results.items() if result['success']]
    failed_models = [name for name, result in model_results.items() if not result['success']]
    
    if successful_models:
        print(f"  ✅ 성공한 모델: {', '.join(successful_models)}")
    if failed_models:
        print(f"  ❌ 실패한 모델: {', '.join(failed_models)}")
    
    print(f"  📊 데이터셋 크기: {len(dataset)}개 이미지")
    print(f"  🔀 데이터 분할: 훈련 {len(train_indices)}, 검증 {len(val_indices)}, 테스트 {len(test_indices)}")


if __name__ == "__main__":
    run_comprehensive_test() 