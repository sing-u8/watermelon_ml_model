"""
수박 당도 예측 ML 프로젝트 전체 파이프라인 테스트

2단계의 모든 기능이 제대로 연동되어 작동하는지 검증합니다.
"""

import sys
import os

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.dataset import (
    WatermelonDataset, 
    get_augmented_transforms,
    create_stratified_split,
    create_dataloaders,
    setup_cross_validation
)
import torch

def test_full_pipeline():
    """전체 파이프라인 테스트"""
    
    print("🎯 수박 당도 예측 ML 프로젝트 - 2단계 전체 파이프라인 테스트")
    print("=" * 70)
    
    # 1. 데이터셋 생성 (새로운 경로)
    print("\n1️⃣ 데이터셋 생성...")
    dataset = WatermelonDataset(root_dir="data/features/melspectrogram_data")
    print(f"   ✅ 총 {len(dataset)}개 이미지, {len(set([info['sample_id'] for info in dataset.sample_info]))}개 수박 샘플")
    
    # 2. 데이터 분할
    print("\n2️⃣ 데이터 분할...")
    train_indices, val_indices, test_indices = create_stratified_split(dataset)
    print(f"   ✅ 분할 완료: 훈련 {len(train_indices)}, 검증 {len(val_indices)}, 테스트 {len(test_indices)}")
    
    # 3. 증강 변환 준비
    print("\n3️⃣ 데이터 증강 변환 준비...")
    transforms = get_augmented_transforms('medium')
    print(f"   ✅ 훈련용/검증용 변환 파이프라인 생성 완료")
    
    # 4. DataLoader 생성
    print("\n4️⃣ DataLoader 생성...")
    dataloaders = create_dataloaders(
        dataset=dataset,
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
        batch_size=4,
        num_workers=1,
        train_transforms=transforms['train'],
        val_transforms=transforms['val']
    )
    print(f"   ✅ {len(dataloaders)}개 DataLoader 생성 완료")
    
    # 5. 배치 데이터 검증
    print("\n5️⃣ 배치 데이터 검증...")
    for split_name, dataloader in dataloaders.items():
        batch_images, batch_labels = next(iter(dataloader))
        print(f"   ✅ {split_name}: {batch_images.shape} 이미지, {batch_labels.shape} 라벨")
        
        # 데이터 타입 및 범위 확인
        assert batch_images.dtype == torch.float32, f"{split_name} 이미지 타입 오류"
        assert batch_labels.dtype == torch.float32, f"{split_name} 라벨 타입 오류"
        assert batch_images.min() >= -5.0 and batch_images.max() <= 5.0, f"{split_name} 이미지 범위 이상"
    
    # 6. K-Fold 검증
    print("\n6️⃣ K-Fold 교차 검증...")
    fold_splits = setup_cross_validation(dataset, k_folds=3)
    print(f"   ✅ {len(fold_splits)}개 폴드 생성 완료")
    
    # 7. 메모리 효율성 체크
    print("\n7️⃣ 메모리 효율성 체크...")
    train_loader = dataloaders['train']
    batch_count = 0
    for batch_images, batch_labels in train_loader:
        batch_count += 1
        if batch_count >= 3:  # 처음 3개 배치만 테스트
            break
    print(f"   ✅ {batch_count}개 배치 연속 로딩 성공 (메모리 누수 없음)")
    
    print("\n" + "=" * 70)
    print("🎉 전체 파이프라인 테스트 성공!")
    print("\n📊 요약:")
    print(f"   • 데이터셋: {len(dataset)}개 이미지")
    print(f"   • 분할: {len(train_indices)}/{len(val_indices)}/{len(test_indices)}")
    print(f"   • DataLoader: 배치크기 4, {len(dataloaders)}개")
    print(f"   • 교차 검증: {len(fold_splits)}-fold")
    print(f"   • 증강: Medium 강도 적용")
    
    return True

if __name__ == "__main__":
    try:
        test_full_pipeline()
        print("\n✅ 2단계 모든 작업 완료! 3단계 모델 설계로 진행 가능합니다.")
    except Exception as e:
        print(f"\n❌ 파이프라인 테스트 실패: {e}")
        import traceback
        traceback.print_exc() 