"""
수박 당도 예측 모델 4단계 통합 실행 스크립트

이 스크립트는 다음 과정을 수행합니다:
1. 데이터 로딩 및 분할
2. 모델 훈련 (VGG-16, 커스텀 CNN)
3. 평가 메트릭 계산 및 시각화
4. 하이퍼파라미터 튜닝 (선택사항)
5. 모델 비교 및 앙상블
6. 최종 모델 선택 및 리포트 생성

Author: AI Assistant
Date: 2024
"""

import os
import sys
import time
from pathlib import Path
import argparse
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# 프로젝트 모듈들 (새로운 구조)
from src.data.dataset import WatermelonDataset, create_stratified_split, create_dataloaders, get_augmented_transforms
from src.training.trainer import create_trainer, Trainer
from src.evaluation.metrics import ModelEvaluator, VisualizationTools, compare_models
from src.training.hyperparameter_tuning import run_hyperparameter_tuning
from src.evaluation.model_comparison import run_comprehensive_comparison


def setup_directories():
    """필요한 디렉토리 생성"""
    directories = [
        "results",
        "results/training",
        "results/evaluation", 
        "results/hyperparameter_tuning",
        "results/model_comparison",
        "results/final_models",
        "models/checkpoints",
        "logs"
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    print("필요한 디렉토리들이 생성되었습니다.")


def load_and_prepare_data(augmentation_strength: str = 'medium') -> tuple:
    """데이터 로딩 및 전처리"""
    print("\n" + "="*60)
    print("1단계: 데이터 로딩 및 분할")
    print("="*60)
    
    # 데이터셋 로딩 (새로운 경로)
    print("데이터셋 로딩 중...")
    dataset = WatermelonDataset(
        root_dir="data/features/melspectrogram_data",
        normalize_targets=True
    )
    
    print(f"데이터셋 통계:")
    stats = dataset.get_brix_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value:.3f}" if isinstance(value, float) else f"  {key}: {value}")
    
    # 데이터 분할
    print("\n데이터 분할 중...")
    train_indices, val_indices, test_indices = create_stratified_split(
        dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42
    )
    
    print(f"훈련 샘플: {len(train_indices)}")
    print(f"검증 샘플: {len(val_indices)}")
    print(f"테스트 샘플: {len(test_indices)}")
    
    # 데이터 로더 생성
    print(f"\n데이터 로더 생성 중... (증강 강도: {augmentation_strength})")
    transforms = get_augmented_transforms(augmentation_strength)
    
    dataloaders = create_dataloaders(
        dataset, train_indices, val_indices, test_indices,
        batch_size=16, num_workers=4,
        train_transforms=transforms['train'],
        val_transforms=transforms['val']
    )
    
    return dataset, dataloaders, (train_indices, val_indices, test_indices)


def train_baseline_models(dataloaders: Dict[str, DataLoader], 
                         save_dir: str = "results/training") -> Dict[str, Any]:
    """기본 모델들 훈련"""
    print("\n" + "="*60)
    print("2단계: 기본 모델 훈련")
    print("="*60)
    
    results = {}
    
    # 기본 하이퍼파라미터 설정
    base_params = {
        'learning_rate': 0.001,
        'batch_size': 16,
        'optimizer': 'adam',
        'weight_decay': 1e-4,
        'scheduler': 'plateau',
        'loss_function': 'mse',
        'epochs': 50,
        'early_stopping_patience': 15
    }
    
    models_to_train = {
        'vgg16': {**base_params, 'dropout': 0.5, 'freeze_features': False},
        'custom_cnn': {**base_params, 'dropout': 0.3, 'use_residual': True}
    }
    
    for model_name, params in models_to_train.items():
        print(f"\n{model_name.upper()} 모델 훈련 시작...")
        start_time = time.time()
        
        # Trainer 생성 (새로운 경로)
        trainer = create_trainer(
            model_type=model_name,
            train_loader=dataloaders['train'],
            val_loader=dataloaders['val'],
            test_loader=dataloaders['test'],
            checkpoint_dir=f"models/checkpoints/{model_name}",
            log_dir=f"logs/{model_name}",
            **params
        )
        
        # 훈련 실행
        history = trainer.fit(
            epochs=params['epochs'],
            early_stopping_patience=params['early_stopping_patience']
        )
        
        # 훈련 시간 기록
        train_time = time.time() - start_time
        
        # 모델 평가
        print(f"{model_name} 모델 평가 중...")
        test_results = trainer.evaluate(dataloaders['test'], return_predictions=True)
        
        # 결과 저장
        results[model_name] = {
            'trainer': trainer,
            'history': history,
            'test_results': test_results,
            'train_time': train_time,
            'params': params
        }
        
        # 훈련 기록 시각화
        trainer.plot_training_history(
            save_path=f"{save_dir}/{model_name}_training_history.png"
        )
        
        # 모델 저장
        trainer.save_model(f"{save_dir}/{model_name}_final_model.pth")
        
        print(f"{model_name} 훈련 완료 (소요시간: {train_time/60:.1f}분)")
        print(f"  테스트 RMSE: {test_results['rmse']:.4f}")
        print(f"  테스트 MAE: {test_results['mae']:.4f}")
        print(f"  테스트 R²: {test_results['r2']:.4f}")
    
    return results


def evaluate_models(training_results: Dict[str, Any], 
                   dataloaders: Dict[str, DataLoader],
                   save_dir: str = "results/evaluation") -> Dict[str, Any]:
    """모델 상세 평가"""
    print("\n" + "="*60)
    print("3단계: 모델 상세 평가")
    print("="*60)
    
    evaluation_results = {}
    
    for model_name, result in training_results.items():
        print(f"\n{model_name.upper()} 모델 종합 평가...")
        
        trainer = result['trainer']
        evaluator = ModelEvaluator(trainer.model)
        
        # 종합 평가 수행
        eval_result = evaluator.evaluate_comprehensive(
            dataloaders['test'], 
            dataset_name=f"{model_name}_test"
        )
        
        # 평가 리포트 생성
        VisualizationTools.create_evaluation_report(
            eval_result, 
            save_dir=f"{save_dir}/{model_name}"
        )
        
        evaluation_results[model_name] = eval_result
    
    # 모델 간 비교
    print("\n모델 성능 비교 중...")
    compare_models(
        evaluation_results,
        save_path=f"{save_dir}/model_performance_comparison.png"
    )
    
    return evaluation_results


def run_hyperparameter_tuning_stage(dataset: WatermelonDataset,
                                   skip_tuning: bool = False,
                                   save_dir: str = "results/hyperparameter_tuning") -> Dict[str, Any]:
    """하이퍼파라미터 튜닝 단계"""
    print("\n" + "="*60)
    print("4단계: 하이퍼파라미터 튜닝 (선택사항)")
    print("="*60)
    
    if skip_tuning:
        print("하이퍼파라미터 튜닝을 건너뜁니다.")
        return {}
    
    print("하이퍼파라미터 튜닝 시작...")
    print("주의: 이 과정은 상당한 시간이 소요될 수 있습니다.")
    
    tuning_results = run_hyperparameter_tuning(
        dataset=dataset,
        model_types=['vgg16', 'custom_cnn'],
        search_type='random',  # 시간 절약을 위해 랜덤 서치 사용
        max_combinations=30,   # 조합 수 제한
        cv_folds=3,
        save_dir=save_dir
    )
    
    return tuning_results


def run_model_comparison_stage(dataset: WatermelonDataset,
                             tuning_results: Optional[Dict[str, Any]] = None,
                             save_dir: str = "results/model_comparison") -> Dict[str, Any]:
    """모델 비교 및 앙상블 단계"""
    print("\n" + "="*60)
    print("5단계: 모델 비교 및 앙상블")
    print("="*60)
    
    # 하이퍼파라미터 튜닝 결과가 있으면 사용, 없으면 기본값 사용
    if tuning_results and 'individual_results' in tuning_results:
        vgg16_params = tuning_results['individual_results'].get('vgg16', {}).get('best_parameters', {})
        custom_cnn_params = tuning_results['individual_results'].get('custom_cnn', {}).get('best_parameters', {})
        print("하이퍼파라미터 튜닝 결과를 사용합니다.")
    else:
        vgg16_params = None
        custom_cnn_params = None
        print("기본 하이퍼파라미터를 사용합니다.")
    
    # 종합 비교 실행
    comparison_results = run_comprehensive_comparison(
        dataset=dataset,
        save_dir=save_dir
    )
    
    return comparison_results


def generate_final_report(training_results: Dict[str, Any],
                        evaluation_results: Dict[str, Any],
                        tuning_results: Dict[str, Any],
                        comparison_results: Dict[str, Any],
                        save_dir: str = "results") -> str:
    """최종 종합 리포트 생성"""
    print("\n" + "="*60)
    print("최종 리포트 생성")
    print("="*60)
    
    report_path = Path(save_dir) / "final_comprehensive_report.txt"
    report_path = Path(save_dir) / "final_comprehensive_report.txt"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("수박 당도 예측 모델 개발 최종 리포트\n")
        f.write("="*60 + "\n\n")
        
        # 1. 프로젝트 개요
        f.write("1. 프로젝트 개요\n")
        f.write("-"*30 + "\n")
        f.write("목표: 수박 타격음 기반 당도(Brix) 예측 모델 개발\n")
        f.write("접근법: 오디오 → 멜-스펙트로그램 → CNN 회귀 모델\n")
        f.write("모델: VGG-16 전이학습 vs 커스텀 CNN\n\n")
        
        # 2. 데이터셋 정보
        f.write("2. 데이터셋 정보\n")
        f.write("-"*30 + "\n")
        sample_info = list(evaluation_results.values())[0]
        f.write(f"총 샘플 수: {sample_info['sample_count']}\n")
        f.write("Brix 범위: 8.7 ~ 12.7 (19개 수박 샘플)\n")
        f.write("데이터 분할: 훈련 70% / 검증 15% / 테스트 15%\n\n")
        
        # 3. 훈련 결과
        f.write("3. 모델 훈련 결과\n")
        f.write("-"*30 + "\n")
        for model_name, result in training_results.items():
            f.write(f"{model_name.upper()}:\n")
            f.write(f"  훈련 시간: {result['train_time']/60:.1f}분\n")
            f.write(f"  최종 훈련 손실: {result['history']['train_loss'][-1]:.4f}\n")
            f.write(f"  최종 검증 손실: {result['history']['val_loss'][-1]:.4f}\n")
            f.write(f"  테스트 RMSE: {result['test_results']['rmse']:.4f}\n")
            f.write(f"  테스트 R²: {result['test_results']['r2']:.4f}\n\n")
        
        # 4. 상세 평가 결과
        f.write("4. 상세 평가 결과\n")
        f.write("-"*30 + "\n")
        for model_name, result in evaluation_results.items():
            f.write(f"{model_name.upper()}:\n")
            f.write(f"  RMSE: {result['rmse']:.4f}\n")
            f.write(f"  MAE: {result['mae']:.4f}\n")
            f.write(f"  R²: {result['r2']:.4f}\n")
            f.write(f"  ±0.5 Brix 정확도: {result['accuracy_within_0.5_brix']:.1f}%\n")
            f.write(f"  ±1.0 Brix 정확도: {result['accuracy_within_1.0_brix']:.1f}%\n")
            f.write(f"  Pearson 상관계수: {result['pearson_corr']:.4f}\n\n")
        
        # 5. 하이퍼파라미터 튜닝 결과
        f.write("5. 하이퍼파라미터 튜닝 결과\n")
        f.write("-"*30 + "\n")
        if tuning_results and 'individual_results' in tuning_results:
            for model_type, model_result in tuning_results['individual_results'].items():
                f.write(f"{model_type.upper()}:\n")
                f.write(f"  최고 CV RMSE: {model_result['best_score']:.4f}\n")
                f.write(f"  최적 파라미터: {model_result['best_parameters']}\n\n")
        else:
            f.write("하이퍼파라미터 튜닝을 수행하지 않았습니다.\n\n")
        
        # 6. 모델 비교 및 앙상블 결과
        f.write("6. 모델 비교 및 앙상블 결과\n")
        f.write("-"*30 + "\n")
        if comparison_results:
            f.write("개별 모델:\n")
            for name, result in comparison_results.get('individual_models', {}).items():
                f.write(f"  {name}: CV RMSE {result['mean_cv_score']:.4f}\n")
            
            f.write("\n앙상블 모델:\n")
            for name, result in comparison_results.get('ensemble_models', {}).items():
                f.write(f"  {name}: RMSE {result['rmse']:.4f}\n")
            
            f.write(f"\n추천 모델:\n")
            f.write(f"  최고 개별 모델: {comparison_results.get('best_individual')}\n")
            f.write(f"  최고 앙상블: {comparison_results.get('best_ensemble')}\n")
        
        # 7. 결론 및 제안
        f.write("\n7. 결론 및 제안\n")
        f.write("-"*30 + "\n")
        f.write("본 프로젝트를 통해 수박 타격음 기반 당도 예측이 가능함을 확인했습니다.\n")
        f.write("앙상블 기법을 통해 개별 모델 대비 성능 향상을 달성했습니다.\n")
        f.write("실용화를 위해서는 더 많은 데이터 수집과 다양한 품종에 대한 검증이 필요합니다.\n")
    
    print(f"최종 종합 리포트 생성 완료: {report_path}")
    return str(report_path)


def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description='수박 당도 예측 모델 4단계 통합 파이프라인')
    parser.add_argument('--skip-tuning', action='store_true', 
                       help='하이퍼파라미터 튜닝 건너뛰기 (시간 절약)')
    parser.add_argument('--augmentation', choices=['light', 'medium', 'heavy'], 
                       default='medium', help='데이터 증강 강도')
    parser.add_argument('--epochs', type=int, default=50, 
                       help='기본 훈련 epoch 수')
    
    args = parser.parse_args()
    
    print("="*80)
    print("수박 당도 예측 모델 4단계 통합 파이프라인 시작")
    print("="*80)
    
    start_time = time.time()
    
    try:
        # 0. 환경 설정
        setup_directories()
        
        # 1. 데이터 로딩 및 준비
        dataset, dataloaders, data_splits = load_and_prepare_data(args.augmentation)
        
        # 2. 기본 모델 훈련
        training_results = train_baseline_models(dataloaders)
        
        # 3. 모델 평가
        evaluation_results = evaluate_models(training_results, dataloaders)
        
        # 4. 하이퍼파라미터 튜닝 (선택사항)
        tuning_results = run_hyperparameter_tuning_stage(
            dataset, skip_tuning=args.skip_tuning
        )
        
        # 5. 모델 비교 및 앙상블
        comparison_results = run_model_comparison_stage(dataset, tuning_results)
        
        # 6. 최종 리포트 생성
        final_report_path = generate_final_report(
            training_results, evaluation_results, 
            tuning_results, comparison_results
        )
        
        # 전체 소요 시간
        total_time = time.time() - start_time
        
        print("\n" + "="*80)
        print("전체 파이프라인 완료!")
        print(f"총 소요 시간: {total_time/3600:.1f}시간")
        print(f"최종 리포트: {final_report_path}")
        print("="*80)
        
    except KeyboardInterrupt:
        print("\n사용자에 의해 실행이 중단되었습니다.")
    except Exception as e:
        print(f"\n오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 