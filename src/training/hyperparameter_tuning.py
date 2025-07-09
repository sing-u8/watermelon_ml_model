"""
수박 당도 예측 모델을 위한 하이퍼파라미터 튜닝 시스템

이 모듈은 다음 기능을 제공합니다:
1. Grid Search 및 Random Search 구현
2. 베이지안 최적화 (Optuna 기반)
3. 교차 검증을 통한 하이퍼파라미터 평가
4. 튜닝 결과 시각화 및 분석
5. 최적 하이퍼파라미터 자동 선택

Author: AI Assistant
Date: 2024
"""

import os
import json
import pickle
import random
from typing import Dict, List, Tuple, Optional, Any, Callable
from pathlib import Path
import numpy as np
import pandas as pd
from itertools import product
import matplotlib.pyplot as plt
import seaborn as sns
from ..utils.font_utils import setup_korean_font

# 한글 폰트 설정
setup_korean_font()
from sklearn.model_selection import StratifiedKFold

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from .trainer import Trainer, create_trainer
from ..evaluation.metrics import ModelEvaluator, RegressionMetrics
from ..data.dataset import WatermelonDataset, create_dataloaders


class HyperparameterSpace:
    """하이퍼파라미터 공간 정의 클래스"""
    
    @staticmethod
    def get_vgg16_space() -> Dict[str, Any]:
        """VGG-16 모델용 하이퍼파라미터 공간"""
        return {
            'learning_rate': [0.0001, 0.0005, 0.001, 0.005, 0.01],
            'batch_size': [8, 16, 32],
            'optimizer': ['adam', 'adamw', 'sgd'],
            'weight_decay': [1e-5, 1e-4, 1e-3],
            'dropout': [0.3, 0.5, 0.7],
            'scheduler': ['step', 'plateau', 'cosine', None],
            'loss_function': ['mse', 'mae', 'huber'],
            'freeze_features': [True, False],
            'use_batch_norm': [True, False]
        }
    
    @staticmethod
    def get_custom_cnn_space() -> Dict[str, Any]:
        """커스텀 CNN 모델용 하이퍼파라미터 공간"""
        return {
            'learning_rate': [0.0001, 0.0005, 0.001, 0.005],
            'batch_size': [8, 16, 32],
            'optimizer': ['adam', 'adamw'],
            'weight_decay': [1e-5, 1e-4, 1e-3],
            'dropout': [0.2, 0.3, 0.5],
            'scheduler': ['step', 'plateau', 'cosine'],
            'loss_function': ['mse', 'mae', 'huber'],
            'use_residual': [True, False]
        }
    
    @staticmethod
    def get_training_space() -> Dict[str, Any]:
        """훈련 설정용 하이퍼파라미터 공간"""
        return {
            'epochs': [30, 50, 80, 100],
            'early_stopping_patience': [10, 15, 20],
            'step_size': [10, 20, 30],  # StepLR용
            'gamma': [0.1, 0.5, 0.8],   # StepLR용
            'patience': [5, 10, 15],    # ReduceLROnPlateau용
            'factor': [0.1, 0.5, 0.8]   # ReduceLROnPlateau용
        }


class GridSearchTuner:
    """Grid Search 기반 하이퍼파라미터 튜닝"""
    
    def __init__(self,
                 model_type: str,
                 dataset: WatermelonDataset,
                 cv_folds: int = 3,
                 random_state: int = 42,
                 save_dir: str = "hyperparameter_results"):
        """
        Args:
            model_type: 모델 타입 ('vgg16' 또는 'custom_cnn')
            dataset: 데이터셋
            cv_folds: 교차 검증 폴드 수
            random_state: 랜덤 시드
            save_dir: 결과 저장 디렉토리
        """
        self.model_type = model_type
        self.dataset = dataset
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # 하이퍼파라미터 공간 설정
        if model_type == 'vgg16':
            self.param_space = HyperparameterSpace.get_vgg16_space()
        else:
            self.param_space = HyperparameterSpace.get_custom_cnn_space()
            
        self.training_space = HyperparameterSpace.get_training_space()
        
        # 결과 저장
        self.results = []
        
        # 교차 검증 설정
        self._setup_cross_validation()
        
    def _setup_cross_validation(self):
        """교차 검증 설정"""
        # Brix 값을 기반으로 계층화된 폴드 생성
        brix_values = [info['brix'] for info in self.dataset.sample_info]
        
        # Brix 값을 구간으로 나누어 계층화
        brix_bins = np.digitize(brix_values, [9.0, 10.5, 12.0])
        
        self.cv_splitter = StratifiedKFold(
            n_splits=self.cv_folds,
            shuffle=True,
            random_state=self.random_state
        )
        
        self.cv_folds_data = list(self.cv_splitter.split(
            range(len(self.dataset)), brix_bins
        ))
        
        print(f"교차 검증 설정 완료: {self.cv_folds} 폴드")
        
    def _generate_param_combinations(self, max_combinations: Optional[int] = None) -> List[Dict]:
        """파라미터 조합 생성"""
        # 모델 파라미터 조합
        model_param_names = list(self.param_space.keys())
        model_param_values = [self.param_space[name] for name in model_param_names]
        model_combinations = list(product(*model_param_values))
        
        # 훈련 파라미터 조합 (더 제한적으로)
        training_combinations = [
            {'epochs': 50, 'early_stopping_patience': 15, 'step_size': 20, 'gamma': 0.5, 'patience': 10, 'factor': 0.5},
            {'epochs': 80, 'early_stopping_patience': 20, 'step_size': 30, 'gamma': 0.1, 'patience': 15, 'factor': 0.1}
        ]
        
        # 모든 조합 생성
        all_combinations = []
        for model_combo in model_combinations:
            model_params = dict(zip(model_param_names, model_combo))
            for training_params in training_combinations:
                combined_params = {**model_params, **training_params}
                all_combinations.append(combined_params)
        
        # 조합 수 제한
        if max_combinations and len(all_combinations) > max_combinations:
            random.seed(self.random_state)
            all_combinations = random.sample(all_combinations, max_combinations)
            
        print(f"총 {len(all_combinations)}개 하이퍼파라미터 조합 생성")
        return all_combinations
    
    def _evaluate_params(self, params: Dict[str, Any]) -> Dict[str, float]:
        """하나의 파라미터 조합 평가"""
        cv_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(self.cv_folds_data):
            try:
                # 데이터 로더 생성
                train_subset = Subset(self.dataset, train_idx)
                val_subset = Subset(self.dataset, val_idx)
                
                train_loader = DataLoader(
                    train_subset, 
                    batch_size=params['batch_size'],
                    shuffle=True,
                    num_workers=2,
                    pin_memory=True
                )
                
                val_loader = DataLoader(
                    val_subset,
                    batch_size=params['batch_size'],
                    shuffle=False,
                    num_workers=2,
                    pin_memory=True
                )
                
                # 스케줄러 파라미터 설정
                scheduler_params = {}
                if params['scheduler'] == 'step':
                    scheduler_params = {
                        'step_size': params['step_size'],
                        'gamma': params['gamma']
                    }
                elif params['scheduler'] == 'plateau':
                    scheduler_params = {
                        'patience': params['patience'],
                        'factor': params['factor']
                    }
                
                # Trainer 생성
                trainer = create_trainer(
                    model_type=self.model_type,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    lr=params['learning_rate'],
                    optimizer=params['optimizer'],
                    weight_decay=params['weight_decay'],
                    dropout=params['dropout'],
                    scheduler=params['scheduler'],
                    loss_function=params['loss_function'],
                    checkpoint_dir=self.save_dir / f"fold_{fold}",
                    verbose=False,
                    **{k: v for k, v in params.items() 
                       if k in ['freeze_features', 'use_batch_norm', 'use_residual']},
                    **scheduler_params
                )
                
                # 훈련
                history = trainer.fit(
                    epochs=params['epochs'],
                    early_stopping_patience=params['early_stopping_patience'],
                    save_best_model=False  # 속도를 위해 저장 안함
                )
                
                # 평가
                val_results = trainer.evaluate(return_predictions=False)
                cv_scores.append(val_results['rmse'])
                
            except Exception as e:
                print(f"Fold {fold} 평가 중 오류: {e}")
                cv_scores.append(float('inf'))  # 실패한 경우 매우 큰 값
        
        # 교차 검증 결과 계산
        cv_scores = np.array(cv_scores)
        valid_scores = cv_scores[cv_scores != float('inf')]
        
        if len(valid_scores) == 0:
            return {
                'mean_cv_score': float('inf'),
                'std_cv_score': float('inf'),
                'valid_folds': 0
            }
        
        return {
            'mean_cv_score': np.mean(valid_scores),
            'std_cv_score': np.std(valid_scores),
            'valid_folds': len(valid_scores)
        }
    
    def run(self, max_combinations: int = 50) -> Dict[str, Any]:
        """Grid Search 실행"""
        print(f"\n{'='*60}")
        print(f"Grid Search 하이퍼파라미터 튜닝 시작")
        print(f"모델: {self.model_type}, 최대 조합: {max_combinations}")
        print(f"{'='*60}")
        
        # 파라미터 조합 생성
        param_combinations = self._generate_param_combinations(max_combinations)
        
        # 각 조합 평가
        best_score = float('inf')
        best_params = None
        
        for i, params in enumerate(param_combinations):
            print(f"\n진행률: {i+1}/{len(param_combinations)}")
            print(f"현재 파라미터: {params}")
            
            # 평가 수행
            scores = self._evaluate_params(params)
            
            # 결과 저장
            result = {
                'combination_id': i,
                'parameters': params,
                **scores
            }
            self.results.append(result)
            
            # 최고 성능 업데이트
            if scores['mean_cv_score'] < best_score:
                best_score = scores['mean_cv_score']
                best_params = params.copy()
                
            print(f"평균 CV RMSE: {scores['mean_cv_score']:.4f} ± {scores['std_cv_score']:.4f}")
            print(f"현재 최고 성능: {best_score:.4f}")
        
        # 결과 정리
        final_results = {
            'best_score': best_score,
            'best_parameters': best_params,
            'all_results': self.results,
            'model_type': self.model_type,
            'cv_folds': self.cv_folds
        }
        
        # 결과 저장
        self._save_results(final_results)
        
        # 결과 시각화
        self._visualize_results()
        
        print(f"\n{'='*60}")
        print(f"Grid Search 완료!")
        print(f"최고 성능: RMSE {best_score:.4f}")
        print(f"최적 파라미터: {best_params}")
        print(f"{'='*60}")
        
        return final_results
    
    def _save_results(self, results: Dict[str, Any]):
        """결과 저장"""
        # JSON 저장
        json_path = self.save_dir / f"grid_search_{self.model_type}_results.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            # NumPy 배열을 리스트로 변환하여 JSON 직렬화 가능하게 만듦
            json_results = self._convert_for_json(results)
            json.dump(json_results, f, indent=2, ensure_ascii=False)
        
        # Pickle 저장 (원본 객체 보존)
        pickle_path = self.save_dir / f"grid_search_{self.model_type}_results.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump(results, f)
            
        print(f"결과 저장: {json_path}, {pickle_path}")
    
    def _convert_for_json(self, obj):
        """JSON 직렬화를 위한 객체 변환"""
        if isinstance(obj, dict):
            return {k: self._convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_for_json(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif obj is None or isinstance(obj, (str, int, float, bool)):
            return obj
        else:
            return str(obj)  # 기타 객체는 문자열로 변환
    
    def _visualize_results(self):
        """결과 시각화"""
        if not self.results:
            return
            
        # 결과를 DataFrame으로 변환
        df_results = []
        for result in self.results:
            row = result['parameters'].copy()
            row['mean_cv_score'] = result['mean_cv_score']
            row['std_cv_score'] = result['std_cv_score']
            df_results.append(row)
        
        df = pd.DataFrame(df_results)
        
        # 유한한 값만 필터링
        df = df[df['mean_cv_score'] != float('inf')]
        
        if len(df) == 0:
            print("시각화할 유효한 결과가 없습니다.")
            return
        
        # 상위 10개 결과 시각화
        top_10 = df.nsmallest(10, 'mean_cv_score')
        
        # 1. 상위 성능 파라미터 히트맵
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 학습률 vs 배치 크기
        pivot1 = df.pivot_table(
            values='mean_cv_score',
            index='learning_rate',
            columns='batch_size',
            aggfunc='mean'
        )
        sns.heatmap(pivot1, annot=True, fmt='.3f', ax=axes[0, 0], cmap='viridis_r')
        axes[0, 0].set_title('학습률 vs 배치 크기')
        
        # 옵티마이저 vs 드롭아웃
        pivot2 = df.pivot_table(
            values='mean_cv_score',
            index='optimizer',
            columns='dropout',
            aggfunc='mean'
        )
        sns.heatmap(pivot2, annot=True, fmt='.3f', ax=axes[0, 1], cmap='viridis_r')
        axes[0, 1].set_title('옵티마이저 vs 드롭아웃')
        
        # 상위 10개 성능 막대 그래프
        axes[1, 0].barh(range(len(top_10)), top_10['mean_cv_score'])
        axes[1, 0].set_yticks(range(len(top_10)))
        axes[1, 0].set_yticklabels([f"#{i+1}" for i in range(len(top_10))])
        axes[1, 0].set_xlabel('Mean CV RMSE')
        axes[1, 0].set_title('상위 10개 조합 성능')
        
        # 파라미터별 성능 분포
        numeric_params = ['learning_rate', 'weight_decay', 'dropout']
        for param in numeric_params:
            if param in df.columns:
                correlation = df['mean_cv_score'].corr(df[param])
                axes[1, 1].scatter(df[param], df['mean_cv_score'], alpha=0.6, label=f'{param} (r={correlation:.3f})')
        axes[1, 1].set_xlabel('Parameter Value')
        axes[1, 1].set_ylabel('Mean CV RMSE')
        axes[1, 1].set_title('파라미터 값 vs 성능')
        axes[1, 1].legend()
        axes[1, 1].set_xscale('log')
        
        plt.tight_layout()
        save_path = self.save_dir / f"grid_search_{self.model_type}_visualization.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"시각화 저장: {save_path}")


class RandomSearchTuner(GridSearchTuner):
    """Random Search 기반 하이퍼파라미터 튜닝"""
    
    def _generate_param_combinations(self, max_combinations: int = 100) -> List[Dict]:
        """랜덤 파라미터 조합 생성"""
        random.seed(self.random_state)
        combinations = []
        
        for _ in range(max_combinations):
            params = {}
            
            # 모델 파라미터 랜덤 선택
            for param_name, param_values in self.param_space.items():
                params[param_name] = random.choice(param_values)
            
            # 훈련 파라미터 랜덤 선택
            params['epochs'] = random.choice([30, 50, 80])
            params['early_stopping_patience'] = random.choice([10, 15, 20])
            params['step_size'] = random.choice([10, 20, 30])
            params['gamma'] = random.choice([0.1, 0.5, 0.8])
            params['patience'] = random.choice([5, 10, 15])
            params['factor'] = random.choice([0.1, 0.5, 0.8])
            
            combinations.append(params)
        
        print(f"총 {len(combinations)}개 랜덤 조합 생성")
        return combinations


def run_hyperparameter_tuning(dataset: WatermelonDataset,
                             model_types: List[str] = ['vgg16', 'custom_cnn'],
                             search_type: str = 'random',
                             max_combinations: int = 50,
                             cv_folds: int = 3,
                             save_dir: str = "hyperparameter_results") -> Dict[str, Any]:
    """
    전체 하이퍼파라미터 튜닝 파이프라인 실행
    
    Args:
        dataset: 데이터셋
        model_types: 튜닝할 모델 타입들
        search_type: 'grid' 또는 'random'
        max_combinations: 최대 조합 수
        cv_folds: 교차 검증 폴드 수
        save_dir: 결과 저장 디렉토리
        
    Returns:
        Dict: 모든 모델의 튜닝 결과
    """
    results = {}
    
    for model_type in model_types:
        print(f"\n{'='*80}")
        print(f"{model_type.upper()} 모델 하이퍼파라미터 튜닝 시작")
        print(f"{'='*80}")
        
        # 튜너 생성
        if search_type == 'grid':
            tuner = GridSearchTuner(
                model_type=model_type,
                dataset=dataset,
                cv_folds=cv_folds,
                save_dir=save_dir
            )
        else:
            tuner = RandomSearchTuner(
                model_type=model_type,
                dataset=dataset,
                cv_folds=cv_folds,
                save_dir=save_dir
            )
        
        # 튜닝 실행
        model_results = tuner.run(max_combinations=max_combinations)
        results[model_type] = model_results
    
    # 모델 간 비교 결과 생성
    comparison_results = compare_tuning_results(results, save_dir)
    
    return {
        'individual_results': results,
        'comparison': comparison_results
    }


def compare_tuning_results(results: Dict[str, Any], save_dir: str):
    """모델별 튜닝 결과 비교"""
    save_dir = Path(save_dir)
    
    comparison_data = []
    for model_type, model_results in results.items():
        comparison_data.append({
            'model': model_type,
            'best_rmse': model_results['best_score'],
            'best_params': model_results['best_parameters']
        })
    
    # 비교 시각화
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = [data['model'] for data in comparison_data]
    scores = [data['best_rmse'] for data in comparison_data]
    
    bars = ax.bar(models, scores, alpha=0.8, color=['skyblue', 'lightcoral'])
    ax.set_ylabel('Best CV RMSE')
    ax.set_title('모델별 최고 성능 비교')
    
    # 막대 위에 값 표시
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{score:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    save_path = save_dir / "model_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    # 비교 리포트 저장
    report_path = save_dir / "tuning_comparison_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("하이퍼파라미터 튜닝 결과 비교\n")
        f.write("="*50 + "\n\n")
        
        for data in sorted(comparison_data, key=lambda x: x['best_rmse']):
            f.write(f"모델: {data['model']}\n")
            f.write(f"최고 성능: RMSE {data['best_rmse']:.4f}\n")
            f.write(f"최적 파라미터:\n")
            for param, value in data['best_params'].items():
                f.write(f"  - {param}: {value}\n")
            f.write("\n")
    
    print(f"모델 비교 결과 저장: {save_path}, {report_path}")
    return comparison_data


if __name__ == "__main__":
    # 테스트 실행 예제
    print("하이퍼파라미터 튜닝 모듈이 성공적으로 로딩되었습니다!")
    print("사용 예제:")
    print("results = run_hyperparameter_tuning(dataset, ['vgg16', 'custom_cnn'])")
    print("이 프로세스는 시간이 오래 걸릴 수 있습니다.") 