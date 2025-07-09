"""
수박 당도 예측 모델 비교 및 앙상블 시스템

이 모듈은 다음 기능을 제공합니다:
1. VGG-16 vs 커스텀 CNN 성능 비교
2. 앙상블 기법 (평균, 가중 평균, 스태킹)
3. 교차 검증을 통한 모델 성능 분석
4. 최종 모델 선택 및 성능 리포트

Author: AI Assistant
Date: 2024
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from ..utils.font_utils import setup_korean_font

# 한글 폰트 설정
setup_korean_font()
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from ..training.trainer import Trainer, create_trainer
from .metrics import ModelEvaluator, VisualizationTools
from ..data.dataset import WatermelonDataset


class ModelComparator:
    """모델 성능 비교 클래스"""
    
    def __init__(self, dataset: WatermelonDataset, cv_folds: int = 5):
        self.dataset = dataset
        self.cv_folds = cv_folds
        self.results = {}
        
        # 교차 검증 설정
        brix_values = [info['brix'] for info in dataset.sample_info]
        brix_bins = np.digitize(brix_values, [9.0, 10.5, 12.0])
        
        self.cv_splitter = StratifiedKFold(
            n_splits=cv_folds, shuffle=True, random_state=42
        )
        self.cv_folds_data = list(self.cv_splitter.split(
            range(len(dataset)), brix_bins
        ))
    
    def compare_models(self, 
                      vgg16_params: Dict = None, 
                      custom_cnn_params: Dict = None) -> Dict[str, Any]:
        """VGG-16과 커스텀 CNN 성능 비교"""
        
        # 기본 파라미터 설정
        if vgg16_params is None:
            vgg16_params = {
                'learning_rate': 0.001, 'batch_size': 16, 'optimizer': 'adam',
                'epochs': 50, 'dropout': 0.5, 'freeze_features': False
            }
        
        if custom_cnn_params is None:
            custom_cnn_params = {
                'learning_rate': 0.001, 'batch_size': 16, 'optimizer': 'adam',
                'epochs': 50, 'dropout': 0.3, 'use_residual': True
            }
        
        models_to_compare = {
            'vgg16': vgg16_params,
            'custom_cnn': custom_cnn_params
        }
        
        print("\n모델 성능 비교 시작...")
        
        for model_name, params in models_to_compare.items():
            print(f"\n{model_name.upper()} 모델 평가 중...")
            
            cv_scores = []
            fold_predictions = []
            fold_targets = []
            
            for fold, (train_idx, val_idx) in enumerate(self.cv_folds_data):
                print(f"  Fold {fold+1}/{self.cv_folds}")
                
                # 데이터 로더 생성
                train_subset = Subset(self.dataset, train_idx)
                val_subset = Subset(self.dataset, val_idx)
                
                train_loader = DataLoader(train_subset, batch_size=params['batch_size'], shuffle=True)
                val_loader = DataLoader(val_subset, batch_size=params['batch_size'], shuffle=False)
                
                # 모델 훈련
                trainer = create_trainer(
                    model_type=model_name,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    verbose=False,
                    **params
                )
                
                trainer.fit(epochs=params['epochs'], save_best_model=False)
                
                # 평가
                evaluator = ModelEvaluator(trainer.model)
                y_pred, y_true = evaluator.predict(val_loader)
                
                # 성능 계산
                rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
                cv_scores.append(rmse)
                fold_predictions.extend(y_pred)
                fold_targets.extend(y_true)
            
            # 결과 저장
            self.results[model_name] = {
                'cv_scores': cv_scores,
                'mean_cv_score': np.mean(cv_scores),
                'std_cv_score': np.std(cv_scores),
                'predictions': np.array(fold_predictions),
                'targets': np.array(fold_targets),
                'parameters': params
            }
            
            print(f"  평균 CV RMSE: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
        
        return self.results
    
    def visualize_comparison(self, save_dir: str = "model_comparison_results"):
        """모델 비교 결과 시각화"""
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. CV 점수 비교
        model_names = list(self.results.keys())
        cv_scores = [self.results[name]['cv_scores'] for name in model_names]
        
        axes[0, 0].boxplot(cv_scores, labels=model_names)
        axes[0, 0].set_ylabel('CV RMSE')
        axes[0, 0].set_title('교차 검증 점수 분포')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 평균 성능 막대 그래프
        mean_scores = [self.results[name]['mean_cv_score'] for name in model_names]
        std_scores = [self.results[name]['std_cv_score'] for name in model_names]
        
        bars = axes[0, 1].bar(model_names, mean_scores, yerr=std_scores, alpha=0.8, capsize=5)
        axes[0, 1].set_ylabel('Mean CV RMSE')
        axes[0, 1].set_title('평균 성능 비교')
        
        # 막대 위에 값 표시
        for bar, score in zip(bars, mean_scores):
            height = bar.get_height()
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                           f'{score:.4f}', ha='center', va='bottom')
        
        # 3. 예측 vs 실제 비교
        colors = ['blue', 'red']
        for i, model_name in enumerate(model_names):
            result = self.results[model_name]
            axes[1, 0].scatter(result['targets'], result['predictions'], 
                             alpha=0.6, color=colors[i], label=model_name, s=30)
        
        min_val = min([np.min(self.results[name]['targets']) for name in model_names])
        max_val = max([np.max(self.results[name]['targets']) for name in model_names])
        axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8)
        axes[1, 0].set_xlabel('실제 Brix 값')
        axes[1, 0].set_ylabel('예측 Brix 값')
        axes[1, 0].set_title('예측 vs 실제값 비교')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 오차 분포 비교
        for i, model_name in enumerate(model_names):
            result = self.results[model_name]
            errors = result['predictions'] - result['targets']
            axes[1, 1].hist(errors, alpha=0.6, label=model_name, bins=20, color=colors[i])
        
        axes[1, 1].axvline(0, color='black', linestyle='--', alpha=0.8)
        axes[1, 1].set_xlabel('예측 오차')
        axes[1, 1].set_ylabel('빈도')
        axes[1, 1].set_title('오차 분포 비교')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = save_dir / "model_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"비교 결과 저장: {save_path}")


class EnsembleModel:
    """앙상블 모델 클래스"""
    
    def __init__(self, models: List[nn.Module], ensemble_type: str = 'average'):
        """
        Args:
            models: 앙상블할 모델들
            ensemble_type: 'average', 'weighted', 'stacking'
        """
        self.models = models
        self.ensemble_type = ensemble_type
        self.weights = None
        self.meta_model = None
        
        for model in self.models:
            model.eval()
    
    def fit_ensemble(self, train_loader: DataLoader, val_loader: DataLoader):
        """앙상블 가중치 또는 메타 모델 학습"""
        
        if self.ensemble_type == 'weighted':
            self._fit_weighted_ensemble(val_loader)
        elif self.ensemble_type == 'stacking':
            self._fit_stacking_ensemble(train_loader, val_loader)
    
    def _fit_weighted_ensemble(self, val_loader: DataLoader):
        """가중 평균 앙상블 가중치 학습"""
        # 각 모델의 검증 성능에 기반한 가중치 계산
        model_scores = []
        
        for model in self.models:
            evaluator = ModelEvaluator(model)
            y_pred, y_true = evaluator.predict(val_loader)
            rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
            model_scores.append(1.0 / rmse)  # 성능이 좋을수록 높은 가중치
        
        # 가중치 정규화
        total_score = sum(model_scores)
        self.weights = [score / total_score for score in model_scores]
        
        print(f"가중 앙상블 가중치: {self.weights}")
    
    def _fit_stacking_ensemble(self, train_loader: DataLoader, val_loader: DataLoader):
        """스태킹 앙상블 메타 모델 학습"""
        device = next(self.models[0].parameters()).device
        
        # 훈련 데이터에서 기본 모델들의 예측 수집
        train_predictions = []
        train_targets = []
        
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            batch_preds = []
            
            with torch.no_grad():
                for model in self.models:
                    outputs = model(inputs)
                    batch_preds.append(outputs.squeeze().cpu().numpy())
            
            train_predictions.append(np.column_stack(batch_preds))
            train_targets.extend(targets.numpy())
        
        X_train = np.vstack(train_predictions)
        y_train = np.array(train_targets)
        
        # 메타 모델 학습 (선형 회귀)
        self.meta_model = LinearRegression()
        self.meta_model.fit(X_train, y_train)
        
        print("스태킹 앙상블 메타 모델 학습 완료")
    
    def predict(self, data_loader: DataLoader) -> np.ndarray:
        """앙상블 예측"""
        device = next(self.models[0].parameters()).device
        all_predictions = []
        
        with torch.no_grad():
            for inputs, _ in data_loader:
                inputs = inputs.to(device)
                
                # 각 모델의 예측 수집
                model_preds = []
                for model in self.models:
                    outputs = model(inputs)
                    model_preds.append(outputs.squeeze().cpu().numpy())
                
                # 앙상블 방법에 따른 최종 예측
                if self.ensemble_type == 'average':
                    batch_pred = np.mean(model_preds, axis=0)
                elif self.ensemble_type == 'weighted':
                    batch_pred = np.average(model_preds, axis=0, weights=self.weights)
                elif self.ensemble_type == 'stacking':
                    batch_features = np.column_stack(model_preds)
                    batch_pred = self.meta_model.predict(batch_features)
                
                all_predictions.extend(batch_pred)
        
        return np.array(all_predictions)


def run_comprehensive_comparison(dataset: WatermelonDataset,
                               save_dir: str = "final_results") -> Dict[str, Any]:
    """종합적인 모델 비교 및 앙상블 평가"""
    
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    print("="*60)
    print("종합 모델 비교 및 앙상블 평가 시작")
    print("="*60)
    
    # 1. 개별 모델 성능 비교
    print("\n1. 개별 모델 성능 비교")
    comparator = ModelComparator(dataset)
    
    # 최적 파라미터로 비교 (실제로는 하이퍼파라미터 튜닝 결과 사용)
    vgg16_params = {
        'learning_rate': 0.0005, 'batch_size': 16, 'optimizer': 'adam',
        'epochs': 50, 'dropout': 0.5, 'freeze_features': False
    }
    
    custom_cnn_params = {
        'learning_rate': 0.001, 'batch_size': 16, 'optimizer': 'adam',
        'epochs': 50, 'dropout': 0.3, 'use_residual': True
    }
    
    comparison_results = comparator.compare_models(vgg16_params, custom_cnn_params)
    comparator.visualize_comparison(save_dir)
    
    # 2. 앙상블 모델 평가
    print("\n2. 앙상블 모델 평가")
    
    # 최고 성능 모델들로 앙상블 구성
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 모델 재훈련 (전체 데이터 사용)
    from torch.utils.data import DataLoader
    full_loader = DataLoader(dataset, batch_size=16, shuffle=False)
    
    # 훈련/검증 분할
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    # VGG16 모델 훈련
    vgg16_trainer = create_trainer('vgg16', train_loader, val_loader, **vgg16_params, verbose=False)
    vgg16_trainer.fit(epochs=50, save_best_model=False)
    
    # 커스텀 CNN 모델 훈련
    custom_trainer = create_trainer('custom_cnn', train_loader, val_loader, **custom_cnn_params, verbose=False)
    custom_trainer.fit(epochs=50, save_best_model=False)
    
    models = [vgg16_trainer.model, custom_trainer.model]
    
    # 다양한 앙상블 방법 평가
    ensemble_results = {}
    ensemble_types = ['average', 'weighted', 'stacking']
    
    for ensemble_type in ensemble_types:
        print(f"  {ensemble_type} 앙상블 평가 중...")
        
        ensemble = EnsembleModel(models, ensemble_type)
        if ensemble_type in ['weighted', 'stacking']:
            ensemble.fit_ensemble(train_loader, val_loader)
        
        # 검증 데이터에서 예측
        ensemble_pred = ensemble.predict(val_loader)
        
        # 실제값 수집
        actual_values = []
        for _, targets in val_loader:
            actual_values.extend(targets.numpy())
        actual_values = np.array(actual_values)
        
        # 성능 계산
        rmse = np.sqrt(np.mean((ensemble_pred - actual_values) ** 2))
        mae = np.mean(np.abs(ensemble_pred - actual_values))
        
        ensemble_results[ensemble_type] = {
            'rmse': rmse,
            'mae': mae,
            'predictions': ensemble_pred,
            'targets': actual_values
        }
        
        print(f"    RMSE: {rmse:.4f}, MAE: {mae:.4f}")
    
    # 3. 최종 결과 정리 및 시각화
    print("\n3. 최종 결과 정리")
    
    final_results = {
        'individual_models': comparison_results,
        'ensemble_models': ensemble_results,
        'best_individual': min(comparison_results.keys(), 
                              key=lambda x: comparison_results[x]['mean_cv_score']),
        'best_ensemble': min(ensemble_results.keys(), 
                            key=lambda x: ensemble_results[x]['rmse'])
    }
    
    # 최종 비교 시각화
    _plot_final_comparison(final_results, save_dir)
    
    # 결과 리포트 생성
    _generate_final_report(final_results, save_dir)
    
    return final_results


def _plot_final_comparison(results: Dict[str, Any], save_dir: Path):
    """최종 비교 결과 시각화"""
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 모든 모델 성능 비교
    all_models = {}
    
    # 개별 모델
    for name, result in results['individual_models'].items():
        all_models[name] = result['mean_cv_score']
    
    # 앙상블 모델
    for name, result in results['ensemble_models'].items():
        all_models[f'{name}_ensemble'] = result['rmse']
    
    # 성능 막대 그래프
    model_names = list(all_models.keys())
    scores = list(all_models.values())
    
    bars = axes[0].bar(model_names, scores, alpha=0.8)
    axes[0].set_ylabel('RMSE')
    axes[0].set_title('모든 모델 성능 비교')
    axes[0].tick_params(axis='x', rotation=45)
    
    # 최고 성능 강조
    best_idx = np.argmin(scores)
    bars[best_idx].set_color('red')
    
    # 상위 3개 모델의 예측 vs 실제 비교
    top_3_models = sorted(all_models.items(), key=lambda x: x[1])[:3]
    colors = ['blue', 'green', 'orange']
    
    for i, (model_name, _) in enumerate(top_3_models):
        if 'ensemble' in model_name:
            ensemble_type = model_name.replace('_ensemble', '')
            result = results['ensemble_models'][ensemble_type]
            y_pred, y_true = result['predictions'], result['targets']
        else:
            result = results['individual_models'][model_name]
            y_pred, y_true = result['predictions'], result['targets']
        
        axes[1].scatter(y_true, y_pred, alpha=0.6, color=colors[i], 
                       label=f'{model_name} (RMSE: {all_models[model_name]:.4f})', s=20)
    
    # 완벽한 예측선
    min_val = min([np.min(results['ensemble_models']['average']['targets'])])
    max_val = max([np.max(results['ensemble_models']['average']['targets'])])
    axes[1].plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8)
    
    axes[1].set_xlabel('실제 Brix 값')
    axes[1].set_ylabel('예측 Brix 값')
    axes[1].set_title('상위 3개 모델 예측 성능')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = save_dir / "final_model_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"최종 비교 결과 저장: {save_path}")


def _generate_final_report(results: Dict[str, Any], save_dir: Path):
    """최종 리포트 생성"""
    
    report_path = save_dir / "final_model_report.txt"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("수박 당도 예측 모델 최종 성능 리포트\n")
        f.write("="*50 + "\n\n")
        
        f.write("1. 개별 모델 성능\n")
        f.write("-"*30 + "\n")
        for name, result in results['individual_models'].items():
            f.write(f"{name.upper()}:\n")
            f.write(f"  평균 CV RMSE: {result['mean_cv_score']:.4f} ± {result['std_cv_score']:.4f}\n")
            f.write(f"  파라미터: {result['parameters']}\n\n")
        
        f.write("2. 앙상블 모델 성능\n")
        f.write("-"*30 + "\n")
        for name, result in results['ensemble_models'].items():
            f.write(f"{name.upper()} 앙상블:\n")
            f.write(f"  RMSE: {result['rmse']:.4f}\n")
            f.write(f"  MAE: {result['mae']:.4f}\n\n")
        
        f.write("3. 최종 추천\n")
        f.write("-"*30 + "\n")
        f.write(f"최고 개별 모델: {results['best_individual']}\n")
        f.write(f"최고 앙상블 모델: {results['best_ensemble']}\n")
        
        # 전체 최고 성능 모델 찾기
        all_scores = {}
        for name, result in results['individual_models'].items():
            all_scores[name] = result['mean_cv_score']
        for name, result in results['ensemble_models'].items():
            all_scores[f'{name}_ensemble'] = result['rmse']
        
        best_overall = min(all_scores.keys(), key=lambda x: all_scores[x])
        f.write(f"전체 최고 성능 모델: {best_overall} (RMSE: {all_scores[best_overall]:.4f})\n")
    
    print(f"최종 리포트 생성: {report_path}")


if __name__ == "__main__":
    print("모델 비교 및 앙상블 모듈이 성공적으로 로딩되었습니다!")
    print("사용 예제:")
    print("results = run_comprehensive_comparison(dataset)") 