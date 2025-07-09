"""
수박 당도 예측 모델을 위한 평가 메트릭 및 분석 도구

이 모듈은 다음 기능을 제공합니다:
1. 회귀 평가 메트릭 (RMSE, MAE, R², MAPE, 상관계수)
2. 당도 범위별 정확도 평가
3. 예측 결과 시각화 및 분석
4. 모델 성능 비교 도구
5. 상세한 오차 분석

Author: AI Assistant
Date: 2024
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from ..utils.font_utils import setup_korean_font

# 한글 폰트 설정
setup_korean_font()
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class RegressionMetrics:
    """회귀 모델을 위한 평가 메트릭 계산 클래스"""
    
    @staticmethod
    def compute_basic_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        기본 회귀 메트릭 계산
        
        Args:
            y_true: 실제 값
            y_pred: 예측 값
            
        Returns:
            Dict: 계산된 메트릭들
        """
        # 기본 메트릭
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # MAPE (Mean Absolute Percentage Error)
        # 0 값 처리를 위해 작은 값 추가
        epsilon = 1e-8
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100
        
        # 상관계수
        pearson_corr, pearson_p = stats.pearsonr(y_true, y_pred)
        spearman_corr, spearman_p = stats.spearmanr(y_true, y_pred)
        
        # 추가 메트릭
        # Mean Bias Error (MBE)
        mbe = np.mean(y_pred - y_true)
        
        # Normalized RMSE
        nrmse = rmse / (np.max(y_true) - np.min(y_true)) * 100
        
        # 결정계수의 조정된 버전 (샘플 수와 변수 수 고려)
        n = len(y_true)
        p = 1  # 회귀에서 독립변수는 1개 (예측값)
        adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1) if n > p + 1 else r2
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'adjusted_r2': adjusted_r2,
            'mape': mape,
            'mbe': mbe,
            'nrmse': nrmse,
            'pearson_corr': pearson_corr,
            'pearson_p_value': pearson_p,
            'spearman_corr': spearman_corr,
            'spearman_p_value': spearman_p
        }
    
    @staticmethod
    def compute_brix_accuracy(y_true: np.ndarray, 
                             y_pred: np.ndarray,
                             tolerances: List[float] = [0.5, 1.0, 1.5, 2.0]) -> Dict[str, float]:
        """
        당도 범위별 정확도 계산
        
        Args:
            y_true: 실제 Brix 값
            y_pred: 예측 Brix 값
            tolerances: 허용 오차 범위 리스트
            
        Returns:
            Dict: 각 허용 오차별 정확도
        """
        accuracies = {}
        
        for tolerance in tolerances:
            # 허용 오차 내에 있는 예측의 비율
            correct_predictions = np.abs(y_true - y_pred) <= tolerance
            accuracy = np.mean(correct_predictions) * 100
            accuracies[f'accuracy_within_{tolerance}_brix'] = accuracy
            
        return accuracies
    
    @staticmethod
    def compute_class_based_metrics(y_true: np.ndarray, 
                                   y_pred: np.ndarray,
                                   class_boundaries: List[float] = [9.0, 10.5, 12.0]) -> Dict[str, Any]:
        """
        당도 구간별 성능 분석
        
        Args:
            y_true: 실제 Brix 값
            y_pred: 예측 Brix 값
            class_boundaries: 구간 경계값들
            
        Returns:
            Dict: 구간별 메트릭
        """
        # 구간 생성 (Low, Medium, High)
        class_names = ['Low', 'Medium', 'High', 'Very_High']
        boundaries = [-np.inf] + class_boundaries + [np.inf]
        
        # 실제 값의 구간 분류
        true_classes = np.digitize(y_true, class_boundaries)
        pred_classes = np.digitize(y_pred, class_boundaries)
        
        class_metrics = {}
        
        for i, class_name in enumerate(class_names):
            # 해당 구간에 속하는 샘플들 찾기
            class_mask = (true_classes == i)
            
            if np.sum(class_mask) > 0:
                class_true = y_true[class_mask]
                class_pred = y_pred[class_mask]
                
                # 해당 구간의 메트릭 계산
                class_metrics[f'{class_name}_count'] = np.sum(class_mask)
                class_metrics[f'{class_name}_rmse'] = np.sqrt(mean_squared_error(class_true, class_pred))
                class_metrics[f'{class_name}_mae'] = mean_absolute_error(class_true, class_pred)
                
                if len(class_true) > 1:
                    class_metrics[f'{class_name}_r2'] = r2_score(class_true, class_pred)
                    corr, _ = stats.pearsonr(class_true, class_pred)
                    class_metrics[f'{class_name}_correlation'] = corr
                else:
                    class_metrics[f'{class_name}_r2'] = 0.0
                    class_metrics[f'{class_name}_correlation'] = 0.0
                    
                # 구간 내 평균 오차
                class_metrics[f'{class_name}_mean_error'] = np.mean(class_pred - class_true)
                class_metrics[f'{class_name}_std_error'] = np.std(class_pred - class_true)
        
        # 구간별 분류 정확도 (예측값이 올바른 구간에 속하는지)
        class_accuracy = np.mean(true_classes == pred_classes) * 100
        class_metrics['class_accuracy'] = class_accuracy
        
        return class_metrics


class ModelEvaluator:
    """모델 평가 및 분석을 위한 통합 클래스"""
    
    def __init__(self, model: nn.Module, device: torch.device = None):
        """
        Args:
            model: 평가할 모델
            device: 디바이스 (None이면 자동 선택)
        """
        self.model = model
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
    def predict(self, data_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """
        데이터 로더에서 예측 수행
        
        Args:
            data_loader: 평가할 데이터 로더
            
        Returns:
            Tuple: (예측값, 실제값)
        """
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(inputs)
                
                all_predictions.extend(outputs.squeeze().cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        return np.array(all_predictions), np.array(all_targets)
    
    def evaluate_comprehensive(self, 
                             data_loader: DataLoader,
                             dataset_name: str = "Test") -> Dict[str, Any]:
        """
        종합적인 모델 평가
        
        Args:
            data_loader: 평가할 데이터 로더
            dataset_name: 데이터셋 이름 (로그용)
            
        Returns:
            Dict: 모든 평가 결과
        """
        print(f"\n{'='*50}")
        print(f"{dataset_name} 데이터 종합 평가")
        print(f"{'='*50}")
        
        # 예측 수행
        y_pred, y_true = self.predict(data_loader)
        
        # 기본 메트릭 계산
        basic_metrics = RegressionMetrics.compute_basic_metrics(y_true, y_pred)
        
        # 당도 정확도 계산
        brix_accuracies = RegressionMetrics.compute_brix_accuracy(y_true, y_pred)
        
        # 구간별 메트릭 계산
        class_metrics = RegressionMetrics.compute_class_based_metrics(y_true, y_pred)
        
        # 결과 통합
        results = {
            'dataset_name': dataset_name,
            'sample_count': len(y_true),
            'predictions': y_pred,
            'targets': y_true,
            **basic_metrics,
            **brix_accuracies,
            **class_metrics
        }
        
        # 결과 출력
        self._print_evaluation_results(results)
        
        return results
    
    def _print_evaluation_results(self, results: Dict[str, Any]):
        """평가 결과 출력"""
        print(f"\n📊 기본 회귀 메트릭:")
        print(f"  • RMSE: {results['rmse']:.4f}")
        print(f"  • MAE: {results['mae']:.4f}")
        print(f"  • R²: {results['r2']:.4f}")
        print(f"  • 조정된 R²: {results['adjusted_r2']:.4f}")
        print(f"  • MAPE: {results['mape']:.2f}%")
        print(f"  • NRMSE: {results['nrmse']:.2f}%")
        print(f"  • 평균 편향 오차: {results['mbe']:.4f}")
        
        print(f"\n🎯 당도 범위별 정확도:")
        for key, value in results.items():
            if key.startswith('accuracy_within'):
                tolerance = key.split('_')[2]
                print(f"  • ±{tolerance} Brix 이내: {value:.1f}%")
        
        print(f"\n📈 상관관계 분석:")
        print(f"  • Pearson 상관계수: {results['pearson_corr']:.4f} (p={results['pearson_p_value']:.4f})")
        print(f"  • Spearman 상관계수: {results['spearman_corr']:.4f} (p={results['spearman_p_value']:.4f})")
        
        print(f"\n🏷️ 당도 구간별 성능:")
        class_names = ['Low', 'Medium', 'High', 'Very_High']
        for class_name in class_names:
            count_key = f'{class_name}_count'
            if count_key in results and results[count_key] > 0:
                rmse = results[f'{class_name}_rmse']
                mae = results[f'{class_name}_mae']
                r2 = results[f'{class_name}_r2']
                count = results[count_key]
                print(f"  • {class_name} 구간 (n={count}): RMSE={rmse:.3f}, MAE={mae:.3f}, R²={r2:.3f}")
        
        print(f"  • 구간 분류 정확도: {results['class_accuracy']:.1f}%")


class VisualizationTools:
    """평가 결과 시각화 도구"""
    
    @staticmethod
    def plot_prediction_vs_actual(y_true: np.ndarray, 
                                  y_pred: np.ndarray,
                                  title: str = "예측값 vs 실제값",
                                  save_path: Optional[str] = None,
                                  show_metrics: bool = True):
        """예측값과 실제값 비교 플롯"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 산점도
        ax1.scatter(y_true, y_pred, alpha=0.6, color='blue', s=50)
        
        # 완벽한 예측선 (y=x)
        min_val = min(np.min(y_true), np.min(y_pred))
        max_val = max(np.max(y_true), np.max(y_pred))
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='완벽한 예측')
        
        # 회귀선 추가
        z = np.polyfit(y_true, y_pred, 1)
        p = np.poly1d(z)
        ax1.plot(y_true, p(y_true), 'g-', alpha=0.8, label=f'회귀선 (기울기: {z[0]:.2f})')
        
        ax1.set_xlabel('실제 Brix 값')
        ax1.set_ylabel('예측 Brix 값')
        ax1.set_title(f'{title} - 산점도')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 메트릭 텍스트 추가
        if show_metrics:
            metrics = RegressionMetrics.compute_basic_metrics(y_true, y_pred)
            textstr = f"RMSE: {metrics['rmse']:.3f}\nMAE: {metrics['mae']:.3f}\nR²: {metrics['r2']:.3f}"
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
            ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=10,
                    verticalalignment='top', bbox=props)
        
        # 오차 히스토그램
        errors = y_pred - y_true
        ax2.hist(errors, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.axvline(np.mean(errors), color='red', linestyle='--', 
                   label=f'평균 오차: {np.mean(errors):.3f}')
        ax2.axvline(0, color='green', linestyle='-', alpha=0.8, label='완벽한 예측 (오차=0)')
        
        ax2.set_xlabel('예측 오차 (예측값 - 실제값)')
        ax2.set_ylabel('빈도')
        ax2.set_title('예측 오차 분포')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"시각화 저장: {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_residual_analysis(y_true: np.ndarray, 
                              y_pred: np.ndarray,
                              save_path: Optional[str] = None):
        """잔차 분석 플롯"""
        
        residuals = y_pred - y_true
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 잔차 vs 예측값
        axes[0, 0].scatter(y_pred, residuals, alpha=0.6)
        axes[0, 0].axhline(y=0, color='red', linestyle='--')
        axes[0, 0].set_xlabel('예측값')
        axes[0, 0].set_ylabel('잔차')
        axes[0, 0].set_title('잔차 vs 예측값')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 잔차 vs 실제값
        axes[0, 1].scatter(y_true, residuals, alpha=0.6)
        axes[0, 1].axhline(y=0, color='red', linestyle='--')
        axes[0, 1].set_xlabel('실제값')
        axes[0, 1].set_ylabel('잔차')
        axes[0, 1].set_title('잔차 vs 실제값')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 잔차 히스토그램
        axes[1, 0].hist(residuals, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[1, 0].axvline(np.mean(residuals), color='red', linestyle='--', 
                          label=f'평균: {np.mean(residuals):.3f}')
        axes[1, 0].set_xlabel('잔차')
        axes[1, 0].set_ylabel('빈도')
        axes[1, 0].set_title('잔차 분포')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Q-Q plot (정규성 검정)
        stats.probplot(residuals, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title('Q-Q Plot (잔차 정규성 검정)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"잔차 분석 저장: {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_class_performance(results: Dict[str, Any],
                              save_path: Optional[str] = None):
        """구간별 성능 비교 플롯"""
        
        class_names = ['Low', 'Medium', 'High', 'Very_High']
        class_labels = ['낮음 (< 9.0)', '보통 (9.0-10.5)', '높음 (10.5-12.0)', '매우 높음 (≥ 12.0)']
        
        # 데이터 준비
        counts = []
        rmse_values = []
        mae_values = []
        r2_values = []
        
        for class_name in class_names:
            count_key = f'{class_name}_count'
            if count_key in results and results[count_key] > 0:
                counts.append(results[count_key])
                rmse_values.append(results[f'{class_name}_rmse'])
                mae_values.append(results[f'{class_name}_mae'])
                r2_values.append(results[f'{class_name}_r2'])
            else:
                counts.append(0)
                rmse_values.append(0)
                mae_values.append(0)
                r2_values.append(0)
        
        # 플롯 생성
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 샘플 수
        axes[0, 0].bar(class_labels, counts, color='skyblue', alpha=0.8)
        axes[0, 0].set_title('구간별 샘플 수')
        axes[0, 0].set_ylabel('샘플 수')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. RMSE
        axes[0, 1].bar(class_labels, rmse_values, color='lightcoral', alpha=0.8)
        axes[0, 1].set_title('구간별 RMSE')
        axes[0, 1].set_ylabel('RMSE')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. MAE
        axes[1, 0].bar(class_labels, mae_values, color='lightgreen', alpha=0.8)
        axes[1, 0].set_title('구간별 MAE')
        axes[1, 0].set_ylabel('MAE')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. R²
        axes[1, 1].bar(class_labels, r2_values, color='gold', alpha=0.8)
        axes[1, 1].set_title('구간별 R²')
        axes[1, 1].set_ylabel('R²')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"구간별 성능 분석 저장: {save_path}")
        
        plt.show()
    
    @staticmethod
    def create_evaluation_report(results: Dict[str, Any],
                               save_dir: str = "evaluation_results"):
        """종합 평가 리포트 생성"""
        
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        
        y_true = results['targets']
        y_pred = results['predictions']
        dataset_name = results.get('dataset_name', 'Dataset')
        
        # 1. 예측 vs 실제값 플롯
        VisualizationTools.plot_prediction_vs_actual(
            y_true, y_pred,
            title=f"{dataset_name} - 예측 성능",
            save_path=save_dir / f"{dataset_name}_prediction_vs_actual.png"
        )
        
        # 2. 잔차 분석
        VisualizationTools.plot_residual_analysis(
            y_true, y_pred,
            save_path=save_dir / f"{dataset_name}_residual_analysis.png"
        )
        
        # 3. 구간별 성능
        VisualizationTools.plot_class_performance(
            results,
            save_path=save_dir / f"{dataset_name}_class_performance.png"
        )
        
        # 4. 텍스트 리포트 저장
        report_path = save_dir / f"{dataset_name}_evaluation_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"수박 당도 예측 모델 평가 리포트\n")
            f.write(f"{'='*50}\n")
            f.write(f"데이터셋: {dataset_name}\n")
            f.write(f"샘플 수: {results['sample_count']}\n\n")
            
            f.write(f"기본 회귀 메트릭:\n")
            f.write(f"  - RMSE: {results['rmse']:.4f}\n")
            f.write(f"  - MAE: {results['mae']:.4f}\n")
            f.write(f"  - R²: {results['r2']:.4f}\n")
            f.write(f"  - 조정된 R²: {results['adjusted_r2']:.4f}\n")
            f.write(f"  - MAPE: {results['mape']:.2f}%\n")
            f.write(f"  - NRMSE: {results['nrmse']:.2f}%\n\n")
            
            f.write(f"당도 범위별 정확도:\n")
            for key, value in results.items():
                if key.startswith('accuracy_within'):
                    tolerance = key.split('_')[2]
                    f.write(f"  - ±{tolerance} Brix 이내: {value:.1f}%\n")
            
            f.write(f"\n구간별 성능:\n")
            class_names = ['Low', 'Medium', 'High', 'Very_High']
            for class_name in class_names:
                count_key = f'{class_name}_count'
                if count_key in results and results[count_key] > 0:
                    rmse = results[f'{class_name}_rmse']
                    mae = results[f'{class_name}_mae']
                    r2 = results[f'{class_name}_r2']
                    count = results[count_key]
                    f.write(f"  - {class_name} 구간 (n={count}): RMSE={rmse:.3f}, MAE={mae:.3f}, R²={r2:.3f}\n")
        
        print(f"종합 평가 리포트 생성 완료: {save_dir}")


def compare_models(model_results: Dict[str, Dict[str, Any]],
                  save_path: Optional[str] = None):
    """여러 모델의 성능 비교"""
    
    model_names = list(model_results.keys())
    metrics = ['rmse', 'mae', 'r2', 'accuracy_within_0.5_brix', 'accuracy_within_1.0_brix']
    metric_labels = ['RMSE', 'MAE', 'R²', '±0.5 Brix 정확도', '±1.0 Brix 정확도']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        values = [model_results[name][metric] for name in model_names]
        
        bars = axes[i].bar(model_names, values, alpha=0.8)
        axes[i].set_title(f'모델별 {label} 비교')
        axes[i].set_ylabel(label)
        axes[i].tick_params(axis='x', rotation=45)
        
        # 막대 위에 값 표시
        for bar, value in zip(bars, values):
            height = bar.get_height()
            axes[i].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{value:.3f}', ha='center', va='bottom')
    
    # 빈 subplot 숨기기
    axes[5].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"모델 비교 결과 저장: {save_path}")
    
    plt.show()


if __name__ == "__main__":
    # 테스트 실행 예제
    print("Evaluation 모듈이 성공적으로 로딩되었습니다!")
    print("사용 예제:")
    print("evaluator = ModelEvaluator(model)")
    print("results = evaluator.evaluate_comprehensive(test_loader)")
    print("VisualizationTools.create_evaluation_report(results)") 