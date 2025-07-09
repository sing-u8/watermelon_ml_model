"""
수박 당도 예측을 위한 훈련 파이프라인

이 모듈은 다음 기능을 제공합니다:
1. Trainer 클래스: 훈련 및 검증 프로세스 관리
2. 얼리 스토핑 구현
3. 모델 체크포인트 및 저장/로딩
4. 훈련 로깅 및 모니터링
5. GPU 지원 및 혼합 정밀도 훈련

Author: AI Assistant
Date: 2024
"""

import os
import time
import json
import logging
from typing import Dict, List, Optional, Tuple, Any, Callable
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from ..utils.font_utils import setup_korean_font

# 한글 폰트 설정
setup_korean_font()

from ..models.cnn_models import ModelFactory, LossManager, OptimizerManager
from ..data.dataset import WatermelonDataset


class EarlyStopping:
    """얼리 스토핑 구현 클래스"""
    
    def __init__(self, 
                 patience: int = 10,
                 min_delta: float = 0.001,
                 mode: str = 'min',
                 verbose: bool = True):
        """
        Args:
            patience (int): 개선이 없어도 기다릴 epoch 수
            min_delta (float): 개선으로 간주할 최소 변화량
            mode (str): 'min' (손실) 또는 'max' (정확도) 모드
            verbose (bool): 메시지 출력 여부
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
        if mode == 'min':
            self.monitor_op = np.less
            self.min_delta *= -1
        else:
            self.monitor_op = np.greater
            
    def __call__(self, score: float) -> bool:
        """
        스코어를 평가하여 얼리 스토핑 여부 결정
        
        Args:
            score (float): 현재 검증 스코어
            
        Returns:
            bool: 얼리 스토핑 여부
        """
        if self.best_score is None:
            self.best_score = score
        elif self.monitor_op(score, self.best_score + self.min_delta):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
                
            if self.counter >= self.patience:
                self.early_stop = True
                
        return self.early_stop


class ModelCheckpoint:
    """모델 체크포인트 관리 클래스"""
    
    def __init__(self, 
                 checkpoint_dir: str = "checkpoints",
                 save_best_only: bool = True,
                 mode: str = 'min',
                 verbose: bool = True):
        """
        Args:
            checkpoint_dir (str): 체크포인트 저장 디렉토리
            save_best_only (bool): 최고 성능 모델만 저장할지 여부
            mode (str): 'min' (손실) 또는 'max' (정확도) 모드
            verbose (bool): 메시지 출력 여부
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.save_best_only = save_best_only
        self.mode = mode
        self.verbose = verbose
        
        self.best_score = None
        
        if mode == 'min':
            self.monitor_op = np.less
        else:
            self.monitor_op = np.greater
            
    def save_checkpoint(self, 
                       model: nn.Module,
                       optimizer: optim.Optimizer,
                       scheduler: Optional[Any],
                       epoch: int,
                       score: float,
                       train_history: Dict,
                       model_name: str = "model") -> bool:
        """
        체크포인트 저장
        
        Args:
            model: 저장할 모델
            optimizer: 옵티마이저
            scheduler: 스케줄러
            epoch: 현재 epoch
            score: 현재 검증 스코어
            train_history: 훈련 기록
            model_name: 모델 이름
            
        Returns:
            bool: 저장 여부
        """
        is_best = False
        
        if self.best_score is None or self.monitor_op(score, self.best_score):
            self.best_score = score
            is_best = True
            
        if not self.save_best_only or is_best:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'score': score,
                'best_score': self.best_score,
                'train_history': train_history
            }
            
            if is_best:
                save_path = self.checkpoint_dir / f"{model_name}_best.pth"
                if self.verbose:
                    print(f"새로운 최고 성능! 모델 저장: {save_path}")
            else:
                save_path = self.checkpoint_dir / f"{model_name}_epoch_{epoch}.pth"
                
            torch.save(checkpoint, save_path)
            
        return is_best
    
    def load_checkpoint(self, 
                       model: nn.Module,
                       optimizer: optim.Optimizer,
                       scheduler: Optional[Any],
                       checkpoint_path: str) -> Dict:
        """체크포인트 로딩"""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if scheduler and checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        return checkpoint


class Trainer:
    """수박 당도 예측 모델 훈련 클래스"""
    
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 test_loader: Optional[DataLoader] = None,
                 optimizer: Optional[optim.Optimizer] = None,
                 scheduler: Optional[Any] = None,
                 loss_fn: Optional[nn.Module] = None,
                 device: Optional[torch.device] = None,
                 checkpoint_dir: str = "checkpoints",
                 log_dir: str = "logs",
                 use_amp: bool = True,
                 verbose: bool = True):
        """
        Args:
            model: 훈련할 모델
            train_loader: 훈련 데이터 로더
            val_loader: 검증 데이터 로더
            test_loader: 테스트 데이터 로더 (선택)
            optimizer: 옵티마이저 (None이면 Adam 사용)
            scheduler: 학습률 스케줄러 (선택)
            loss_fn: 손실함수 (None이면 MSE 사용)
            device: 디바이스 (None이면 자동 선택)
            checkpoint_dir: 체크포인트 저장 디렉토리
            log_dir: 로그 저장 디렉토리
            use_amp: 혼합 정밀도 훈련 사용 여부
            verbose: 상세 출력 여부
        """
        # 디바이스 설정
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        print(f"사용 디바이스: {self.device}")
        
        # 모델 설정
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        # 옵티마이저 설정
        if optimizer is None:
            self.optimizer = OptimizerManager.get_optimizer(
                self.model, 'adam', lr=0.001, weight_decay=1e-4
            )
        else:
            self.optimizer = optimizer
            
        # 스케줄러 설정
        self.scheduler = scheduler
        
        # 손실함수 설정
        if loss_fn is None:
            self.loss_fn = LossManager.get_loss_function('mse')
        else:
            self.loss_fn = loss_fn
            
        # 디렉토리 설정
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir = Path(log_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.log_dir.mkdir(exist_ok=True)
        
        # 혼합 정밀도 설정
        self.use_amp = use_amp and torch.cuda.is_available()
        if self.use_amp:
            self.scaler = GradScaler()
            
        # 체크포인트 관리자
        self.checkpoint_manager = ModelCheckpoint(
            checkpoint_dir=checkpoint_dir,
            save_best_only=True,
            mode='min',
            verbose=verbose
        )
        
        # 훈련 기록
        self.train_history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'epoch_time': []
        }
        
        self.verbose = verbose
        
        # 로깅 설정
        self._setup_logging()
        
    def _setup_logging(self):
        """로깅 설정"""
        log_file = self.log_dir / "training.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler() if self.verbose else logging.NullHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        
    def train_epoch(self) -> float:
        """한 epoch 훈련"""
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        with tqdm(self.train_loader, desc="Training", disable=not self.verbose) as pbar:
            for batch_idx, (inputs, targets) in enumerate(pbar):
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                
                self.optimizer.zero_grad()
                
                if self.use_amp:
                    with autocast():
                        outputs = self.model(inputs)
                        loss = self.loss_fn(outputs.squeeze(), targets)
                    
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    outputs = self.model(inputs)
                    loss = self.loss_fn(outputs.squeeze(), targets)
                    
                    loss.backward()
                    self.optimizer.step()
                
                total_loss += loss.item()
                
                # 진행률 업데이트
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Avg Loss': f'{total_loss/(batch_idx+1):.4f}'
                })
                
        return total_loss / num_batches
    
    def validate_epoch(self) -> float:
        """한 epoch 검증"""
        self.model.eval()
        total_loss = 0.0
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            with tqdm(self.val_loader, desc="Validation", disable=not self.verbose) as pbar:
                for inputs, targets in pbar:
                    inputs = inputs.to(self.device, non_blocking=True)
                    targets = targets.to(self.device, non_blocking=True)
                    
                    if self.use_amp:
                        with autocast():
                            outputs = self.model(inputs)
                            loss = self.loss_fn(outputs.squeeze(), targets)
                    else:
                        outputs = self.model(inputs)
                        loss = self.loss_fn(outputs.squeeze(), targets)
                    
                    total_loss += loss.item()
                    
                    pbar.set_postfix({'Val Loss': f'{loss.item():.4f}'})
                    
        return total_loss / num_batches
    
    def fit(self,
            epochs: int,
            early_stopping_patience: Optional[int] = 15,
            save_best_model: bool = True) -> Dict[str, List[float]]:
        """
        모델 훈련
        
        Args:
            epochs: 훈련 에포크 수
            early_stopping_patience: 얼리 스토핑 patience (None이면 비활성화)
            save_best_model: 최고 성능 모델 저장 여부
            
        Returns:
            Dict: 훈련 기록
        """
        # 얼리 스토핑 설정
        early_stopping = None
        if early_stopping_patience:
            early_stopping = EarlyStopping(
                patience=early_stopping_patience,
                min_delta=0.001,
                mode='min',
                verbose=self.verbose
            )
        
        self.logger.info(f"훈련 시작 - Epochs: {epochs}, Device: {self.device}")
        self.logger.info(f"모델 파라미터 수: {sum(p.numel() for p in self.model.parameters())}")
        
        start_time = time.time()
        
        for epoch in range(epochs):
            epoch_start_time = time.time()
            
            # 훈련
            train_loss = self.train_epoch()
            
            # 검증
            val_loss = self.validate_epoch()
            
            # 스케줄러 업데이트
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # 기록 업데이트
            epoch_time = time.time() - epoch_start_time
            current_lr = self.optimizer.param_groups[0]['lr']
            
            self.train_history['train_loss'].append(train_loss)
            self.train_history['val_loss'].append(val_loss)
            self.train_history['learning_rate'].append(current_lr)
            self.train_history['epoch_time'].append(epoch_time)
            
            # 로깅
            self.logger.info(
                f"Epoch {epoch+1}/{epochs} - "
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"LR: {current_lr:.6f}, "
                f"Time: {epoch_time:.2f}s"
            )
            
            # 체크포인트 저장
            if save_best_model:
                is_best = self.checkpoint_manager.save_checkpoint(
                    self.model, self.optimizer, self.scheduler,
                    epoch + 1, val_loss, self.train_history,
                    model_name=self.model.__class__.__name__
                )
            
            # 얼리 스토핑 체크
            if early_stopping and early_stopping(val_loss):
                self.logger.info(f"얼리 스토핑 - Epoch {epoch+1}")
                break
                
        total_time = time.time() - start_time
        self.logger.info(f"훈련 완료 - 총 시간: {total_time:.2f}s")
        
        return self.train_history
    
    def evaluate(self, 
                test_loader: Optional[DataLoader] = None,
                return_predictions: bool = False) -> Dict[str, float]:
        """
        모델 평가
        
        Args:
            test_loader: 테스트 데이터 로더 (None이면 검증 데이터 사용)
            return_predictions: 예측값 반환 여부
            
        Returns:
            Dict: 평가 결과
        """
        if test_loader is None:
            test_loader = self.val_loader
            
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        total_loss = 0.0
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs.squeeze(), targets)
                
                total_loss += loss.item()
                all_predictions.extend(outputs.squeeze().cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        predictions = np.array(all_predictions)
        targets = np.array(all_targets)
        
        # 기본 메트릭 계산
        mse = np.mean((predictions - targets) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - targets))
        
        # R-squared 계산
        ss_res = np.sum((targets - predictions) ** 2)
        ss_tot = np.sum((targets - np.mean(targets)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        results = {
            'loss': total_loss / len(test_loader),
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
        
        self.logger.info(f"평가 결과 - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
        
        if return_predictions:
            results['predictions'] = predictions
            results['targets'] = targets
            
        return results
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """훈련 기록 시각화"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 손실 곡선
        axes[0, 0].plot(self.train_history['train_loss'], label='Train Loss', color='blue')
        axes[0, 0].plot(self.train_history['val_loss'], label='Val Loss', color='red')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 학습률 곡선
        axes[0, 1].plot(self.train_history['learning_rate'], color='green')
        axes[0, 1].set_title('Learning Rate')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Learning Rate')
        axes[0, 1].set_yscale('log')
        axes[0, 1].grid(True)
        
        # Epoch 시간
        axes[1, 0].plot(self.train_history['epoch_time'], color='orange')
        axes[1, 0].set_title('Training Time per Epoch')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Time (seconds)')
        axes[1, 0].grid(True)
        
        # 손실 비교 (확대)
        if len(self.train_history['train_loss']) > 10:
            last_10_epochs = list(range(len(self.train_history['train_loss']) - 10, len(self.train_history['train_loss'])))
            axes[1, 1].plot(last_10_epochs, self.train_history['train_loss'][-10:], 
                          label='Train Loss', color='blue')
            axes[1, 1].plot(last_10_epochs, self.train_history['val_loss'][-10:], 
                          label='Val Loss', color='red')
            axes[1, 1].set_title('Loss (Last 10 Epochs)')
        else:
            axes[1, 1].plot(self.train_history['train_loss'], label='Train Loss', color='blue')
            axes[1, 1].plot(self.train_history['val_loss'], label='Val Loss', color='red')
            axes[1, 1].set_title('Training and Validation Loss')
            
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"훈련 기록 그래프 저장: {save_path}")
        
        plt.show()
    
    def save_model(self, save_path: str, include_optimizer: bool = False):
        """모델 저장"""
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'model_class': self.model.__class__.__name__,
            'train_history': self.train_history
        }
        
        if include_optimizer:
            save_dict['optimizer_state_dict'] = self.optimizer.state_dict()
            if self.scheduler:
                save_dict['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(save_dict, save_path)
        self.logger.info(f"모델 저장: {save_path}")
    
    def load_model(self, load_path: str, load_optimizer: bool = False):
        """모델 로딩"""
        checkpoint = torch.load(load_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
        if load_optimizer and self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        if 'train_history' in checkpoint:
            self.train_history = checkpoint['train_history']
            
        self.logger.info(f"모델 로딩: {load_path}")


def create_trainer(model_type: str = 'vgg16',
                  train_loader: DataLoader = None,
                  val_loader: DataLoader = None,
                  test_loader: Optional[DataLoader] = None,
                  **kwargs) -> Trainer:
    """
    Trainer 인스턴스 생성 헬퍼 함수
    
    Args:
        model_type: 모델 타입 ('vgg16', 'custom_cnn')
        train_loader: 훈련 데이터 로더
        val_loader: 검증 데이터 로더
        test_loader: 테스트 데이터 로더
        **kwargs: 추가 설정
        
    Returns:
        Trainer: 설정된 Trainer 인스턴스
    """
    # 모델 생성에 필요한 파라미터만 필터링
    if model_type.lower() == 'vgg16':
        model_params = {k: v for k, v in kwargs.items() 
                       if k in ['num_classes', 'dropout', 'freeze_features', 'use_batch_norm', 'use_pretrained']}
    elif model_type.lower() == 'custom_cnn':
        model_params = {k: v for k, v in kwargs.items() 
                       if k in ['input_channels', 'num_classes', 'dropout', 'use_residual']}
    else:
        model_params = {}
    
    # 모델 생성
    model = ModelFactory.create_model(model_type, **model_params)
    
    # 옵티마이저 생성에 필요한 파라미터 필터링
    optimizer_type = kwargs.get('optimizer', 'adam')
    optimizer_params = {k: v for k, v in kwargs.items() 
                       if k in ['lr', 'learning_rate', 'weight_decay', 'momentum', 'betas', 'eps']}
    # lr과 learning_rate 통일
    if 'learning_rate' in optimizer_params and 'lr' not in optimizer_params:
        optimizer_params['lr'] = optimizer_params.pop('learning_rate')
    
    optimizer = OptimizerManager.get_optimizer(model, optimizer_type, **optimizer_params)
    
    # 스케줄러 생성에 필요한 파라미터 필터링
    scheduler_type = kwargs.get('scheduler', None)
    scheduler = None
    if scheduler_type:
        scheduler_params = {k: v for k, v in kwargs.items() 
                           if k in ['step_size', 'gamma', 'patience', 'factor', 'T_max', 'eta_min']}
        scheduler = OptimizerManager.get_scheduler(optimizer, scheduler_type, **scheduler_params)
    
    # 손실함수 생성 (대부분 추가 파라미터 불필요)
    loss_type = kwargs.get('loss_function', 'mse')
    loss_params = {k: v for k, v in kwargs.items() if k in ['eps', 'reduction']}
    loss_fn = LossManager.get_loss_function(loss_type, **loss_params)
    
    # Trainer 생성
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        **{k: v for k, v in kwargs.items() 
           if k in ['device', 'checkpoint_dir', 'log_dir', 'use_amp', 'verbose']}
    )
    
    return trainer


if __name__ == "__main__":
    # 테스트 실행 예제
    print("Trainer 모듈이 성공적으로 로딩되었습니다!")
    print("사용 예제:")
    print("trainer = create_trainer('vgg16', train_loader, val_loader)")
    print("history = trainer.fit(epochs=50)") 