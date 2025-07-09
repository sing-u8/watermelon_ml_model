"""
수박 당도 예측을 위한 CNN 모델 설계

이 모듈은 다음 기능을 제공합니다:
1. VGG-16 기반 전이학습 모델 (회귀용으로 수정)
2. 수박 스펙트로그램 특화 커스텀 CNN 모델
3. 손실함수 및 옵티마이저 설정
4. 모델 초기화 및 관리 유틸리티

Author: AI Assistant  
Date: 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR
import torchvision.models as models
from typing import Dict, Any, Optional, Tuple
import math


class VGG16Regressor(nn.Module):
    """
    VGG-16 기반 회귀 모델
    
    사전훈련된 VGG-16 모델을 활용하여 수박 당도 예측을 위한 회귀 모델로 수정.
    분류기 부분을 회귀용으로 교체하고 Fine-tuning 전략을 적용.
    
    Features:
    - 사전훈련된 특성 추출기 활용
    - 배치 정규화 및 드롭아웃 추가
    - 유연한 Fine-tuning 전략 (특성 추출기 고정/해제)
    """
    
    def __init__(self, 
                 num_classes: int = 1,
                 dropout: float = 0.5,
                 freeze_features: bool = False,
                 use_batch_norm: bool = True,
                 use_pretrained: bool = True):
        """
        Args:
            num_classes (int): 출력 차원 (회귀의 경우 1)
            dropout (float): 드롭아웃 비율
            freeze_features (bool): 특성 추출기를 고정할지 여부
            use_batch_norm (bool): 배치 정규화 사용 여부
            use_pretrained (bool): 사전훈련된 가중치 사용 여부
        """
        super(VGG16Regressor, self).__init__()
        
        # VGG-16 로딩 (사전훈련 여부 선택 가능)
        try:
            if use_pretrained:
                self.vgg16 = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
            else:
                self.vgg16 = models.vgg16(weights=None)
        except Exception as e:
            print(f"사전훈련된 모델 로딩 실패: {e}")
            print("사전훈련되지 않은 모델을 사용합니다.")
            self.vgg16 = models.vgg16(weights=None)
        
        # 특성 추출기 고정 설정
        if freeze_features:
            for param in self.vgg16.features.parameters():
                param.requires_grad = False
        
        # 분류기를 회귀용으로 교체
        # VGG-16의 원래 분류기: Linear(25088, 4096) -> ReLU -> Dropout -> Linear(4096, 4096) -> ReLU -> Dropout -> Linear(4096, 1000)
        
        classifier_layers = []
        
        # 첫 번째 완전연결층
        classifier_layers.append(nn.Linear(25088, 4096))
        if use_batch_norm:
            classifier_layers.append(nn.BatchNorm1d(4096))
        classifier_layers.append(nn.ReLU(inplace=True))
        classifier_layers.append(nn.Dropout(dropout))
        
        # 두 번째 완전연결층
        classifier_layers.append(nn.Linear(4096, 1024))
        if use_batch_norm:
            classifier_layers.append(nn.BatchNorm1d(1024))
        classifier_layers.append(nn.ReLU(inplace=True))
        classifier_layers.append(nn.Dropout(dropout))
        
        # 세 번째 완전연결층 (중간 차원)
        classifier_layers.append(nn.Linear(1024, 256))
        if use_batch_norm:
            classifier_layers.append(nn.BatchNorm1d(256))
        classifier_layers.append(nn.ReLU(inplace=True))
        classifier_layers.append(nn.Dropout(dropout / 2))  # 마지막 드롭아웃은 절반으로
        
        # 출력층 (회귀)
        classifier_layers.append(nn.Linear(256, num_classes))
        
        # 기존 분류기 교체
        self.vgg16.classifier = nn.Sequential(*classifier_layers)
        
        # 가중치 초기화
        self._initialize_weights()
        
    def _initialize_weights(self):
        """분류기 가중치를 He 초기화로 설정"""
        for m in self.vgg16.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        순전파
        
        Args:
            x (torch.Tensor): 입력 이미지 텐서 [B, C, H, W]
            
        Returns:
            torch.Tensor: 예측된 Brix 값 [B, 1]
        """
        return self.vgg16(x)
    
    def unfreeze_features(self, unfreeze_from_layer: int = -1):
        """
        특성 추출기의 일부 또는 전체를 학습 가능하게 설정
        
        Args:
            unfreeze_from_layer (int): 해제할 시작 레이어 (-1이면 전체 해제)
        """
        layers = list(self.vgg16.features.children())
        
        if unfreeze_from_layer == -1:
            # 전체 해제
            for param in self.vgg16.features.parameters():
                param.requires_grad = True
        else:
            # 특정 레이어부터 해제
            for i, layer in enumerate(layers):
                if i >= unfreeze_from_layer:
                    for param in layer.parameters():
                        param.requires_grad = True


class WatermelonCNN(nn.Module):
    """
    수박 스펙트로그램 특화 커스텀 CNN 모델
    
    수박 타격음 스펙트로그램의 특성을 고려한 커스텀 아키텍처.
    잔차 연결과 글로벌 평균 풀링을 활용하여 과적합을 방지.
    
    Features:
    - 점진적 채널 확장 (32 -> 64 -> 128 -> 256)
    - 잔차 연결 (ResNet 스타일)
    - 글로벌 평균 풀링으로 매개변수 수 감소
    - 배치 정규화 및 드롭아웃
    """
    
    def __init__(self, 
                 input_channels: int = 3,
                 num_classes: int = 1,
                 dropout: float = 0.3,
                 use_residual: bool = True):
        """
        Args:
            input_channels (int): 입력 채널 수 (RGB=3)
            num_classes (int): 출력 차원 (회귀의 경우 1)
            dropout (float): 드롭아웃 비율
            use_residual (bool): 잔차 연결 사용 여부
        """
        super(WatermelonCNN, self).__init__()
        
        self.use_residual = use_residual
        
        # 첫 번째 합성곱 블록 (3 -> 32)
        self.conv1 = self._make_conv_block(input_channels, 32, kernel_size=7, stride=2, padding=3)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # 두 번째 합성곱 블록 (32 -> 64)
        self.conv2_1 = self._make_conv_block(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = self._make_conv_block(64, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 세 번째 합성곱 블록 (64 -> 128) with residual
        self.conv3_1 = self._make_conv_block(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv3_2 = self._make_conv_block(128, 128, kernel_size=3, stride=1, padding=1)
        
        # 네 번째 합성곱 블록 (128 -> 256) with residual
        self.conv4_1 = self._make_conv_block(128, 256, kernel_size=3, stride=2, padding=1)
        self.conv4_2 = self._make_conv_block(256, 256, kernel_size=3, stride=1, padding=1)
        
        # 잔차 연결을 위한 1x1 합성곱
        if use_residual:
            self.shortcut3 = nn.Conv2d(64, 128, kernel_size=1, stride=2, bias=False)
            self.shortcut4 = nn.Conv2d(128, 256, kernel_size=1, stride=2, bias=False)
        
        # 글로벌 평균 풀링
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 완전연결 분류기
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 4),
            nn.Linear(64, num_classes)
        )
        
        # 가중치 초기화
        self._initialize_weights()
    
    def _make_conv_block(self, in_channels: int, out_channels: int, 
                        kernel_size: int = 3, stride: int = 1, padding: int = 1) -> nn.Sequential:
        """합성곱 블록 생성 (Conv -> BatchNorm -> ReLU)"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def _initialize_weights(self):
        """가중치 초기화 (He 초기화)"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        순전파
        
        Args:
            x (torch.Tensor): 입력 이미지 텐서 [B, C, H, W]
            
        Returns:
            torch.Tensor: 예측된 Brix 값 [B, 1]
        """
        # 첫 번째 블록
        x = self.conv1(x)
        x = self.pool1(x)
        
        # 두 번째 블록
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.pool2(x)
        
        # 세 번째 블록 (잔차 연결)
        if self.use_residual:
            identity = self.shortcut3(x)
            x = self.conv3_1(x)
            x = self.conv3_2(x)
            x = x + identity
        else:
            x = self.conv3_1(x)
            x = self.conv3_2(x)
        
        # 네 번째 블록 (잔차 연결)
        if self.use_residual:
            identity = self.shortcut4(x)
            x = self.conv4_1(x)
            x = self.conv4_2(x)
            x = x + identity
        else:
            x = self.conv4_1(x)
            x = self.conv4_2(x)
        
        # 글로벌 평균 풀링
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        
        # 분류기
        x = self.classifier(x)
        
        return x


class ModelFactory:
    """모델 생성을 위한 팩토리 클래스"""
    
    @staticmethod
    def create_model(model_type: str, **kwargs) -> nn.Module:
        """
        모델 타입에 따라 모델 생성
        
        Args:
            model_type (str): 모델 타입 ('vgg16', 'custom_cnn')
            **kwargs: 모델별 추가 인자
            
        Returns:
            nn.Module: 생성된 모델
        """
        if model_type.lower() == 'vgg16':
            return VGG16Regressor(**kwargs)
        elif model_type.lower() == 'custom_cnn':
            return WatermelonCNN(**kwargs)
        else:
            raise ValueError(f"지원되지 않는 모델 타입: {model_type}")


class LossManager:
    """
    손실함수 관리 클래스
    
    회귀 문제에 적합한 다양한 손실함수를 제공하고
    필요에 따라 가중치나 정규화를 적용할 수 있습니다.
    """
    
    @staticmethod
    def get_loss_function(loss_type: str, **kwargs) -> nn.Module:
        """
        손실함수 생성
        
        Args:
            loss_type (str): 손실함수 타입
            **kwargs: 손실함수별 추가 인자
            
        Returns:
            nn.Module: 손실함수
        """
        if loss_type.lower() == 'mse':
            return nn.MSELoss(**kwargs)
        elif loss_type.lower() == 'mae' or loss_type.lower() == 'l1':
            return nn.L1Loss(**kwargs)
        elif loss_type.lower() == 'huber' or loss_type.lower() == 'smooth_l1':
            return nn.SmoothL1Loss(**kwargs)
        elif loss_type.lower() == 'rmse':
            return RMSELoss(**kwargs)
        else:
            raise ValueError(f"지원되지 않는 손실함수: {loss_type}")


class RMSELoss(nn.Module):
    """Root Mean Square Error 손실함수"""
    
    def __init__(self, eps: float = 1e-8):
        super(RMSELoss, self).__init__()
        self.eps = eps
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(F.mse_loss(pred, target) + self.eps)


class OptimizerManager:
    """
    옵티마이저 관리 클래스
    
    다양한 옵티마이저와 학습률 스케줄러를 제공합니다.
    """
    
    @staticmethod
    def get_optimizer(model: nn.Module, optimizer_type: str, **kwargs) -> optim.Optimizer:
        """
        옵티마이저 생성
        
        Args:
            model (nn.Module): 모델
            optimizer_type (str): 옵티마이저 타입
            **kwargs: 옵티마이저별 추가 인자
            
        Returns:
            optim.Optimizer: 옵티마이저
        """
        if optimizer_type.lower() == 'adam':
            return optim.Adam(model.parameters(), **kwargs)
        elif optimizer_type.lower() == 'adamw':
            return optim.AdamW(model.parameters(), **kwargs)
        elif optimizer_type.lower() == 'sgd':
            return optim.SGD(model.parameters(), **kwargs)
        elif optimizer_type.lower() == 'rmsprop':
            return optim.RMSprop(model.parameters(), **kwargs)
        else:
            raise ValueError(f"지원되지 않는 옵티마이저: {optimizer_type}")
    
    @staticmethod
    def get_scheduler(optimizer: optim.Optimizer, scheduler_type: str, **kwargs):
        """
        학습률 스케줄러 생성
        
        Args:
            optimizer (optim.Optimizer): 옵티마이저
            scheduler_type (str): 스케줄러 타입
            **kwargs: 스케줄러별 추가 인자
            
        Returns:
            학습률 스케줄러
        """
        if scheduler_type.lower() == 'step':
            return StepLR(optimizer, **kwargs)
        elif scheduler_type.lower() == 'plateau':
            return ReduceLROnPlateau(optimizer, **kwargs)
        elif scheduler_type.lower() == 'cosine':
            return CosineAnnealingLR(optimizer, **kwargs)
        else:
            raise ValueError(f"지원되지 않는 스케줄러: {scheduler_type}")


def get_model_info(model: nn.Module) -> Dict[str, Any]:
    """
    모델 정보 출력
    
    Args:
        model (nn.Module): 모델
        
    Returns:
        Dict[str, Any]: 모델 정보
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'model_type': model.__class__.__name__,
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'frozen_parameters': total_params - trainable_params,
        'model_size_mb': total_params * 4 / (1024 * 1024)  # 4 bytes per float32
    }


def print_model_summary(model: nn.Module, input_size: Tuple[int, ...] = (3, 224, 224)):
    """
    모델 요약 정보 출력
    
    Args:
        model (nn.Module): 모델
        input_size (Tuple[int, ...]): 입력 크기 (C, H, W)
    """
    info = get_model_info(model)
    
    print(f"\n{'='*50}")
    print(f"모델 요약: {info['model_type']}")
    print(f"{'='*50}")
    print(f"총 매개변수 수: {info['total_parameters']:,}")
    print(f"학습 가능한 매개변수: {info['trainable_parameters']:,}")
    print(f"고정된 매개변수: {info['frozen_parameters']:,}")
    print(f"모델 크기: {info['model_size_mb']:.2f} MB")
    
    # 샘플 입력으로 순전파 테스트
    try:
        model.eval()
        with torch.no_grad():
            sample_input = torch.randn(1, *input_size)
            output = model(sample_input)
            print(f"입력 크기: {list(sample_input.shape)}")
            print(f"출력 크기: {list(output.shape)}")
    except Exception as e:
        print(f"순전파 테스트 실패: {e}")
    
    print(f"{'='*50}\n")


if __name__ == "__main__":
    # 모델 테스트 예제
    print("🔧 모델 설계 테스트")
    
    # VGG-16 모델 테스트
    print("\n1. VGG-16 기반 회귀 모델")
    vgg_model = ModelFactory.create_model('vgg16', dropout=0.5, freeze_features=False, use_pretrained=False)
    print_model_summary(vgg_model)
    
    # 커스텀 CNN 모델 테스트
    print("\n2. 커스텀 CNN 모델")
    custom_model = ModelFactory.create_model('custom_cnn', dropout=0.3, use_residual=True)
    print_model_summary(custom_model)
    
    # 손실함수 테스트
    print("\n3. 손실함수 테스트")
    loss_fns = ['mse', 'mae', 'huber', 'rmse']
    for loss_type in loss_fns:
        loss_fn = LossManager.get_loss_function(loss_type)
        print(f"  - {loss_type.upper()}: {loss_fn.__class__.__name__}")
    
    # 옵티마이저 테스트
    print("\n4. 옵티마이저 테스트")
    optimizers = ['adam', 'adamw', 'sgd']
    for opt_type in optimizers:
        optimizer = OptimizerManager.get_optimizer(custom_model, opt_type, lr=0.001)
        print(f"  - {opt_type.upper()}: {optimizer.__class__.__name__}")
    
    print("\n✅ 모델 설계 완료!") 