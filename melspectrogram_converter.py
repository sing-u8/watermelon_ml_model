#!/usr/bin/env python3
"""
수박 당도 판별 프로젝트 - 멜-스펙트로그램 변환 모듈
Mel-Spectrogram Conversion Module for Watermelon Brix Detection Project
"""

import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from typing import Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# 백엔드 설정 (GUI 없는 환경에서도 작동)
matplotlib.use('Agg')

class MelSpectrogramConverter:
    def __init__(self,
                 sr: int = 22050,
                 n_fft: int = 2048,
                 hop_length: int = 512,
                 n_mels: int = 128,
                 fmin: float = 0.0,
                 fmax: float = None,
                 power: float = 2.0):
        """
        멜-스펙트로그램 변환기 초기화
        
        Args:
            sr: 샘플링 레이트
            n_fft: FFT 윈도우 크기
            hop_length: 프레임 간격
            n_mels: 멜 필터 뱅크 수
            fmin: 최소 주파수
            fmax: 최대 주파수 (None이면 sr/2)
            power: 스펙트로그램의 거듭제곱 (1.0: 진폭, 2.0: 파워)
        """
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax if fmax is not None else sr // 2
        self.power = power
        
        # 멜 필터 뱅크 미리 계산
        self.mel_filter = librosa.filters.mel(
            sr=self.sr,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax
        )
        
    def audio_to_melspectrogram(self, audio: np.ndarray) -> np.ndarray:
        """
        오디오 신호를 멜-스펙트로그램으로 변환합니다.
        
        Args:
            audio: 입력 오디오 신호
            
        Returns:
            멜-스펙트로그램 (dB 스케일)
        """
        # STFT 계산
        stft = librosa.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window='hann',
            center=True,
            pad_mode='constant'
        )
        
        # 파워/진폭 스펙트로그램
        magnitude = np.abs(stft) ** self.power
        
        # 멜-스펙트로그램 변환
        mel_spectrogram = np.dot(self.mel_filter, magnitude)
        
        # dB 스케일로 변환
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
        
        return mel_spectrogram_db
    
    def save_melspectrogram_image(self, 
                                 mel_spec: np.ndarray, 
                                 save_path: Path,
                                 figsize: Tuple[int, int] = (10, 4),
                                 dpi: int = 100,
                                 cmap: str = 'viridis') -> None:
        """
        멜-스펙트로그램을 이미지로 저장합니다.
        
        Args:
            mel_spec: 멜-스펙트로그램 데이터
            save_path: 저장 경로
            figsize: 그림 크기
            dpi: 해상도
            cmap: 컬러맵
        """
        plt.figure(figsize=figsize, dpi=dpi)
        
        # 축과 프레임 제거하여 순수한 스펙트로그램만 저장
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        plt.axis('off')
        
        # 멜-스펙트로그램 표시
        librosa.display.specshow(
            mel_spec,
            sr=self.sr,
            hop_length=self.hop_length,
            x_axis='time',
            y_axis='mel',
            fmin=self.fmin,
            fmax=self.fmax,
            cmap=cmap
        )
        
        # 저장 디렉토리 생성
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 이미지 저장
        plt.savefig(
            str(save_path),
            dpi=dpi,
            bbox_inches='tight',
            pad_inches=0,
            facecolor='black'
        )
        plt.close()
    
    def save_melspectrogram_with_labels(self,
                                      mel_spec: np.ndarray,
                                      save_path: Path,
                                      title: str = "",
                                      figsize: Tuple[int, int] = (12, 6),
                                      dpi: int = 100,
                                      cmap: str = 'viridis') -> None:
        """
        레이블과 축이 포함된 멜-스펙트로그램을 저장합니다.
        
        Args:
            mel_spec: 멜-스펙트로그램 데이터
            save_path: 저장 경로
            title: 그래프 제목
            figsize: 그림 크기
            dpi: 해상도
            cmap: 컬러맵
        """
        plt.figure(figsize=figsize, dpi=dpi)
        
        # 멜-스펙트로그램 표시
        librosa.display.specshow(
            mel_spec,
            sr=self.sr,
            hop_length=self.hop_length,
            x_axis='time',
            y_axis='mel',
            fmin=self.fmin,
            fmax=self.fmax,
            cmap=cmap
        )
        
        plt.colorbar(format='%+2.0f dB')
        plt.title(title, fontsize=14, pad=20)
        plt.xlabel('Time (s)', fontsize=12)
        plt.ylabel('Mel Frequency', fontsize=12)
        plt.tight_layout()
        
        # 저장 디렉토리 생성
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 이미지 저장
        plt.savefig(str(save_path), dpi=dpi, bbox_inches='tight')
        plt.close()
    
    def process_audio_file(self, 
                          audio_path: Path, 
                          save_dir: Path,
                          save_raw_image: bool = True,
                          save_labeled_image: bool = True) -> Optional[np.ndarray]:
        """
        오디오 파일을 로드하고 멜-스펙트로그램으로 변환 후 저장합니다.
        
        Args:
            audio_path: 오디오 파일 경로
            save_dir: 저장 디렉토리
            save_raw_image: 순수 스펙트로그램 이미지 저장 여부
            save_labeled_image: 레이블 포함 이미지 저장 여부
            
        Returns:
            멜-스펙트로그램 배열 (실패시 None)
        """
        try:
            # 오디오 로드
            audio, sr = librosa.load(str(audio_path), sr=self.sr)
            
            # 멜-스펙트로그램 변환
            mel_spec = self.audio_to_melspectrogram(audio)
            
            print(f"📁 처리 중: {audio_path.name}")
            print(f"   - 오디오 길이: {len(audio)/sr:.2f}초")
            print(f"   - 스펙트로그램 크기: {mel_spec.shape}")
            
            # 저장 경로 설정
            base_name = audio_path.stem
            
            if save_raw_image:
                raw_save_path = save_dir / f"{base_name}_melspec.png"
                self.save_melspectrogram_image(mel_spec, raw_save_path)
                print(f"   💾 순수 이미지 저장: {raw_save_path}")
            
            if save_labeled_image:
                labeled_save_path = save_dir / f"{base_name}_melspec_labeled.png"
                title = f"Mel-Spectrogram: {audio_path.name}"
                self.save_melspectrogram_with_labels(mel_spec, labeled_save_path, title)
                print(f"   💾 레이블 이미지 저장: {labeled_save_path}")
            
            return mel_spec
            
        except Exception as e:
            print(f"❌ 처리 실패 ({audio_path.name}): {e}")
            return None
    
    def process_dataset(self,
                       data_root: Path,
                       output_root: Path,
                       use_preprocessed: bool = True) -> None:
        """
        전체 데이터셋을 멜-스펙트로그램으로 변환합니다.
        
        Args:
            data_root: 원본 데이터 루트 경로
            output_root: 출력 루트 경로
            use_preprocessed: 전처리된 데이터 사용 여부
        """
        print("🎼 데이터셋 멜-스펙트로그램 변환 시작...")
        output_root.mkdir(parents=True, exist_ok=True)
        
        if use_preprocessed:
            data_source = Path("preprocessed_data")
            print(f"📁 전처리된 데이터 사용: {data_source}")
        else:
            data_source = data_root
            print(f"📁 원본 데이터 사용: {data_source}")
        
        if not data_source.exists():
            print(f"❌ 데이터 폴더를 찾을 수 없습니다: {data_source}")
            return
        
        # 모든 수박 폴더 찾기
        if use_preprocessed:
            watermelon_folders = sorted([f for f in data_source.iterdir() 
                                       if f.is_dir() and '_' in f.name])
        else:
            watermelon_folders = sorted([f for f in data_source.iterdir() 
                                       if f.is_dir() and '_' in f.name])
        
        processed_count = 0
        failed_count = 0
        
        for folder in watermelon_folders:
            folder_name = folder.name
            print(f"\n📂 폴더 처리 중: {folder_name}")
            
            # 출력 폴더 생성
            output_folder = output_root / folder_name
            output_folder.mkdir(exist_ok=True)
            
            # 오디오 파일 찾기
            if use_preprocessed:
                audio_files = list(folder.glob("*.wav"))
            else:
                audio_folder = folder / 'audio'
                if audio_folder.exists():
                    audio_files = [f for f in audio_folder.iterdir() if f.is_file()]
                else:
                    audio_files = []
            
            for audio_file in audio_files:
                mel_spec = self.process_audio_file(
                    audio_file, 
                    output_folder,
                    save_raw_image=True,
                    save_labeled_image=False  # 용량 절약을 위해 레이블 이미지는 생략
                )
                
                if mel_spec is not None:
                    processed_count += 1
                else:
                    failed_count += 1
        
        print(f"\n🎉 멜-스펙트로그램 변환 완료!")
        print(f"✅ 성공: {processed_count}개")
        print(f"❌ 실패: {failed_count}개")
        print(f"📁 저장 위치: {output_root}")

def main():
    """메인 실행 함수"""
    print("🎼 수박 당도 판별 프로젝트 - 멜-스펙트로그램 변환")
    print("="*60)
    
    # 변환기 초기화
    converter = MelSpectrogramConverter(
        sr=22050,           # 22.05kHz
        n_fft=2048,         # FFT 크기
        hop_length=512,     # 홉 길이
        n_mels=128,         # 멜 필터 수
        fmin=0.0,           # 최소 주파수
        fmax=8000.0,        # 최대 주파수 (수박 타격음 대역)
        power=2.0           # 파워 스펙트로그램
    )
    
    # 경로 설정
    data_root = Path("watermelon_data")
    output_root = Path("melspectrogram_data")
    
    # 데이터셋 변환 (전처리된 데이터 사용)
    converter.process_dataset(
        data_root=data_root,
        output_root=output_root,
        use_preprocessed=True
    )
    
    # 샘플 변환 테스트
    print("\n📋 샘플 변환 테스트...")
    preprocessed_folder = Path("preprocessed_data/01_10.5")
    test_output = Path("test_melspectrogram")
    
    if preprocessed_folder.exists():
        sample_files = list(preprocessed_folder.glob("*.wav"))[:2]  # 첫 2개 파일만
        
        for sample_file in sample_files:
            print(f"\n🔧 테스트: {sample_file.name}")
            mel_spec = converter.process_audio_file(
                sample_file, 
                test_output,
                save_raw_image=True,
                save_labeled_image=True
            )
            if mel_spec is not None:
                print(f"   결과: {mel_spec.shape[0]}개 멜 밴드, {mel_spec.shape[1]}개 시간 프레임")
                print(f"   범위: {mel_spec.min():.2f} ~ {mel_spec.max():.2f} dB")

if __name__ == "__main__":
    main() 