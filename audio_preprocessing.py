#!/usr/bin/env python3
"""
수박 당도 판별 프로젝트 - 오디오 전처리 파이프라인
Audio Preprocessing Pipeline for Watermelon Brix Detection Project
"""

import os
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from scipy import signal
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class AudioPreprocessor:
    def __init__(self, 
                 target_sr: int = 22050,
                 target_duration: float = 3.0,
                 normalize: bool = True,
                 apply_noise_reduction: bool = True):
        """
        오디오 전처리기 초기화
        
        Args:
            target_sr: 목표 샘플링 레이트 (Hz)
            target_duration: 목표 길이 (초)
            normalize: 정규화 적용 여부
            apply_noise_reduction: 노이즈 제거 적용 여부
        """
        self.target_sr = target_sr
        self.target_duration = target_duration
        self.target_length = int(target_sr * target_duration)
        self.normalize = normalize
        self.apply_noise_reduction = apply_noise_reduction
        
    def load_audio(self, file_path: Path) -> Tuple[np.ndarray, int]:
        """
        오디오 파일을 로드합니다.
        
        Args:
            file_path: 오디오 파일 경로
            
        Returns:
            Tuple[오디오 신호, 샘플링 레이트]
        """
        try:
            # Path 객체를 문자열로 변환
            audio_signal, sample_rate = librosa.load(str(file_path), sr=self.target_sr)
            return audio_signal, sample_rate
        except Exception as e:
            print(f"❌ 오디오 로드 실패 ({file_path.name}): {e}")
            return None, None
    
    def remove_noise(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        스펙트럴 서브트랙션을 이용한 노이즈 제거
        
        Args:
            audio: 입력 오디오 신호
            sr: 샘플링 레이트
            
        Returns:
            노이즈가 제거된 오디오 신호
        """
        # 짧은 시간 푸리에 변환
        stft = librosa.stft(audio)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # 노이즈 추정 (첫 번째와 마지막 프레임들의 평균)
        noise_frames = 5
        noise_magnitude = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)
        
        # 스펙트럴 서브트랙션
        alpha = 2.0  # 과소 제거 계수
        beta = 0.01  # 잔여 노이즈 플로어
        
        cleaned_magnitude = magnitude - alpha * noise_magnitude
        cleaned_magnitude = np.maximum(cleaned_magnitude, beta * magnitude)
        
        # 역변환
        cleaned_stft = cleaned_magnitude * np.exp(1j * phase)
        cleaned_audio = librosa.istft(cleaned_stft)
        
        return cleaned_audio
    
    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        오디오 신호를 정규화합니다.
        
        Args:
            audio: 입력 오디오 신호
            
        Returns:
            정규화된 오디오 신호
        """
        # RMS 기반 정규화
        rms = np.sqrt(np.mean(audio**2))
        if rms > 0:
            target_rms = 0.1  # 목표 RMS 값
            audio = audio * (target_rms / rms)
        
        # 클리핑 방지
        audio = np.clip(audio, -1.0, 1.0)
        
        return audio
    
    def adjust_length(self, audio: np.ndarray) -> np.ndarray:
        """
        오디오 길이를 목표 길이로 조정합니다.
        
        Args:
            audio: 입력 오디오 신호
            
        Returns:
            길이 조정된 오디오 신호
        """
        current_length = len(audio)
        
        if current_length > self.target_length:
            # 중앙 부분 추출
            start_idx = (current_length - self.target_length) // 2
            audio = audio[start_idx:start_idx + self.target_length]
        elif current_length < self.target_length:
            # 제로 패딩
            padding = self.target_length - current_length
            pad_left = padding // 2
            pad_right = padding - pad_left
            audio = np.pad(audio, (pad_left, pad_right), mode='constant')
        
        return audio
    
    def apply_bandpass_filter(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        수박 타격음에 적합한 대역통과 필터를 적용합니다.
        
        Args:
            audio: 입력 오디오 신호
            sr: 샘플링 레이트
            
        Returns:
            필터링된 오디오 신호
        """
        # 수박 타격음의 주요 주파수 대역: 100Hz ~ 4000Hz
        lowcut = 100.0
        highcut = 4000.0
        
        # 버터워스 필터 설계
        nyquist = sr / 2
        low = lowcut / nyquist
        high = highcut / nyquist
        
        b, a = signal.butter(4, [low, high], btype='band')
        filtered_audio = signal.filtfilt(b, a, audio)
        
        return filtered_audio
    
    def preprocess_audio(self, file_path: Path, save_path: Optional[Path] = None) -> Optional[np.ndarray]:
        """
        오디오 파일을 전처리합니다.
        
        Args:
            file_path: 입력 오디오 파일 경로
            save_path: 전처리된 오디오 저장 경로 (선택사항)
            
        Returns:
            전처리된 오디오 신호 (실패시 None)
        """
        # 1. 오디오 로드
        audio, sr = self.load_audio(file_path)
        if audio is None:
            return None
        
        print(f"📁 처리 중: {file_path.name} (길이: {len(audio)/sr:.2f}초, SR: {sr}Hz)")
        
        # 2. 대역통과 필터 적용
        audio = self.apply_bandpass_filter(audio, sr)
        
        # 3. 노이즈 제거
        if self.apply_noise_reduction:
            audio = self.remove_noise(audio, sr)
        
        # 4. 정규화
        if self.normalize:
            audio = self.normalize_audio(audio)
        
        # 5. 길이 조정
        audio = self.adjust_length(audio)
        
        # 6. 저장 (선택사항)
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(save_path), audio, sr)
            print(f"💾 저장됨: {save_path}")
        
        print(f"✅ 전처리 완료: 최종 길이 {len(audio)/sr:.2f}초")
        return audio
    
    def process_dataset(self, data_root: Path, output_root: Path):
        """
        전체 데이터셋을 전처리합니다.
        
        Args:
            data_root: 원본 데이터 루트 경로
            output_root: 전처리된 데이터 저장 경로
        """
        print("🔄 데이터셋 전처리 시작...")
        output_root.mkdir(parents=True, exist_ok=True)
        
        # 모든 수박 폴더 찾기
        watermelon_folders = sorted([f for f in data_root.iterdir() 
                                   if f.is_dir() and '_' in f.name])
        
        processed_count = 0
        failed_count = 0
        
        for folder in watermelon_folders:
            folder_name = folder.name
            print(f"\n📂 폴더 처리 중: {folder_name}")
            
            # 출력 폴더 생성
            output_folder = output_root / folder_name
            output_folder.mkdir(exist_ok=True)
            
            # audio 폴더의 모든 파일 처리
            audio_folder = folder / 'audio'
            if audio_folder.exists():
                audio_files = [f for f in audio_folder.iterdir() if f.is_file()]
                
                for audio_file in audio_files:
                    # 출력 파일명 (모두 wav로 통일)
                    output_file = output_folder / f"{audio_file.stem}_processed.wav"
                    
                    # 전처리 수행
                    processed_audio = self.preprocess_audio(audio_file, output_file)
                    
                    if processed_audio is not None:
                        processed_count += 1
                    else:
                        failed_count += 1
        
        print(f"\n🎉 데이터셋 전처리 완료!")
        print(f"✅ 성공: {processed_count}개")
        print(f"❌ 실패: {failed_count}개")
        print(f"📁 저장 위치: {output_root}")

def main():
    """메인 실행 함수"""
    print("🎵 수박 당도 판별 프로젝트 - 오디오 전처리")
    print("="*60)
    
    # 전처리기 초기화
    preprocessor = AudioPreprocessor(
        target_sr=22050,        # 22.05kHz
        target_duration=3.0,    # 3초
        normalize=True,
        apply_noise_reduction=True
    )
    
    # 경로 설정
    data_root = Path("watermelon_data")
    output_root = Path("preprocessed_data")
    
    # 데이터셋 전처리
    if data_root.exists():
        preprocessor.process_dataset(data_root, output_root)
    else:
        print(f"❌ 데이터 폴더를 찾을 수 없습니다: {data_root}")
    
    # 샘플 전처리 테스트
    print("\n📋 샘플 전처리 테스트...")
    sample_folder = data_root / "01_10.5" / "audio"
    if sample_folder.exists():
        sample_files = list(sample_folder.glob("*"))[:2]  # 첫 2개 파일만
        
        for sample_file in sample_files:
            print(f"\n🔧 테스트: {sample_file.name}")
            processed_audio = preprocessor.preprocess_audio(sample_file)
            if processed_audio is not None:
                print(f"   결과: 길이 {len(processed_audio)} 샘플, RMS {np.sqrt(np.mean(processed_audio**2)):.4f}")

if __name__ == "__main__":
    main() 