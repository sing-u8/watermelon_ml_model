#!/usr/bin/env python3
"""
수박 당도 판별 프로젝트 - 데이터셋 구조 분석
Dataset Structure Analysis for Watermelon Brix Detection Project
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import librosa
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

class WatermelonDatasetAnalyzer:
    def __init__(self, data_root="watermelon_data"):
        self.data_root = Path(data_root)
        self.dataset_info = []
        
    def analyze_dataset_structure(self):
        """데이터셋 구조를 분석하고 통계 정보를 수집합니다."""
        print("🔍 데이터셋 구조 분석 시작...")
        
        # 모든 수박 폴더 찾기
        watermelon_folders = sorted([f for f in self.data_root.iterdir() 
                                   if f.is_dir() and '_' in f.name])
        
        print(f"📁 총 수박 샘플 폴더 수: {len(watermelon_folders)}")
        
        for folder in watermelon_folders:
            sample_info = self._analyze_single_sample(folder)
            self.dataset_info.append(sample_info)
        
        # DataFrame으로 변환
        self.df = pd.DataFrame(self.dataset_info)
        return self.df
    
    def _analyze_single_sample(self, folder_path):
        """개별 수박 샘플의 정보를 분석합니다."""
        folder_name = folder_path.name
        
        # 폴더명에서 샘플 번호와 Brix 추출
        try:
            sample_num, brix_str = folder_name.split('_')
            brix_value = float(brix_str)
        except ValueError:
            print(f"⚠️  폴더명 파싱 오류: {folder_name}")
            return None
        
        sample_info = {
            'sample_num': int(sample_num),
            'brix': brix_value,
            'folder_name': folder_name,
            'folder_path': str(folder_path)
        }
        
        # 각 하위 폴더의 파일 수 확인
        for subfolder in ['audio', 'audios', 'chu', 'picture']:
            subfolder_path = folder_path / subfolder
            if subfolder_path.exists():
                if subfolder == 'chu':
                    # chu 폴더는 하위 폴더 수를 세기
                    chu_dirs = [d for d in subfolder_path.iterdir() if d.is_dir()]
                    sample_info[f'{subfolder}_count'] = len(chu_dirs)
                else:
                    # 다른 폴더는 파일 수를 세기
                    files = list(subfolder_path.glob('*'))
                    sample_info[f'{subfolder}_count'] = len([f for f in files if f.is_file()])
                    
                    # 오디오 파일의 확장자 확인
                    if subfolder in ['audio', 'audios']:
                        extensions = [f.suffix.lower() for f in files if f.is_file()]
                        sample_info[f'{subfolder}_extensions'] = list(set(extensions))
            else:
                sample_info[f'{subfolder}_count'] = 0
                if subfolder in ['audio', 'audios']:
                    sample_info[f'{subfolder}_extensions'] = []
        
        return sample_info
    
    def print_summary_statistics(self):
        """데이터셋 요약 통계를 출력합니다."""
        print("\n" + "="*60)
        print("📊 데이터셋 요약 통계")
        print("="*60)
        
        print(f"✅ 총 수박 샘플 수: {len(self.df)}")
        print(f"📈 Brix 범위: {self.df['brix'].min():.1f} ~ {self.df['brix'].max():.1f}")
        print(f"📊 Brix 평균: {self.df['brix'].mean():.2f} ± {self.df['brix'].std():.2f}")
        
        print(f"\n🎵 오디오 파일 통계:")
        print(f"   - audio 폴더 파일 수 (평균): {self.df['audio_count'].mean():.1f}")
        print(f"   - audios 폴더 파일 수 (평균): {self.df['audios_count'].mean():.1f}")
        print(f"   - chu 폴더 하위 디렉토리 수 (평균): {self.df['chu_count'].mean():.1f}")
        
        # 오디오 파일 확장자 통계
        all_audio_ext = []
        all_audios_ext = []
        
        for _, row in self.df.iterrows():
            all_audio_ext.extend(row['audio_extensions'])
            all_audios_ext.extend(row['audios_extensions'])
        
        print(f"\n📁 파일 형식:")
        print(f"   - audio 폴더 확장자: {set(all_audio_ext)}")
        print(f"   - audios 폴더 확장자: {set(all_audios_ext)}")
        
    def visualize_brix_distribution(self):
        """Brix 분포를 시각화합니다."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('🍉 수박 당도(Brix) 분포 분석', fontsize=16, fontweight='bold')
        
        # 히스토그램
        axes[0,0].hist(self.df['brix'], bins=10, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[0,0].set_title('Brix 히스토그램')
        axes[0,0].set_xlabel('Brix 값')
        axes[0,0].set_ylabel('빈도')
        axes[0,0].grid(True, alpha=0.3)
        
        # 박스플롯
        axes[0,1].boxplot(self.df['brix'])
        axes[0,1].set_title('Brix 박스플롯')
        axes[0,1].set_ylabel('Brix 값')
        axes[0,1].grid(True, alpha=0.3)
        
        # 샘플별 Brix 값
        axes[1,0].scatter(self.df['sample_num'], self.df['brix'], alpha=0.7, color='red')
        axes[1,0].set_title('샘플별 Brix 값')
        axes[1,0].set_xlabel('샘플 번호')
        axes[1,0].set_ylabel('Brix 값')
        axes[1,0].grid(True, alpha=0.3)
        
        # 통계 정보 텍스트
        stats_text = f"""
        총 샘플 수: {len(self.df)}
        Brix 범위: {self.df['brix'].min():.1f} ~ {self.df['brix'].max():.1f}
        평균: {self.df['brix'].mean():.2f}
        표준편차: {self.df['brix'].std():.2f}
        중간값: {self.df['brix'].median():.2f}
        """
        axes[1,1].text(0.1, 0.5, stats_text, transform=axes[1,1].transAxes, 
                       fontsize=12, verticalalignment='center',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        axes[1,1].set_title('통계 요약')
        axes[1,1].axis('off')
        
        plt.tight_layout()
        plt.savefig('brix_distribution_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("📊 Brix 분포 그래프가 'brix_distribution_analysis.png'로 저장되었습니다.")
    
    def analyze_audio_files(self, sample_limit=3):
        """오디오 파일의 기본 정보를 분석합니다."""
        print(f"\n🎵 오디오 파일 분석 (샘플 {sample_limit}개)...")
        
        audio_analysis = []
        
        for i, (_, row) in enumerate(self.df.head(sample_limit).iterrows()):
            folder_path = Path(row['folder_path'])
            
            # audio 폴더의 첫 번째 파일 분석
            audio_folder = folder_path / 'audio'
            if audio_folder.exists():
                audio_files = list(audio_folder.glob('*'))
                if audio_files:
                    first_audio = audio_files[0]
                    try:
                        # librosa로 오디오 정보 확인
                        y, sr = librosa.load(first_audio, sr=None)
                        duration = len(y) / sr
                        
                        file_info = {
                            'sample': row['folder_name'],
                            'brix': row['brix'],
                            'file_name': first_audio.name,
                            'extension': first_audio.suffix,
                            'duration_sec': duration,
                            'sample_rate': sr,
                            'samples': len(y)
                        }
                        audio_analysis.append(file_info)
                        
                        print(f"   📁 {row['folder_name']}: {first_audio.name}")
                        print(f"      - 길이: {duration:.2f}초, 샘플레이트: {sr}Hz, 샘플 수: {len(y)}")
                        
                    except Exception as e:
                        print(f"   ❌ {first_audio.name} 분석 실패: {e}")
        
        return pd.DataFrame(audio_analysis)

def main():
    """메인 실행 함수"""
    print("🍉 수박 당도 판별 프로젝트 - 데이터셋 분석")
    print("="*60)
    
    # 분석기 초기화
    analyzer = WatermelonDatasetAnalyzer()
    
    # 데이터셋 구조 분석
    dataset_df = analyzer.analyze_dataset_structure()
    
    # 요약 통계 출력
    analyzer.print_summary_statistics()
    
    # Brix 분포 시각화
    analyzer.visualize_brix_distribution()
    
    # 오디오 파일 분석
    audio_df = analyzer.analyze_audio_files()
    
    # 결과 저장
    dataset_df.to_csv('dataset_analysis.csv', index=False, encoding='utf-8-sig')
    print(f"\n💾 데이터셋 분석 결과가 'dataset_analysis.csv'로 저장되었습니다.")
    
    return analyzer, dataset_df, audio_df

if __name__ == "__main__":
    analyzer, dataset_df, audio_df = main() 