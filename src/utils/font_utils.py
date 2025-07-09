"""
한글 폰트 설정 유틸리티

이 모듈은 matplotlib에서 한글을 올바르게 표시하기 위한 폰트 설정을 제공합니다.
플랫폼별로 적절한 폰트를 자동으로 선택하고 설정합니다.

Author: AI Assistant  
Date: 2024
"""

import platform
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import warnings


def setup_korean_font():
    """
    한글 폰트를 설정합니다.
    플랫폼에 따라 적절한 폰트를 자동으로 선택합니다.
    """
    system = platform.system()
    
    try:
        # 사용 가능한 폰트 리스트 확인
        available_fonts = [f.name for f in fm.fontManager.ttflist]
        
        # 플랫폼별 우선순위 폰트 리스트
        if system == "Darwin":  # macOS
            font_candidates = [
                'AppleGothic', 'Apple Gothic', 'AppleSDGothicNeo-Regular',
                'Helvetica', 'Arial Unicode MS', 'DejaVu Sans'
            ]
        elif system == "Windows":  # Windows
            font_candidates = [
                'Malgun Gothic', '맑은 고딕', 'Gulim', '굴림',
                'Dotum', '돋움', 'Batang', '바탕', 'Arial Unicode MS'
            ]
        else:  # Linux
            font_candidates = [
                'Noto Sans CJK KR', 'Noto Sans Korean', 'DejaVu Sans',
                'Liberation Sans', 'Droid Sans Fallback', 'Arial Unicode MS'
            ]
        
        # 사용 가능한 첫 번째 폰트 선택
        selected_font = None
        for font in font_candidates:
            if font in available_fonts:
                selected_font = font
                break
        
        # 폰트 설정 적용
        if selected_font:
            plt.rcParams['font.family'] = selected_font
            # print(f"✅ 한글 폰트 설정 완료: {selected_font}")
        else:
            # 기본 설정
            plt.rcParams['font.family'] = 'DejaVu Sans'
            warnings.warn("⚠️ 한글 폰트를 찾을 수 없어 기본 폰트를 사용합니다. 한글이 깨질 수 있습니다.")
        
        # 마이너스 기호 깨짐 방지
        plt.rcParams['axes.unicode_minus'] = False
        
        # 폰트 크기 기본값 설정
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['axes.labelsize'] = 10
        plt.rcParams['xtick.labelsize'] = 9
        plt.rcParams['ytick.labelsize'] = 9
        plt.rcParams['legend.fontsize'] = 9
        plt.rcParams['figure.titlesize'] = 14
        
    except Exception as e:
        warnings.warn(f"⚠️ 폰트 설정 중 오류 발생: {e}")
        # 최소한의 설정
        plt.rcParams['axes.unicode_minus'] = False


def get_korean_font_name():
    """현재 설정된 한글 폰트 이름을 반환합니다."""
    return plt.rcParams['font.family']


def list_available_korean_fonts():
    """시스템에서 사용 가능한 한글 폰트 목록을 반환합니다."""
    korean_fonts = []
    
    # 한글 폰트 키워드
    korean_keywords = [
        'Gothic', 'Gulim', 'Dotum', 'Batang', 'Malgun', 'Apple',
        'Noto', 'CJK', 'Korean', '고딕', '굴림', '돋움', '바탕', '맑은'
    ]
    
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    for font in available_fonts:
        for keyword in korean_keywords:
            if keyword.lower() in font.lower():
                korean_fonts.append(font)
                break
    
    return sorted(list(set(korean_fonts)))


def test_korean_display():
    """한글 표시 테스트를 수행합니다."""
    import matplotlib.pyplot as plt
    import numpy as np
    
    # 테스트 데이터
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    
    # 테스트 플롯
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, label='사인 곡선')
    plt.title('한글 폰트 테스트 - 수박 당도 예측')
    plt.xlabel('시간 (초)')
    plt.ylabel('진폭')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 한글 텍스트 추가
    plt.text(5, 0.5, '한글이 정상적으로 표시되나요?', 
             ha='center', va='center', 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    
    plt.tight_layout()
    plt.show()
    
    print(f"현재 사용 중인 폰트: {get_korean_font_name()}")


if __name__ == "__main__":
    # 폰트 설정 및 테스트
    print("🔧 한글 폰트 설정 중...")
    setup_korean_font()
    
    print("\n📋 사용 가능한 한글 폰트:")
    for font in list_available_korean_fonts():
        print(f"  - {font}")
    
    print("\n🧪 한글 표시 테스트 실행...")
    test_korean_display() 