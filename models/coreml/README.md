# 🍉 수박 당도 예측 Core ML 모델

## 📝 개요

이 Core ML 모델은 수박 타격음에서 추출한 멜-스펙트로그램을 분석하여 수박의 당도(Brix)를 예측합니다.

## 📊 모델 정보

- **모델 이름**: WatermelonBrixPredictor
- **파일 크기**: 4.8 MB
- **입력**: 224×224 RGB 이미지 (멜-스펙트로그램)
- **출력**: Float 값 (당도, Brix 단위)
- **성능**: RMSE 0.75, 당도 정확도(±1.0 Brix) 85.2%

## 🚀 Swift 프로젝트에 적용하기

### 1. 모델 파일 추가

1. `WatermelonBrixPredictor.mlmodel` 파일을 Xcode 프로젝트에 드래그 앤 드롭
2. "Copy items if needed" 체크
3. Target에 추가

### 2. 기본 사용법

```swift
import CoreML

// 모델 초기화
let predictor = WatermelonBrixPredictor()

// 이미지로 예측
if let brix = predictor.predict(image: melspectrogramImage) {
    print("예측된 당도: \(brix) Brix")
}
```

### 3. 완전한 구현 예제

전체 구현 예제는 `SwiftUsageExample.swift` 파일을 참고하세요.

## 📱 주요 기능

### WatermelonBrixPredictor 클래스

- `predict(image: UIImage) -> Float?`: UIImage 입력으로 당도 예측
- `predict(pixelBuffer: CVPixelBuffer) -> Float?`: CVPixelBuffer 입력으로 당도 예측

### 당도 품질 등급

- **12+ Brix**: 🟢 매우 달콤 (최고급)
- **10-12 Brix**: 🟡 달콤 (고급)
- **8-10 Brix**: 🟠 보통 (중급)
- **8 미만**: 🔴 부족 (저급)

## 📋 요구사항

### iOS 프로젝트

- **iOS 13.0+**
- **Xcode 11.0+**
- **Core ML Framework**

### 필요한 Frameworks

```swift
import CoreML
import UIKit
import Vision      // (선택사항: 이미지 전처리용)
import AVFoundation // (선택사항: 카메라 입력용)
```

## 🔧 이미지 전처리

모델은 다음과 같은 전처리가 된 이미지를 필요로 합니다:

1. **크기**: 224×224 픽셀
2. **형식**: RGB 3채널
3. **정규화**: ImageNet 표준 (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])

```swift
// 이미지 전처리 예제
let processedImage = originalImage
    .resized(to: CGSize(width: 224, height: 224))
    .normalized()
```

## 📊 성능 최적화

### 추론 성능

- **평균 추론 시간**: ~20ms (iPhone 12 기준)
- **메모리 사용량**: ~10MB
- **배터리 효율**: iOS Core ML 최적화로 저전력

### 권장사항

1. **백그라운드 스레드**에서 추론 실행
2. **이미지 캐싱**으로 중복 처리 방지
3. **배치 처리**로 여러 이미지 동시 처리

```swift
DispatchQueue.global(qos: .userInitiated).async {
    let brix = predictor.predict(image: image)

    DispatchQueue.main.async {
        // UI 업데이트
        updateResult(brix: brix)
    }
}
```

## 🐛 트러블슈팅

### 일반적인 문제들

#### 1. 모델 로드 실패

```
❌ 모델 파일을 찾을 수 없습니다.
```

**해결책**: 모델 파일이 Bundle에 제대로 추가되었는지 확인

#### 2. 예측 실패

```
❌ 이미지를 CVPixelBuffer로 변환 실패
```

**해결책**: 이미지 형식과 크기 확인 (224×224 RGB)

#### 3. 성능 문제

**해결책**:

- 메인 스레드가 아닌 백그라운드에서 추론 실행
- 이미지 크기 최적화
- 메모리 관리 개선

### 디버깅 팁

```swift
// 모델 정보 확인
if let model = try? MLModel(contentsOf: modelURL) {
    let description = model.modelDescription
    print("입력: \(description.inputDescriptionsByName)")
    print("출력: \(description.outputDescriptionsByName)")
}

// 입력 이미지 검증
print("이미지 크기: \(image.size)")
print("픽셀 버퍼 형식: \(CVPixelBufferGetPixelFormatType(pixelBuffer))")
```

## 📈 모델 업데이트

새로운 버전의 모델이 릴리스되면:

1. 새 `.mlmodel` 파일로 교체
2. 버전 호환성 확인
3. 앱 테스트 실행
4. App Store 업데이트

## 🔗 관련 링크

- [Apple Core ML 문서](https://developer.apple.com/documentation/coreml)
- [Core ML Tools](https://coremltools.readme.io/)
- [Vision Framework](https://developer.apple.com/documentation/vision)

## 📧 지원

문제가 발생하거나 질문이 있으시면:

- 이슈 트래커에 버그 리포트 작성
- 예제 코드와 에러 메시지 포함
- iOS 버전과 기기 정보 명시

---

**WatermelonAI Team** | MIT License | 2024
