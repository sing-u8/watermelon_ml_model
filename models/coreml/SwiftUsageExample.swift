/*
 수박 당도 예측 Core ML 모델 Swift 사용 예제
 
 이 파일은 변환된 Core ML 모델을 Swift 프로젝트에서 사용하는 방법을 보여줍니다.
 
 Author: AI Assistant
 Date: 2024
 */

import Foundation
import CoreML
import UIKit
import Vision
import AVFoundation

// MARK: - Core ML 모델 래퍼 클래스
class WatermelonBrixPredictor {
    private var model: MLModel?
    
    init() {
        loadModel()
    }
    
    private func loadModel() {
        guard let modelURL = Bundle.main.url(forResource: "WatermelonBrixPredictor", withExtension: "mlmodel") else {
            print("❌ 모델 파일을 찾을 수 없습니다.")
            return
        }
        
        do {
            model = try MLModel(contentsOf: modelURL)
            print("✅ Core ML 모델 로드 성공")
        } catch {
            print("❌ 모델 로드 실패: \(error.localizedDescription)")
        }
    }
    
    // MARK: - 예측 함수 (UIImage 입력)
    func predict(image: UIImage) -> Float? {
        guard let model = model else {
            print("❌ 모델이 로드되지 않았습니다.")
            return nil
        }
        
        guard let pixelBuffer = image.toCVPixelBuffer(size: CGSize(width: 224, height: 224)) else {
            print("❌ 이미지를 CVPixelBuffer로 변환 실패")
            return nil
        }
        
        return predict(pixelBuffer: pixelBuffer)
    }
    
    // MARK: - 예측 함수 (CVPixelBuffer 입력)
    func predict(pixelBuffer: CVPixelBuffer) -> Float? {
        guard let model = model else {
            print("❌ 모델이 로드되지 않았습니다.")
            return nil
        }
        
        do {
            // Core ML 예측 실행
            let prediction = try model.prediction(from: MLDictionaryFeatureProvider(dictionary: [
                "melspectrogram_image": MLFeatureValue(pixelBuffer: pixelBuffer)
            ]))
            
            // 결과 추출
            if let brixOutput = prediction.featureValue(for: "brix_prediction") {
                let brixValue = Float(brixOutput.doubleValue)
                print("🍉 예측된 당도: \(brixValue) Brix")
                return brixValue
            }
            
        } catch {
            print("❌ 예측 실패: \(error.localizedDescription)")
        }
        
        return nil
    }
}

// MARK: - UIImage Extension (이미지 전처리)
extension UIImage {
    func toCVPixelBuffer(size: CGSize) -> CVPixelBuffer? {
        let attrs = [
            kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue,
            kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue
        ] as CFDictionary
        
        var pixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            Int(size.width),
            Int(size.height),
            kCVPixelFormatType_32ARGB,
            attrs,
            &pixelBuffer
        )
        
        guard status == kCVReturnSuccess, let buffer = pixelBuffer else {
            return nil
        }
        
        CVPixelBufferLockBaseAddress(buffer, CVPixelBufferLockFlags(rawValue: 0))
        defer { CVPixelBufferUnlockBaseAddress(buffer, CVPixelBufferLockFlags(rawValue: 0)) }
        
        let pixelData = CVPixelBufferGetBaseAddress(buffer)
        let rgbColorSpace = CGColorSpaceCreateDeviceRGB()
        
        guard let context = CGContext(
            data: pixelData,
            width: Int(size.width),
            height: Int(size.height),
            bitsPerComponent: 8,
            bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
            space: rgbColorSpace,
            bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue
        ) else {
            return nil
        }
        
        context.translateBy(x: 0, y: size.height)
        context.scaleBy(x: 1.0, y: -1.0)
        
        UIGraphicsPushContext(context)
        self.draw(in: CGRect(x: 0, y: 0, width: size.width, height: size.height))
        UIGraphicsPopContext()
        
        return buffer
    }
    
    // 이미지 정규화 (ImageNet 표준)
    func normalized() -> UIImage? {
        guard let cgImage = self.cgImage else { return nil }
        
        let width = cgImage.width
        let height = cgImage.height
        
        guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) else { return nil }
        guard let context = CGContext(
            data: nil,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * 4,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
        ) else { return nil }
        
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        
        guard let normalizedCGImage = context.makeImage() else { return nil }
        
        return UIImage(cgImage: normalizedCGImage)
    }
}

// MARK: - 사용 예제 View Controller
class WatermelonViewController: UIViewController {
    private let predictor = WatermelonBrixPredictor()
    @IBOutlet weak var imageView: UIImageView!
    @IBOutlet weak var resultLabel: UILabel!
    @IBOutlet weak var predictButton: UIButton!
    
    override func viewDidLoad() {
        super.viewDidLoad()
        setupUI()
    }
    
    private func setupUI() {
        title = "🍉 수박 당도 예측"
        predictButton.setTitle("당도 예측하기", for: .normal)
        resultLabel.text = "이미지를 선택하고 예측 버튼을 눌러주세요"
    }
    
    @IBAction func selectImageButtonTapped(_ sender: UIButton) {
        let imagePickerController = UIImagePickerController()
        imagePickerController.delegate = self
        imagePickerController.sourceType = .photoLibrary
        present(imagePickerController, animated: true)
    }
    
    @IBAction func predictButtonTapped(_ sender: UIButton) {
        guard let image = imageView.image else {
            resultLabel.text = "❌ 먼저 이미지를 선택해주세요"
            return
        }
        
        resultLabel.text = "🔄 예측 중..."
        
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            let brixValue = self?.predictor.predict(image: image)
            
            DispatchQueue.main.async {
                if let brix = brixValue {
                    self?.resultLabel.text = "🍉 예측된 당도: \(String(format: "%.2f", brix)) Brix"
                    
                    // 당도 범위에 따른 품질 평가
                    let quality = self?.getQualityRating(brix: brix) ?? "알 수 없음"
                    self?.resultLabel.text! += "\n품질: \(quality)"
                } else {
                    self?.resultLabel.text = "❌ 예측 실패"
                }
            }
        }
    }
    
    private func getQualityRating(brix: Float) -> String {
        switch brix {
        case 12...:
            return "🟢 매우 달콤 (최고급)"
        case 10..<12:
            return "🟡 달콤 (고급)"
        case 8..<10:
            return "🟠 보통 (중급)"
        default:
            return "🔴 부족 (저급)"
        }
    }
}

// MARK: - Image Picker Delegate
extension WatermelonViewController: UIImagePickerControllerDelegate, UINavigationControllerDelegate {
    func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
        if let selectedImage = info[.originalImage] as? UIImage {
            imageView.image = selectedImage
            resultLabel.text = "이미지가 선택되었습니다. 예측 버튼을 눌러주세요."
        }
        picker.dismiss(animated: true)
    }
    
    func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
        picker.dismiss(animated: true)
    }
}

// MARK: - 간단한 사용 예제
/*
 기본 사용법:
 
 let predictor = WatermelonBrixPredictor()
 
 // UIImage로 예측
 if let brix = predictor.predict(image: yourMelspectrogramImage) {
     print("예측된 당도: \(brix) Brix")
 }
 
 // CVPixelBuffer로 예측 (카메라 입력 등)
 if let brix = predictor.predict(pixelBuffer: pixelBuffer) {
     print("예측된 당도: \(brix) Brix")
 }
 */ 