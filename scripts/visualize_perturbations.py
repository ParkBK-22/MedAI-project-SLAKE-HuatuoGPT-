import os
import sys
import matplotlib.pyplot as plt
from PIL import Image

# 프로젝트 루트 경로 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.perturbations import ImagePerturber

def main():
    # 1. 테스트 이미지 로드 (샘플 하나 준비하세요)
    img_path = "data/slake/images/xmlab0/scan_0.jpg" # 예시 경로
    if not os.path.exists(img_path):
        print("테스트 이미지가 없습니다. 경로를 확인하세요.")
        return

    img = Image.open(img_path).convert('RGB')
    perturber = ImagePerturber(lpf_sigma=3, hpf_sigma=25, patch_size=16)
    conditions = ["original", "black", "lpf", "hpf", "patch_shuffle"]

    # 2. 시각화 설정
    plt.figure(figsize=(20, 4))
    for i, cond in enumerate(conditions):
        p_img = perturber.apply(img, cond)
        plt.subplot(1, 5, i+1)
        plt.imshow(p_img)
        plt.title(f"Condition: {cond.upper()}", fontsize=12)
        plt.axis('off')

    # 3. 결과 저장
    os.makedirs("results/viz", exist_ok=True)
    plt.savefig("results/viz/perturbation_check.png")
    print("시각화 결과가 results/viz/perturbation_check.png에 저장되었습니다.")

if __name__ == "__main__":
    main()