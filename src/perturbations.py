import numpy as np
import cv2
from PIL import Image
import warnings

# NumPy 호환성 경고 무시
warnings.filterwarnings('ignore', category=UserWarning)

class ImagePerturber:
    def __init__(self, lpf_sigma=3, hpf_sigma=25, patch_size=16):
        self.lpf_sigma = lpf_sigma
        self.hpf_sigma = hpf_sigma
        self.patch_size = patch_size

    def apply(self, image_pil, condition):
        # PIL 이미지를 numpy(OpenCV 포맷)로 변환
        img = np.array(image_pil.convert('RGB'))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        if condition == "black":
            res = np.zeros_like(img)
        elif condition == "lpf":
            # Gaussian Low-Pass Filter: 고주파(디테일) 제거
            res = cv2.GaussianBlur(img, (0, 0), self.lpf_sigma)
        elif condition == "hpf":
            # High-Pass Filter: 저주파(구조) 제거하고 엣지만 보존
            low_freq = cv2.GaussianBlur(img, (0, 0), self.hpf_sigma)
            res = cv2.addWeighted(img, 1.0, low_freq, -1.0, 128)
        elif condition == "patch_shuffle":
            res = self._shuffle(img)
        else: # original
            res = img

        # 다시 PIL 이미지로 복원하여 반환
        res_rgb = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
        return Image.fromarray(res_rgb)

    def _shuffle(self, img):
        h, w, c = img.shape
        p = self.patch_size
        
        # 이미지 크기를 패치 사이즈에 맞게 크롭 (에러 방지)
        h_new, w_new = (h // p) * p, (w // p) * p
        img = img[:h_new, :w_new]

        # 패치 단위로 쪼개기
        patches = []
        for i in range(0, h_new, p):
            for j in range(0, w_new, p):
                patches.append(img[i:i+p, j:j+p].copy())
        
        # 무작위 섞기
        np.random.shuffle(patches)

        # 재조립
        res = np.zeros_like(img)
        idx = 0
        for i in range(0, h_new, p):
            for j in range(0, w_new, p):
                res[i:i+p, j:j+p] = patches[idx]
                idx += 1
        return res