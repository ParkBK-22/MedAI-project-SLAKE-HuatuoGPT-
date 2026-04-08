import json
import os
from PIL import Image
import warnings

# NumPy 호환성 경고 무시
warnings.filterwarnings('ignore', category=UserWarning)

class SlakeDataset:
    def __init__(self, json_path, img_dir):
        """
        SLAKE 데이터셋 로더
        
        Args:
            json_path: questions.json 파일 경로
            img_dir: 이미지 디렉토리 경로
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        # 첫 번째 샘플 구조 확인
        if len(raw_data) > 0:
            first_sample = raw_data[0]
            print(f"Sample JSON structure: {list(first_sample.keys())}")
        
        # 영어 질문만 필터링 (필드명 자동 감지)
        self.data = []
        for d in raw_data:
            # q_lang, lang, 또는 language 필드 확인
            lang_field = d.get('q_lang') or d.get('lang') or d.get('language')
            if lang_field == 'en' or lang_field == 'English':
                self.data.append(d)
            # 필드가 없으면 모든 데이터 포함
            elif 'q_lang' not in d and 'lang' not in d and 'language' not in d:
                self.data.append(d)
        
        if len(self.data) == 0:
            print(f"Warning: No English data found. Using all {len(raw_data)} samples.")
            self.data = raw_data
        
        self.img_dir = img_dir
        print(f"Loaded {len(self.data)} samples from {json_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        데이터셋 샘플 반환
        
        Returns:
            dict: img_id, image, question, answer, q_type, modality
        """
        item = self.data[idx]
        
        # 필드명 자동 감지
        img_name = item.get('img_name') or item.get('image_name') or item.get('file_name') or item.get('image')
        question = item.get('question') or item.get('q')
        answer = item.get('answer') or item.get('a') or item.get('gt')
        q_type = item.get('q_type') or item.get('qtype') or item.get('type') or 'Unknown'
        modality = item.get('modality') or item.get('image_type') or 'Unknown'
        img_id = item.get('img_id') or item.get('id') or str(idx)
        
        # 이미지 로드
        img_path = os.path.join(self.img_dir, img_name)
        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            print(f"Warning: Image not found at {img_path}")
            # 회색 이미지로 대체
            image = Image.new('RGB', (224, 224), color=(128, 128, 128))
        except Exception as e:
            print(f"Warning: Could not load image {img_path}: {e}")
            image = Image.new('RGB', (224, 224), color=(128, 128, 128))
        
        return {
            "img_id": img_id,
            "image": image,
            "question": question,
            "answer": answer,
            "q_type": q_type,
            "modality": modality
        }