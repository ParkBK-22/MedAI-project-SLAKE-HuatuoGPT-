# 🚀 HuatuoGPT-Vision 실험 구현 가이드

최신 수정사항을 바탕으로 공식 API 방식으로 업데이트되었습니다.

---

## 📋 **변경 사항 요약**

### ✅ **model_huatuo.py** - 완전 재구현
- **공식 HuatuoChatbot 방식** 우선 지원
- **Transformers + LLaVA 방식** fallback 제공
- 두 가지 방식 자동 선택 (`use_official_cli` 파라미터)

```python
# 방식 1: 공식 CLI 사용 (권장)
model = HuatuoInference(cfg, use_official_cli=True)

# 방식 2: Transformers 직접 사용
model = HuatuoInference(cfg, use_official_cli=False)
```

### ✅ **dataset_slake.py** - 견고성 강화
- JSON 필드명 **자동 감지**
- 첫 번째 샘플 구조 출력
- 누락된 필드 처리 및 예외 처리

### ✅ **run_slake_exp.py** - 모델 호출 업데이트
- HuatuoInference 생성자 파라미터 수정
- 모델 초기화 로직 개선

---

## 🔧 **HuatuoGPT 두 가지 사용 방식**

### **방식 1: 공식 HuatuoChatbot (가장 권장)**
```python
from cli import HuatuoChatbot

# 모델 초기화
bot = HuatuoChatbot('/path/to/HuatuoGPT-Vision-7B')

# 추론 (파일 경로 필요)
output = bot.inference('What is this?', ['image.jpg'])
```

**장점:**
- 공식 지원 방식
- 모든 기능 포함
- 가장 안정적

**요구사항:**
- `cli.py` 파일 필요 (공식 저장소에서 제공)

---

### **방식 2: Transformers + LLaVA (fallback)**
```python
from transformers import AutoProcessor, AutoModelForCausalLM

# Processor: 이미지 전처리 담당
processor = AutoProcessor.from_pretrained('FreedomIntelligence/HuatuoGPT-Vision-7B')

# 모델: 추론 담당
model = AutoModelForCausalLM.from_pretrained('FreedomIntelligence/HuatuoGPT-Vision-7B')

# 이미지 + 텍스트 함께 입력
inputs = processor(text="What is this?", images=image, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=20)
```

**장점:**
- 공식 저장소 없이 사용 가능
- Transformers 생태계와 호환

---

## 📊 **SLAKE 데이터셋 JSON 필드 확인**

### **자동 감지되는 필드:**

| 필드 | 감지 우선순위 |
|------|----------------|
| **이미지** | `img_name` → `image_name` → `file_name` → `image` |
| **질문** | `question` → `q` |
| **답변** | `answer` → `a` → `gt` |
| **타입** | `q_type` → `qtype` → `type` |
| **모달리티** | `modality` → `image_type` |
| **ID** | `img_id` → `id` |

### **확인 방법:**

**1단계:** 데이터셋 다운로드 후 첫 샘플 확인
```python
import json
data = json.load(open('data/slake/questions.json'))
print(data[0])  # 첫 번째 샘플 구조 확인
print(data[0].keys())  # JSON 필드명 출력
```

**2단계:** 실험 실행 시 자동 출력
```bash
python scripts/run_slake_exp.py
# 출력: Sample JSON structure: ['img_id', 'image_name', 'question', ...]
```

---

## 🎯 **Multimodal 입력 방식 쉬운 설명**

### **텍스트만 입력 (Unimodal)**
```
입력: "What is pneumonia?"
출력: "Pneumonia is a lung infection..."
```

### **이미지 + 텍스트 (Multimodal)** ← HuatuoGPT
```
입력:
  - 이미지: [CT 스캔]
  - 텍스트: "What is shown in this image?"

출력: "This is a CT scan showing pneumonia in the right lung."
```

### **내부 동작:**
```
1. 이미지 처리 (Vision Encoder)
   [CT 이미지] → CLIP Vision → [시각 특징 벡터]

2. 텍스트 처리 (Tokenizer)
   "What is..." → 토크나이저 → [텍스트 토큰]

3. 합치기 (Fusion)
   [시각 벡터] + [텍스트 토큰] → 결합 표현

4. 답변 생성 (LLM)
   [결합 표현] → LLaMA/Qwen → "This is a CT scan..."
```

---

## 📝 **사용 예시**

### **로컬 테스트 (모델 없이)**
```bash
# Mock 모드로 자동 작동
python scripts/run_slake_exp.py
# 출력: "test_answer" (플레이스홀더)
```

### **실제 실험**

**1단계:** 필수 파일 준비
```
data/slake/
  ├── questions.json      # SLAKE 메타데이터
  └── images/             # CT, MRI 이미지 폴더
       ├── image001.jpg
       ├── image002.jpg
       └── ...
```

**2단계:** 환경 설정
```bash
python scripts/setup_slake.py
pip install -r requirements_huatuogpt.txt
```

**3단계:** 실험 실행
```bash
# Transformers 방식 (기본설정)
python scripts/run_slake_exp.py

# 또는 공식 CLI를 사용하려면 config 수정 필요
```

**4단계:** 결과 분석
```bash
python scripts/analyze_slake.py --export
```

---

## 🔍 **문제 해결**

### **모델 로드 실패**
```
⚠ Warning: Could not load model (...)
→ Mock 모드로 자동 전환됨
```

**해결:**
1. GPU 메모리 확인 (7B 모델 = 최소 16GB)
2. `transformers==4.40.0` 버전 확인
3. 공식 저장소에서 `cli.py` 다운로드

### **이미지 로드 실패**
```
Warning: Image not found at /path/to/image.jpg
→ 회색 대체 이미지 사용
```

**해결:**
1. 이미지 경로 확인
2. JSON 파일의 이미지명 필드 확인
3. `dataset_slake.py` 실행 시 출력되는 필드명 확인

---

## 💡 **주요 설정값**

| 설정 | 값 | 설명 |
|------|-----|------|
| `max_new_tokens` | 20 | 단답형 제한 (무한 생성 방지) |
| `temperature` | 0.0 | Greedy 디코딩 (결정적) |
| `patch_size` | 16 | Patch Shuffle 크기 |
| `lpf_sigma` | 3 | 저주파 필터 강도 |
| `hpf_sigma` | 25 | 고주파 필터 강도 |

---

## 📚 **학습 자료**

- **HuatuoGPT GitHub:** https://github.com/FreedomIntelligence/HuatuoGPT-Vision
- **LLaVA 아키텍처:** https://arxiv.org/abs/2304.08485
- **SLAKE 데이터셋:** https://github.com/vqdang/SLAKE_Dataset

---

## ✅ **체크리스트**

- [ ] SLAKE 데이터셋 다운로드
- [ ] `requirements_huatuogpt.txt` 설치
- [ ] `setup_slake.py` 실행
- [ ] 데이터셋 JSON 필드명 확인
- [ ] `run_slake_exp.py` 테스트 실행
- [ ] `analyze_slake.py`로 결과 분석
- [ ] 커밋 및 배포

---

**최종 수정:** 2026년 4월 8일  
**상태:** ✅ 공식 API 방식 완전 지원
