# 🚀 SLAKE HuatuoGPT 실험 - 완전 실행 가이드

## 📋 목차
1. [환경 준비](#1-환경-준비)
2. [프로젝트 설정](#2-프로젝트-설정)
3. [데이터셋 준비](#3-데이터셋-준비)
4. [모델 검증](#4-모델-검증)
5. [실험 실행](#5-실험-실행)
6. [결과 분석](#6-결과-분석)

---

## 1. 환경 준비

### 1.1 Git 클론 (처음 설정할 때만)
```bash
# 프로젝트 클론
git clone https://github.com/YourUserName/MedAI-project-SLAKE-HuatuoGPT.git
cd MedAI-project-SLAKE-HuatuoGPT
```

### 1.2 Python 가상환경 생성
```bash
# venv 생성 (Python 3.10+)
python -m venv venv

# 가상환경 활성화
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 1.3 기본 의존성 설치
```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements_base.txt
```

### 1.4 PyTorch + CUDA 설치
```bash
# CUDA 12.1 버전 (RTX 4090 권장)
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121
```

### 1.5 HuatuoGPT 의존성 설치
```bash
# 최신 transformers (llava_qwen2 지원)
pip install git+https://github.com/huggingface/transformers.git --upgrade

# 나머지 패키지
pip install -r requirements_huatuogpt.txt
```

### 1.5 HuggingFace 토큰 설정 (필수!)

```bash
# 방법 1: 환경변수로 설정 (권장)
# Linux/Mac
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxx"

# Windows PowerShell
$env:HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxx"

# 방법 2: HuggingFace CLI로 로그인
huggingface-cli login
# → https://huggingface.co/settings/tokens 에서 토큰 생성 후 입력

# 방법 3: Python에서 설정
python -c "from huggingface_hub import login; login(token='hf_xxxxxxxxxxxxxxxxxxxx')"
```

**토큰 생성:**
1. [HuggingFace 토큰 페이지](https://huggingface.co/settings/tokens) 방문
2. "New token" 클릭
3. 권한: `read` 선택
4. 토큰 복사 (시작: `hf_`)
5. 터미널에 `HF_TOKEN` 설정

---

## 2. 프로젝트 설정

**HuggingFace 토큰이 설정되어 있는지 확인:**

```bash
# 환경변수 확인
# Linux/Mac
echo $HF_TOKEN

# Windows PowerShell
$env:HF_TOKEN
```

**토큰이 없으면 설정:**
```bash
export HF_TOKEN="hf_your_token_here"
```

**프로젝트 설정 실행:**

```bash
# 디렉토리 생성 및 초기 설정
python scripts/setup_slake.py
```

**출력 확인 사항:**
```
✓ PyTorch: 2.4.0+cu121
✓ CUDA Available: True
✓ GPU: NVIDIA GeForce RTX 4090 (25.3 GB)
✓ Dataset structure is complete!
```

---

## 3. 데이터셋 준비

### 3.1 SLAKE 데이터셋 다운로드

[SLAKE 공식 페이지](https://www.med-ai.com/slake)에서 다운로드 또는:

```bash
# 만약 다운로드 스크립트가 있다면
bash data/download_data.sh
```

### 3.2 데이터 경로 확인

```bash
# Windows
dir data\slake\
# 또는 Linux/Mac
ls -la data/slake/
```

**필수 파일 구조:**
```
data/slake/
├── train.json           (2.8 MB)
├── test.json            (620 KB)
├── validation.json      (624 KB)
└── imgs/                (643개 이미지)
    ├── xmlab0/
    ├── xmlab1/
    └── ...
```

### 3.3 데이터 구성 확인

```bash
# Python으로 JSON 샘플 확인
python << 'EOF'
import json
with open('data/slake/train.json', 'r') as f:
    data = json.load(f)
print(f"Train samples: {len(data)}")
print(f"First sample keys: {list(data[0].keys())}")
print(f"Example: {data[0]}")
EOF
```

---

## 4. 모델 검증

```bash
# 모델 로드 테스트
python scripts/verify_models.py
```

**성공 메시지:**
```
✓ CUDA Available: True
  GPU: NVIDIA GeForce RTX 4090
  VRAM: 25.3 GB
✓ Config loaded from configs/slake_config.yaml
✓ Model loaded successfully!
✨ VERIFICATION COMPLETE - Ready for experiments!
```

**만약 에러가 발생하면:**
```bash
# transformers 최신 버전으로 업그레이드
pip install --upgrade transformers
```

---

## 5. 실험 실행

### 5.1 전체 실험 (모든 이미지 조건 적용)

```bash
# 약 2-4시간 소요 (GPU 기준)
python scripts/run_slake_exp.py
```

**진행 순서:**
1. 모델 로드 (5-10분)
2. 데이터셋 로딩
3. 5가지 이미지 조건 적용:
   - Original (원본)
   - Black (검은색)
   - LPF (저주파 필터)
   - HPF (고주파 필터)
   - Patch Shuffle (패치 섞임)
4. 진단 지표 계산 (VRS, L-Drop, K-Ratio)

### 5.2 결과 파일 생성

```
✓ results/slake_results.csv                    # 전체 결과 (조건별, 질문별)
✓ results/diagnostics.yaml                     # 진단 지표
✓ results/question_type_analysis.yaml          # 질문 유형별 분석
```

### 5.3 진행 상황 모니터링

```bash
# 실시간 CSV 확인 (Linux/Mac)
tail -f results/slake_results.csv

# Windows PowerShell
Get-Content results/slake_results.csv -Tail 10 -Wait
```

---

## 6. 결과 분석

### 6.1 상세 분석 실행

```bash
# 전체 결과 분석 + 시각화 정보 출력
python scripts/analyze_slake.py
```

**출력 포함:**
- 📊 전체 정확도 (조건별)
- 📈 감도 지표 (VRS, L-Drop, K-Ratio)
- 🎯 질문 유형별 성능
- 🔬 모달리티별 분석 (CT, MRI, X-ray 등)
- 🧠 시각적 근거 인사이트

### 6.2 결과 내보내기 (선택사항)

```bash
# 상세 결과 내보내기 (Excel, JSON 형식)
python scripts/analyze_slake.py --export

# 또는
python scripts/analyze_slake.py --format csv,json,html
```

### 6.3 개별 결과 파일 확인

```bash
# CSV 결과 확인 (상위 10줄)
# Windows
powershell "Get-Content results/slake_results.csv | Select-Object -First 10"
# Linux/Mac
head -10 results/slake_results.csv

# YAML 진단 지표 확인
# Windows
type results/diagnostics.yaml
# Linux/Mac
cat results/diagnostics.yaml

# 질문 유형별 분석
# Windows
type results/question_type_analysis.yaml
# Linux/Mac
cat results/question_type_analysis.yaml
```

### 6.4 시각화 (옵션: 추가 분석)

```bash
# 이미지 perturbation 시각화
python scripts/visualize_perturbations.py

# 결과 저장 위치: results/viz/perturbation_check.png
```

---

## 📊 전체 실행 시간 추정

| 단계 | 소요 시간 | 비고 |
|------|---------|------|
| 환경 설정 | 10-15분 | pip 설치 포함 |
| 프로젝트 설정 | 1분 | `setup_slake.py` |
| 모델 검증 | 5-10분 | `verify_models.py` |
| **실험 실행** | **2-4시간** | **GPU 기반** |
| 결과 분석 | 2-5분 | `analyze_slake.py` |
| **총합** | **~3-5시간** | **One-time** |

---

## 🔧 트러블슈팅

### 문제: `ModuleNotFoundError: No module named 'src'`
```bash
# 해결: 프로젝트 루트에서 실행
cd /path/to/MedAI-project-SLAKE-HuatuoGPT
python scripts/run_slake_exp.py
```

### 문제: `FileNotFoundError: data/slake/train.json`
```bash
# 해결: 데이터셋 확인
ls data/slake/
# 필수 파일이 없으면 SLAKE 페이지에서 다운로드
```

### 문제: `CUDA out of memory`
```bash
# 해결: 메모리 부족 시 배치 크기 조정
# configs/slake_config.yaml 수정:
# batch_size: 1 (기본값)
```

### 문제: Transformers 아키텍처 오류
```bash
# 해결: 소스에서 최신 버전 설치
pip install git+https://github.com/huggingface/transformers.git --upgrade
```

---

## 📝 설정 파일 확인

```bash
# 실험 설정 확인
# Windows
type configs/slake_config.yaml
# Linux/Mac
cat configs/slake_config.yaml
```

---

## ✅ 체크리스트

- [ ] Python 3.10+ 설치됨
- [ ] 가상환경 활성화됨
- [ ] PyTorch + CUDA 설치됨
- [ ] SLAKE 데이터셋 다운로드 완료
- [ ] `setup_slake.py` 실행 완료
- [ ] `verify_models.py` 성공
- [ ] `run_slake_exp.py` 실행
- [ ] `analyze_slake.py` 결과 확인

---

## 🎯 다음 단계

1. **결과 해석:**
   - VRS > 0.3이면 시각 정보 중요도 높음
   - L-Drop > 0.1이면 LPF 영향 큼
   - Question Type별 성능 차이 분석

2. **추가 분석:**
   - Modality별 성능 비교 (CT vs MRI vs X-ray)
   - 질문 타입별 감도 분석 (Location > Shape > Organ)

3. **논문 작성:**
   - 결과를 `results/` 폴더의 CSV/YAML로 그래프 생성
   - 진단 지표를 표로 정리

---

## 📞 빠른 명령어 모음

```bash
# 한 줄로 전체 실행 (순차 실행)
python scripts/setup_slake.py && python scripts/verify_models.py && python scripts/run_slake_exp.py && python scripts/analyze_slake.py

# 또는 각 단계별 실행
python scripts/setup_slake.py
python scripts/verify_models.py
python scripts/run_slake_exp.py
python scripts/analyze_slake.py
```

---

**마지막 업데이트:** 2026-04-08
**상태:** ✅ 완전 자동화됨
