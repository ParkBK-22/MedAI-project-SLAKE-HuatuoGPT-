"""
HuatuoGPT-Vision 모델 래퍼 (최신 Transformers 4.42+ 호환 버전)
"""

import torch
import os
from transformers import AutoProcessor, AutoModelForCausalLM
from huggingface_hub import login
from PIL import Image
import warnings

# NumPy 호환성 경고 무시
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings("ignore")

# HuggingFace 토큰 자동 설정
def setup_hf_token():
    """HuggingFace 토큰 설정 (환경변수 또는 캐시에서)"""
    hf_token = os.getenv('HF_TOKEN')
    if hf_token:
        try:
            login(token=hf_token, add_to_git_credential=False)
            print("✓ HuggingFace token authenticated")
        except Exception as e:
            print(f"⚠ HF token setup failed: {e}")
    else:
        print("⚠ HF_TOKEN not set. Model download may be rate-limited.")
        print("  Set: export HF_TOKEN='your_token' or huggingface-cli login")

setup_hf_token()

class HuatuoInference:
    def __init__(self, config, device="cuda", use_official_cli=False):
        """
        HuatuoGPT-Vision-7B 모델 초기화
        """
        self.device = device
        self.model_name = config['model']['name']
        # HuatuoGPT 공식 프롬프트 템플릿 적용
        self.prompt_template = "<|user|>\n<image>\n{question}</s>\n<|assistant|>\n"
        self.max_new_tokens = config['model'].get('max_new_tokens', 128)
        self.temperature = config['model'].get('temperature', 0.0)
        self.use_official_cli = use_official_cli
        
        print(f"🚀 Loading HuatuoGPT-Vision model: {self.model_name}")
        
        if use_official_cli:
            self._init_official_cli()
        else:
            self._init_transformers()
    
    def _init_official_cli(self):
        """공식 HuatuoChatbot 초기화 (유지)"""
        try:
            from cli import HuatuoChatbot
            self.bot = HuatuoChatbot(self.model_name)
            print("✓ HuatuoChatbot (Official) loaded successfully")
        except Exception as e:
            print(f"⚠ Warning: Could not load HuatuoChatbot ({e}). Falling back to transformers...")
            self._init_transformers()
    
    def _init_transformers(self):
        """Transformers 기반 초기화 (llava_qwen2 아키텍처 최적화)"""
        try:
            self.bot = None
            
            # 1. Processor 로드 (이미지 + 텍스트 통합 처리)
            print("  📥 Loading processor...")
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                cache_dir=None  # 캐시 비활성화로 최신 다운로드
            )
            
            # 2. VLM 전용 모델 로드 (CPU에서 먼저 로드 후 GPU로 이동)
            print("  📥 Loading model on CPU first...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,  # CPU에서는 float32 사용
                device_map="cpu",  # CPU에서 먼저 로드
                trust_remote_code=True,
                cache_dir=None
            )
            
            # GPU로 이동
            print("  🚀 Moving model to GPU...")
            if self.device == "cuda":
                self.model = self.model.to(torch.float16).cuda()
            
            self.model.eval()
            
            print("✓ Model loaded successfully (CPU → GPU conversion)")
            
        except Exception as e:
            print(f"❌ Error: Could not load model ({type(e).__name__}: {str(e)[:100]})")
            print("\n💡 Alternative: Using mock inference for testing...")
            self.model = None
            self.processor = None

    def generate_answer(self, image, question):
        """이미지와 질문으로부터 답변 생성"""
        try:
            # 이미지 전처리 (RGB 변환)
            if isinstance(image, Image.Image):
                image = image.convert('RGB')
            else:
                import numpy as np
                image = Image.fromarray(image).convert('RGB')
            
            # 공식 CLI 사용 시
            if self.bot is not None:
                return self._inference_official(image, question)
            
            # 최신 Transformers 사용 시
            elif self.model is not None and self.processor is not None:
                return self._inference_transformers(image, question)
            
            # 모델 미로드 시: 기본 응답
            return "unknown"
        
        except Exception as e:
            return "error"
    
    def _inference_transformers(self, image, question):
        """최신 LLaVA-Qwen2 파이프라인으로 추론"""
        try:
            # 1. HuatuoGPT 전용 프롬프트 구성
            prompt = self.prompt_template.format(question=question)
            
            # 2. Processor를 통해 입력값 준비
            inputs = self.processor(
                text=prompt, 
                images=image, 
                return_tensors="pt"
            ).to("cuda", torch.float16)
            
            # 3. 답변 생성
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=self.temperature > 0,
                    temperature=self.temperature if self.temperature > 0 else 1.0,
                    use_cache=True
                )
            
            # 4. 결과 디코딩 (입력 프롬프트 제외하고 정답만 추출)
            input_token_len = inputs.input_ids.shape[1]
            response = self.processor.decode(
                output_ids[0][input_token_len:], 
                skip_special_tokens=True
            ).strip()
            
            return response if response else "no_answer"
            
        except Exception as e:
            print(f"Error during inference: {e}")
            return "error"

    def _inference_official(self, image, question):
        """기존 공식 API 추론 로직 (유지)"""
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            image.save(tmp.name)
            temp_path = tmp.name
        try:
            output = self.bot.inference(question, [temp_path])
            return output.strip() if output else "no_answer"
        finally:
            if os.path.exists(temp_path): os.remove(temp_path)