"""
HuatuoGPT-Vision 모델 래퍼
공식 CLI와 transformers 기반 두 가지 방식 지원
"""

import torch
from transformers import AutoProcessor, AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import warnings

# NumPy 호환성 경고 무시
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings("ignore")

class HuatuoInference:
    def __init__(self, config, device="cuda", use_official_cli=False):
        """
        HuatuoGPT-Vision-7B 모델 초기화
        
        Args:
            config: 설정 객체
            device: 'cuda' or 'cpu'
            use_official_cli: True면 공식 HuatuoChatbot 사용, False면 transformers 직접 사용
        """
        self.device = device
        self.model_name = config['model']['name']
        self.prompt_template = config['model'].get('prompt_template', '')
        self.max_new_tokens = config['model'].get('max_new_tokens', 20)
        self.temperature = config['model'].get('temperature', 0.0)
        self.use_official_cli = use_official_cli
        
        print(f"Loading HuatuoGPT-Vision model: {self.model_name}")
        
        if use_official_cli:
            self._init_official_cli()
        else:
            self._init_transformers()
    
    def _init_official_cli(self):
        """공식 HuatuoChatbot 초기화"""
        try:
            from cli import HuatuoChatbot
            self.bot = HuatuoChatbot(self.model_name)
            self.model = None
            self.processor = None
            self.tokenizer = None
            print("✓ HuatuoChatbot (Official) loaded successfully")
        except ImportError:
            print("⚠ Warning: cli.py not found. Falling back to transformers...")
            self._init_transformers()
        except Exception as e:
            print(f"⚠ Warning: Could not load HuatuoChatbot ({e})")
            self._init_transformers()
    
    def _init_transformers(self):
        """Transformers 기반 초기화 (LLaVA 방식)"""
        try:
            self.bot = None
            
            # Processor 로드 (이미지 전처리 담당)
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # 모델 로드
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map=self.device,
                trust_remote_code=True
            )
            self.model.eval()
            
            # 토크나이저 (fallback 용)
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                use_fast=False
            )
            
            print("✓ Model loaded successfully (Transformers + LLaVA)")
            
        except Exception as e:
            print(f"⚠ Warning: Could not load model ({e})")
            print("  Falling back to mock inference for testing")
            self.model = None
            self.processor = None
            self.tokenizer = None

    def generate_answer(self, image, question):
        """
        이미지와 질문으로부터 답변 생성
        
        Args:
            image: PIL Image 객체 또는 numpy array
            question: 질문 문자열
            
        Returns:
            생성된 답변 문자열
        """
        # 이미지 정규화
        if isinstance(image, Image.Image):
            image = image.convert('RGB')
        else:
            import numpy as np
            image = Image.fromarray(image).convert('RGB')
        
        # 공식 CLI 사용
        if self.bot is not None:
            return self._inference_official(image, question)
        
        # Transformers 사용
        elif self.model is not None and self.processor is not None:
            return self._inference_transformers(image, question)
        
        # 모델 로드 실패 - Mock 모드
        else:
            return "test_answer"
    
    def _inference_official(self, image, question):
        """공식 HuatuoChatbot API로 추론"""
        try:
            import tempfile
            import os
            
            # 임시 파일에 이미지 저장 (CLI는 파일 경로를 요구함)
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                image.save(tmp.name)
                temp_path = tmp.name
            
            try:
                # 공식 API 호출
                output = self.bot.inference(question, [temp_path])
                return output.strip() if output else "no_answer"
            finally:
                # 임시 파일 정리
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    
        except Exception as e:
            print(f"Error in official inference: {e}")
            return "error"
    
    def _inference_transformers(self, image, question):
        """Transformers + LLaVA로 추론"""
        try:
            # 프롬프트 구성
            if self.prompt_template:
                prompt = self.prompt_template.format(question=question)
            else:
                # 기본 프롬프트 (LLaVA 포맷)
                prompt = f"Question: {question}\nAnswer:"
            
            # Processor: 이미지 + 텍스트 전처리
            inputs = self.processor(
                text=prompt,
                images=image,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            # 모델 추론
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    do_sample=False,
                    top_p=0.9,
                    eos_token_id=self.processor.tokenizer.eos_token_id
                )
            
            # 텍스트 디코딩
            response = self.processor.decode(
                outputs[0],
                skip_special_tokens=True
            )
            
            # 프롬프트 부분 제거하고 답변만 추출
            if "Answer:" in response:
                response = response.split("Answer:")[-1].strip()
            
            return response.strip() if response else "no_answer"
            
        except Exception as e:
            print(f"Error during transformers inference: {e}")
            return "error"
    