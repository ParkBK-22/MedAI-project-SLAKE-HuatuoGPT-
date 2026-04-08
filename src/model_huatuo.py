import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import numpy as np
import warnings
warnings.filterwarnings("ignore")

class HuatuoInference:
    def __init__(self, config, device="cuda"):
        """
        HuatuoGPT-Vision-7B 모델 및 토크나이저 초기화
        
        Args:
            config: 설정 시전
            device: 'cuda' or 'cpu'
        """
        self.device = device
        self.model_name = config['model']['name']
        self.prompt_template = config['model']['prompt_template']
        self.max_new_tokens = config['model'].get('max_new_tokens', 20)
        self.temperature = config['model'].get('temperature', 0.0)
        
        print(f"Loading model: {self.model_name}")
        try:
            # HuatuoGPT-Vision 모델 로드
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                use_fast=False
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map=self.device,
                trust_remote_code=True
            )
            self.model.eval()
            print("✓ Model loaded successfully")
        except Exception as e:
            print(f"⚠ Warning: Could not load model ({e})")
            print("  Falling back to mock inference for testing")
            self.model = None
            self.tokenizer = None

    def generate_answer(self, image, question):
        """
        이미지와 질문으로부터 답변 생성
        
        Args:
            image: PIL Image 객체
            question: 질문 문자열
            
        Returns:
            생성된 답변 문자열
        """
        if self.model is None or self.tokenizer is None:
            # Mock 모드 (테스트용)
            return "test_answer"
        
        try:
            # 프롬프트 구성
            full_prompt = self.prompt_template.format(question=question)
            
            # PIL 이미지를 모델 입력 형식으로 변환
            if isinstance(image, Image.Image):
                image = image.convert('RGB')
            else:
                image = Image.fromarray(image)
            
            # 토크나이저로 프롬프트 처리
            # HuatuoGPT는 <image> 토큰을 인식하므로 그대로 전달
            inputs = self.tokenizer(
                full_prompt,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.device)
            
            # 모델 추론 (max_new_tokens=20으로 제한하여 짧은 답변 생성)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    do_sample=False,
                    top_p=0.9,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # 토크 디코딩
            response = self.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            )
            
            # 프롬프트 부분 제거하고 답변만 추출
            # 예: "Question: ... Answer: yes"에서 "yes"만 추출
            if "Answer:" in response or "<|assistant|>" in response:
                response = response.split("Answer:")[-1].strip()
                response = response.split("<|assistant|>")[-1].strip()
            
            return response.strip()
            
        except Exception as e:
            print(f"Error during inference: {e}")
            return "error"
    