import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image

def main():
    model_id = "FreedomIntelligence/HuatuoGPT-Vision-7B"
    print(f"🚀 Loading model: {model_id}...")
    
    try:
        # 1. 모델 및 프로세서 로드 (LLaVA-Qwen2 구조 최적화)
        # AutoProcessor는 이미지 전처리와 토큰화를 한 번에 처리합니다.
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            torch_dtype=torch.float16, # 4090에서는 float16이 안정적입니다.
            device_map="auto", 
            trust_remote_code=True
        )
        model.eval()
        print("✅ Model & Processor loaded successfully!")

        # 2. 실제 추론 테스트 (Sanity Check)
        print("🧪 Running a quick inference test...")
        
        # 가짜 입력 생성 (흰색 배경 이미지)
        dummy_img = Image.new('RGB', (224, 224), color='white')
        question = "What is shown in this image?"
        
        # HuatuoGPT 전용 프롬프트 템플릿 적용
        prompt = f"<|user|>\n<image>\n{question}</s>\n<|assistant|>\n"
        
        # 데이터 준비 (GPU 이동)
        inputs = processor(text=prompt, images=dummy_img, return_tensors="pt").to("cuda", torch.float16)

        # 답변 생성
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
                use_cache=True
            )

        # 결과 디코딩
        input_token_len = inputs.input_ids.shape[1]
        response = processor.decode(output_ids[0][input_token_len:], skip_special_tokens=True).strip()
        
        print(f"✨ Model Verification Success!")
        print(f"📝 Model Response: {response}")

    except Exception as e:
        print(f"❌ Model Verification Failed: {e}")
        print("\n💡 Tip: 'pip install --upgrade transformers'를 했는지 다시 확인해보세요.")

if __name__ == "__main__":
    main()