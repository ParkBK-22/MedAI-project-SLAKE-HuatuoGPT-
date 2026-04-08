"""
HuatuoGPT-Vision-7B 모델 검증 스크립트
"""

import torch
import sys
import os
import yaml
import warnings

# 프로젝트 루트 경로 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# HuggingFace 토큰 설정
hf_token = os.getenv('HF_TOKEN')
if hf_token:
    os.environ['HF_TOKEN'] = hf_token
    try:
        from huggingface_hub import login
        login(token=hf_token, add_to_git_credential=False)
    except:
        pass

# 경고 무시
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


def main():
    print("=" * 80)
    print("🚀 HuatuoGPT-Vision-7B MODEL VERIFICATION")
    print("=" * 80)
    
    # GPU 확인
    print(f"\n✓ CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name()}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    try:
        # 설정 로드
        with open('configs/slake_config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        print(f"✓ Config loaded from configs/slake_config.yaml")
        
        # 모델 초기화
        print(f"\n📥 Loading model: {config['model']['name']}...")
        from src.model_huatuo import HuatuoInference
        
        model = HuatuoInference(
            config, 
            device='cuda' if torch.cuda.is_available() else 'cpu',
            use_official_cli=False
        )
        
        print("✅ Model loaded successfully!")
        print("\n" + "=" * 80)
        print("✨ VERIFICATION COMPLETE - Ready for experiments!")
        print("=" * 80)
        
        return 0
        
    except Exception as e:
        print(f"\n❌ VERIFICATION FAILED: {e}")
        print("\n💡 Troubleshooting:")
        print("  1. pip install --upgrade transformers")
        print("  2. pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
        print("  3. Check configs/slake_config.yaml exists")
        return 1


if __name__ == "__main__":
    exit(main())