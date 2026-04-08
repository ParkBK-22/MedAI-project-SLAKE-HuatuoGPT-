import os
import subprocess
import sys
import warnings

# NumPy 호환성 경고 무시 (PyTorch 로드 전에 설정)
warnings.filterwarnings('ignore', category=UserWarning, module='torch')
warnings.filterwarnings("ignore")

def run_command(command, shell_type="powershell", suppress_output=False):
    """
    터미널 명령어를 실행합니다.
    
    Args:
        command: 실행할 명령어
        shell_type: 쉘 타입 (기본값: powershell)
        suppress_output: True면 출력을 최소화 (다운로드 등 장시간 작업용)
    """
    if not suppress_output:
        print(f"Executing: {command}")
    
    try:
        if suppress_output:
            # 출력 억제 (진행 중일 때 진행 메시지만 표시)
            subprocess.run(
                command,
                shell=True,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            print("  ✓ Download completed!")
        else:
            subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while executing: {command}")
        print(e)
        return False
    return True

def main():
    print("=" * 60)
    print("SLAKE Project Setup")
    print("=" * 60)

    # 1. 필수 폴더 생성
    print("\n[1/5] Creating directories...")
    folders = [
        'data',
        'data/slake',
        'results',
        'checkpoints',
        'results/viz'
    ]
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"  ✓ Created: {folder}")
        else:
            print(f"  • Already exists: {folder}")

    # 2. 환경 검증 (PyTorch 및 CUDA)
    print("\n[2/5] Verifying PyTorch & CUDA...")
    try:
        import torch
        print(f"  ✓ PyTorch Version: {torch.__version__}")
        print(f"  ✓ CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  ✓ GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("  ⚠ Warning: PyTorch is not installed.")
        print("     Run: pip install -r requirements_base.txt")

    # 3. 필수 Python 패키지 확인
    print("\n[3/5] Checking required packages...")
    required_packages = [
        'PIL',
        'cv2',
        'numpy',
        'pandas',
        'torch',
        'transformers'
    ]
    
    missing_packages = []
    for pkg in required_packages:
        try:
            __import__(pkg)
            print(f"  ✓ {pkg}")
        except ImportError:
            print(f"  ✗ {pkg} (missing)")
            missing_packages.append(pkg)
    
    if missing_packages:
        print(f"\n  ⚠ Missing packages: {', '.join(missing_packages)}")
        print(f"     Run: pip install -r requirements_huatuogpt.txt")

    # 4. 데이터 다운로드 스크립트 실행 (선택적)
    print("\n[4/5] Dataset preparation...")
    if os.path.exists('data/download_data.sh'):
        response = input("  Found data/download_data.sh. Download data now? (y/n): ").lower()
        if response == 'y':
            print("  Running data download script (verbose output suppressed)...")
            run_command("bash data/download_data.sh", suppress_output=True)
        else:
            print("  Skipped data download. You can run it manually later.")
    else:
        print("  ⚠ data/download_data.sh not found.")
        print("     Please ensure SLAKE dataset is in: data/slake/")
        print("     - data/slake/images/")
        print("     - data/slake/questions.json")

    # 5. .gitkeep 생성 (빈 폴더 유지를 위해)
    print("\n[5/5] Creating .gitkeep files...")
    gitkeep_paths = [
        'results/.gitkeep',
        'checkpoints/.gitkeep'
    ]
    for path in gitkeep_paths:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            pass
        print(f"  ✓ Created: {path}")

    print("\n" + "=" * 60)
    print("✓ Setup Complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Prepare SLAKE dataset files:")
    print("   - Place images in: data/slake/images/")
    print("   - Place metadata in: data/slake/questions.json")
    print("\n2. Install dependencies:")
    print("   - pip install -r requirements_base.txt")
    print("   - pip install -r requirements_huatuogpt.txt")
    print("\n3. Run the experiment:")
    print("   - python scripts/run_slake_exp.py")
    print("=" * 60)

if __name__ == "__main__":
    main()