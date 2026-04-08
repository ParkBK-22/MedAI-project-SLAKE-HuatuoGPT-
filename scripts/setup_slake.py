import os
import subprocess
import sys
import warnings
import re
import zipfile
import shutil
from tqdm import tqdm
from pathlib import Path

# NumPy 호환성 경고 무시
warnings.filterwarnings('ignore', category=UserWarning, module='torch')
warnings.filterwarnings("ignore")


def run_command(command, shell_type="powershell", show_progress=False):
    """
    터미널 명령어를 실행합니다.
    
    Args:
        command: 실행할 명령어
        shell_type: 쉘 타입 (기본값: powershell)
        show_progress: True면 진행 게이지 표시 (다운로드용)
    """
    if not show_progress:
        print(f"  Executing: {command}")
    
    try:
        if show_progress:
            # 진행률 표시와 함께 실행
            process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            file_count = 0
            pbar = None
            
            for line in process.stdout:
                # 폴더 경로 감지 (예: ./imgs/xmlab99:)
                if re.match(r'\./imgs/\w+:', line.strip()):
                    file_count += 1
                    
                    # 첫 번째 파일일 때 progress bar 초기화
                    if pbar is None:
                        pbar = tqdm(
                            total=1300,  # 예상 총 파일 수
                            desc="⏳ Downloading",
                            unit="file",
                            ncols=80,
                            leave=True
                        )
                    
                    pbar.update(1)
            
            process.wait()
            
            if pbar:
                pbar.close()
            
            if process.returncode == 0:
                print(f"  ✓ Download completed! ({file_count} files)")
            else:
                raise subprocess.CalledProcessError(process.returncode, command)
        else:
            subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n  ❌ Error: {command}")
        print(e)
        return False
    return True


def extract_zip(zip_path, extract_to):
    """
    ZIP 파일을 추출합니다.
    
    Args:
        zip_path: ZIP 파일 경로
        extract_to: 추출할 디렉토리
    """
    if not os.path.exists(zip_path):
        print(f"  ⚠ ZIP not found: {zip_path}")
        return False
    
    try:
        print(f"  📦 Extracting: {os.path.basename(zip_path)}")
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            total_files = len(zip_ref.namelist())
            
            with tqdm(total=total_files, desc="📂 Extracting", unit="file", ncols=80) as pbar:
                for file_info in zip_ref.filelist:
                    zip_ref.extract(file_info, extract_to)
                    pbar.update(1)
        
        print(f"  ✓ Extracted {total_files} files to: {extract_to}")
        return True
        
    except Exception as e:
        print(f"  ❌ Extraction failed: {e}")
        return False


def cleanup_zip(zip_path):
    """
    ZIP 파일을 삭제하고 용량 정보를 표시합니다.
    
    Args:
        zip_path: 삭제할 ZIP 파일 경로
    """
    if os.path.exists(zip_path):
        try:
            size_mb = os.path.getsize(zip_path) / (1024 * 1024)
            os.remove(zip_path)
            print(f"  ✓ Cleaned up ZIP ({size_mb:.1f} MB freed): {os.path.basename(zip_path)}")
            return True
        except Exception as e:
            print(f"  ⚠ Could not delete ZIP: {e}")
            return False
    return False


def verify_dataset(data_dir):
    """
    데이터셋 구조를 검증합니다.
    
    Args:
        data_dir: 데이터 디렉토리 경로
    """
    print("\n  Verifying dataset structure...")
    
    required_items = {
        'imgs': 'Directory with images',
        'train.json': 'Training data JSON',
        'test.json': 'Test data JSON',
        'validation.json': 'Validation data JSON'
    }
    
    all_present = True
    for item, desc in required_items.items():
        path = os.path.join(data_dir, item)
        if os.path.exists(path):
            if os.path.isdir(path):
                num_items = len(os.listdir(path))
                print(f"    ✓ {item:20} ({desc}) [{num_items} items]")
            else:
                size_kb = os.path.getsize(path) / 1024
                print(f"    ✓ {item:20} ({desc}) [{size_kb:.1f} KB]")
        else:
            print(f"    ✗ {item:20} (MISSING - {desc})")
            all_present = False
    
    return all_present


def main():
    # 1. 필수 폴더 생성
    print("\n[1/7] Setting up HuggingFace token...")
    import os
    hf_token = os.getenv('HF_TOKEN')
    if not hf_token:
        print("  ⚠️  HF_TOKEN not set!")
        print("  Set it with:")
        print("    # Linux/Mac:")
        print("    export HF_TOKEN='your_token_here'")
        print("    # Windows PowerShell:")
        print("    $env:HF_TOKEN='your_token_here'")
        print("  Or login with: huggingface-cli login")
        response = input("\n  Continue without HF_TOKEN? (y/n): ").lower()
        if response != 'y':
            print("  Exiting...")
            return
    else:
        print(f"  ✓ HF_TOKEN is set")
    
    print("\n[2/7] Creating directories...")
    folders = [
        'data/slake',
        'results',
        'checkpoints',
        'results/viz'
    ]
    for folder in folders:
        Path(folder).mkdir(parents=True, exist_ok=True)
        print(f"  ✓ {folder}")

    # 2. 환경 검증
    print("\n[3/7] Verifying PyTorch & CUDA...")
    try:
        import torch
        print(f"  ✓ PyTorch: {torch.__version__}")
        print(f"  ✓ CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            device = torch.cuda.get_device_name(0)
            memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  ✓ GPU: {device} ({memory:.1f} GB)")
    except ImportError:
        print("  ⚠ PyTorch not installed. Run: pip install -r requirements_huatuogpt.txt")

    # 3. 패키지 확인
    print("\n[4/7] Checking required packages...")
    required_packages = ['PIL', 'cv2', 'numpy', 'pandas', 'yaml', 'transformers']
    missing = []
    
    for pkg in required_packages:
        try:
            __import__(pkg)
            print(f"  ✓ {pkg}")
        except ImportError:
            print(f"  ✗ {pkg} (missing)")
            missing.append(pkg)
    
    if missing:
        print(f"\n  ⚠ Install missing packages: pip install -r requirements_huatuogpt.txt")

    # 4. 데이터셋 처리
    print("\n[5/7] Dataset preparation...")
    
    data_dir = "data/slake"
    
    # ZIP 파일 찾기 및 추출
    zip_files = list(Path(data_dir).glob("*.zip"))
    
    if zip_files:
        print(f"  Found {len(zip_files)} ZIP file(s)")
        for zip_file in zip_files:
            print(f"\n  📦 Processing: {zip_file.name}")
            
            # 추출
            if extract_zip(str(zip_file), data_dir):
                # 정리
                cleanup_zip(str(zip_file))
            else:
                print(f"  ⚠ Skipped cleanup for {zip_file.name}")
    else:
        print("  • No ZIP files found in data/slake")

    # 5. 데이터셋 검증
    print("\n[6/7] Verifying dataset...")
    if verify_dataset(data_dir):
        print("  ✓ Dataset structure is complete!")
    else:
        print("  ⚠ Some files are missing. Please download the SLAKE dataset manually.")

    # 6. .gitkeep 생성
    print("\n[7/7] Creating .gitkeep files...")
    for folder in ['results', 'checkpoints']:
        gitkeep = os.path.join(folder, '.gitkeep')
        os.makedirs(os.path.dirname(gitkeep), exist_ok=True)
        Path(gitkeep).touch()
        print(f"  ✓ {gitkeep}")

    # 최종 요약
    print("\n" + "=" * 70)
    print("✅ SETUP COMPLETE!")
    print("=" * 70)
    print("\n📊 Dataset Summary:")
    print(f"  Location: {os.path.abspath(data_dir)}")
    print(f"  Size: {sum(os.path.getsize(os.path.join(dirpath, filename)) for dirpath, dirnames, filenames in os.walk(data_dir) for filename in filenames) / (1024**3):.2f} GB")
    
    print("\n🚀 Next steps:")
    print("  1. Verify config: configs/slake_config.yaml")
    print("  2. Run experiment: python scripts/run_slake_exp.py")
    print("  3. Analyze results: python scripts/analyze_slake.py")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()