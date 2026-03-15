#!/usr/bin/env python3
"""
Environment setup script for QLoRA Starter Kit.
Installs dependencies and verifies the environment.
"""

import os
import platform
import subprocess
import sys


def check_python_version():
    """Check if Python version is 3.10+."""
    print("=" * 50)
    print("Checking Python version...")
    version = sys.version_info
    print(f"Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print("ERROR: Python 3.10+ is required")
        return False
    
    print("✓ Python version OK")
    return True


def check_cuda():
    """Check CUDA availability."""
    print("\n" + "=" * 50)
    print("Checking CUDA...")
    
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        
        if cuda_available:
            print(f"✓ CUDA available: {torch.version.cuda}")
            print(f"  Device count: {torch.cuda.device_count()}")
            if torch.cuda.device_count() > 0:
                print(f"  Current device: {torch.cuda.get_device_name(0)}")
                print(f"  Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("⚠ CUDA not available - training will use CPU (very slow)")
            
        return True
    except ImportError:
        print("⚠ PyTorch not installed yet - run pip install first")
        return True


def check_system():
    """Check system information."""
    print("\n" + "=" * 50)
    print("System Information:")
    print(f"  OS: {platform.system()} {platform.release()}")
    print(f"  Architecture: {platform.machine()}")
    print(f"  Processor: {platform.processor()}")


def install_requirements():
    """Install requirements from requirements.txt."""
    print("\n" + "=" * 50)
    print("Installing dependencies...")
    
    req_file = os.path.join(os.path.dirname(__file__), "requirements.txt")
    
    if not os.path.exists(req_file):
        print(f"ERROR: {req_file} not found")
        return False
    
    cmd = [sys.executable, "-m", "pip", "install", "-r", req_file]
    
    try:
        subprocess.run(cmd, check=True)
        print("✓ Dependencies installed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Failed to install dependencies: {e}")
        return False


def install_dev_requirements():
    """Install development dependencies."""
    print("\n" + "=" * 50)
    print("Installing development dependencies...")
    
    cmd = [sys.executable, "-m", "pip", "install", "-e", ".[dev]"]
    
    try:
        subprocess.run(cmd, check=True)
        print("✓ Dev dependencies installed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Failed to install dev dependencies: {e}")
        return False


def verify_install():
    """Verify key packages are installed."""
    print("\n" + "=" * 50)
    print("Verifying installation...")
    
    packages = [
        "torch",
        "transformers",
        "peft",
        "bitsandbytes",
        "accelerate",
        "trl",
        "datasets",
    ]
    
    all_ok = True
    for pkg in packages:
        try:
            __import__(pkg)
            print(f"✓ {pkg}")
        except ImportError:
            print(f"✗ {pkg} - NOT INSTALLED")
            all_ok = False
    
    return all_ok


def check_gpu_memory():
    """Check available GPU memory."""
    print("\n" + "=" * 50)
    print("GPU Memory Check...")
    
    try:
        import torch
        if torch.cuda.is_available():
            mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"Total VRAM: {mem:.1f} GB")
            
            if mem >= 24:
                print("✓ Excellent for QLoRA (can run 70B models)")
            elif mem >= 16:
                print("✓ Good for QLoRA (can run 13B models comfortably)")
            elif mem >= 8:
                print("✓ Adequate for QLoRA (can run 7B models)")
            else:
                print("⚠ Limited VRAM - may need extra optimizations")
        else:
            print("⚠ No GPU detected")
    except Exception as e:
        print(f"⚠ Could not check GPU: {e}")


def create_directories():
    """Create necessary directories."""
    print("\n" + "=" * 50)
    print("Creating directories...")
    
    dirs = [
        "data/raw",
        "data/processed",
        "output",
        "models",
    ]
    
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"✓ {d}/")
    
    return True


def main():
    """Main setup function."""
    print("=" * 50)
    print("QLoRA Starter Kit - Environment Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check system
    check_system()
    
    # Install requirements (commented out for safety)
    # install_requirements()
    
    # Install dev requirements (commented out for safety)
    # install_dev_requirements()
    
    # Verify installation
    verify_install()
    
    # Check CUDA
    check_cuda()
    
    # Check GPU memory
    check_gpu_memory()
    
    # Create directories
    create_directories()
    
    print("\n" + "=" * 50)
    print("Setup complete!")
    print("=" * 50)
    print("\nNext steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Copy .env.example to .env and add your tokens")
    print("3. Run: python train.py --config configs/qlora_config.yaml")


if __name__ == "__main__":
    main()
