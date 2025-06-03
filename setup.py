#!/usr/bin/env python3
"""
Setup script for Consumer Grade EEG project
"""

import os
import sys
import subprocess

def check_python_version():
    """Check if Python version is 3.11 or higher"""
    if sys.version_info < (3, 11):
        print("❌ Error: Python 3.11 or higher is required")
        print(f"   Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version}")
    return True

def install_requirements():
    """Install required packages"""
    print("\n📦 Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All packages installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e}")
        return False

def create_data_directory():
    """Create Data directory if it doesn't exist"""
    data_dir = "Data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"✅ Created {data_dir} directory")
    else:
        print(f"✅ {data_dir} directory already exists")

def check_gpu_support():
    """Check for GPU support"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️  CUDA not available - will use CPU training")
    except ImportError:
        print("⚠️  PyTorch not installed yet")

def main():
    """Main setup function"""
    print("🚀 Consumer Grade EEG - Setup Script")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install requirements
    if not install_requirements():
        sys.exit(1)
    
    # Create data directory
    create_data_directory()
    
    # Check GPU support
    check_gpu_support()
    
    print("\n" + "=" * 50)
    print("✅ Setup completed successfully!")
    print("\n📝 Next steps:")
    print("1. Place your MindBigData file in the Data/ directory")
    print("2. Run a quick test: python main_script.py --quick-test")
    print("3. Run full analysis: python main_script.py")
    print("\n📚 See README.md for detailed usage instructions")

if __name__ == "__main__":
    main()
