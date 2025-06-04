#!/usr/bin/env python3
"""
Example usage script for Consumer Grade EEG project
Demonstrates basic functionality and usage patterns
"""

import os
import sys
import numpy as np

def check_data_file():
    """Check if data file exists"""
    data_file = "Data/EP1.01.txt"
    if os.path.exists(data_file):
        print(f"✅ Data file found: {data_file}")
        return data_file
    else:
        print(f"❌ Data file not found: {data_file}")
        print("   Please place your MindBigData file in the Data/ directory")
        return None

def example_spatial_analysis():
    """Example of spatial analysis pipeline"""
    print("\n🧠 Example: Spatial Analysis Pipeline")
    print("-" * 40)
    
    try:
        from main_script import SpatialDigitClassifier
        
        # Initialize classifier
        classifier = SpatialDigitClassifier(device='EP', sampling_rate=128)
        
        # Check data file
        data_file = check_data_file()
        if data_file is None:
            return False
        
        # Quick test
        print("\n🧪 Running quick test...")
        success = classifier.quick_test(data_file)
        
        if success:
            print("✅ Quick test passed!")
            print("   You can now run the full analysis with:")
            print("   python main_script.py")
        else:
            print("❌ Quick test failed - check your data format")
        
        return success
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Make sure all dependencies are installed")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def example_deep_learning():
    """Example of deep learning pipeline"""
    print("\n🤖 Example: Deep Learning Pipeline")
    print("-" * 40)
    
    try:
        # Check if PyTorch is available
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        
        # Check if TensorFlow is available
        try:
            import tensorflow as tf
            print(f"✅ TensorFlow version: {tf.__version__}")
        except ImportError:
            print("⚠️  TensorFlow not available")
        
        print("\n📝 Available deep learning scripts:")
        scripts = [
            "eeg_pytorch.py - PyTorch EEGNet implementation",
            "eeg_deep_learning.py - TensorFlow/Keras implementation", 
            "eeg_lstm_wavelet.py - LSTM with wavelet features",
            "eeg_transformer.py - Transformer with wavelets",
            "eeg_pytorch_improved.py - Enhanced PyTorch model"
        ]
        
        for script in scripts:
            print(f"   • {script}")
        
        print("\n💡 To run a deep learning model:")
        print("   python eeg_pytorch.py")
        
        return True
        
    except ImportError as e:
        print(f"❌ Deep learning frameworks not available: {e}")
        print("   Install with: pip install torch tensorflow")
        return False

def example_wavelet_analysis():
    """Example of wavelet analysis"""
    print("\n🌊 Example: Wavelet Analysis")
    print("-" * 40)
    
    try:
        import pywt
        print(f"✅ PyWavelets version: {pywt.__version__}")
        
        # List available wavelets
        wavelets = pywt.wavelist(kind='discrete')[:10]  # First 10
        print(f"   Available wavelets (first 10): {wavelets}")
        
        print("\n📝 Wavelet analysis scripts:")
        print("   • advanced_wavelet_features.py - Comprehensive wavelet features")
        print("   • wavelet_visualization.py - Wavelet visualization")
        
        print("\n💡 To run wavelet analysis:")
        print("   python advanced_wavelet_features.py")
        
        return True
        
    except ImportError as e:
        print(f"❌ PyWavelets not available: {e}")
        print("   Install with: pip install PyWavelets")
        return False

def show_project_structure():
    """Show project structure"""
    print("\n📁 Project Structure")
    print("-" * 40)
    
    files = [
        "main_script.py - Main spatial classification pipeline",
        "eeg_*.py - Deep learning implementations",
        "advanced_wavelet_features.py - Wavelet feature extraction",
        "Data/ - Dataset directory (place your data here)",
        "requirements.txt - Python dependencies",
        "README.md - Comprehensive documentation"
    ]
    
    for file in files:
        print(f"   • {file}")

def main():
    """Main example function"""
    print("🚀 Consumer Grade EEG - Example Usage")
    print("=" * 50)
    
    # Show project structure
    show_project_structure()
    
    # Run examples
    spatial_ok = example_spatial_analysis()
    dl_ok = example_deep_learning()
    wavelet_ok = example_wavelet_analysis()
    
    print("\n" + "=" * 50)
    print("📊 Summary:")
    print(f"   Spatial Analysis: {'✅' if spatial_ok else '❌'}")
    print(f"   Deep Learning: {'✅' if dl_ok else '❌'}")
    print(f"   Wavelet Analysis: {'✅' if wavelet_ok else '❌'}")
    
    if spatial_ok:
        print("\n🎉 Ready to run! Try:")
        print("   python main_script.py --quick-test")
    else:
        print("\n🔧 Setup needed:")
        print("   1. Install dependencies: pip install -r requirements.txt")
        print("   2. Place data file in Data/ directory")
        print("   3. Run setup: python setup.py")

if __name__ == "__main__":
    main()
