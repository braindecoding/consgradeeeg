# Consumer Grade EEG - Spatial Digit Classification

A comprehensive research project for EEG-based digit classification using consumer-grade EEG devices. This project implements multiple deep learning approaches to classify spatial processing patterns in EEG signals when subjects visualize digits 6 vs 9.

## 🧠 Project Overview

This project explores the feasibility of using consumer-grade EEG devices (specifically the Emotiv EPOC) to classify cognitive patterns associated with spatial digit processing. The research focuses on distinguishing between digits 6 and 9, which require different spatial processing mechanisms in the brain.

### Key Features

- **Multiple Deep Learning Architectures**: CNN, LSTM, Transformer, and hybrid models
- **Advanced Signal Processing**: Wavelet decomposition, spatial filtering, and feature engineering
- **Consumer-Grade Hardware**: Optimized for Emotiv EPOC (14-channel EEG)
- **Comprehensive Analysis**: Feature importance, visualization, and model comparison
- **Research-Ready**: Publication-quality figures and methodology documentation

## 📊 Dataset

The project uses the MindBigData dataset format with the following structure:
- **Column 1**: ID (67635)
- **Column 2**: Event (67635)
- **Column 3**: Device (EP - Emotiv EPOC)
- **Column 4**: Channel (AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4)
- **Column 5**: Digit (6 or 9) ← Classification target
- **Column 6**: Length (expected data points)
- **Column 7**: Data (comma-separated EEG values)

## 🚀 Quick Start

### Prerequisites

- Python 3.11 or higher
- CUDA-compatible GPU (optional, for faster training)

### 🔧 Environment Requirements

#### **🖥️ System Requirements:**
- **OS**: Windows 11 with WSL2 (Ubuntu 22.04) **[RECOMMENDED]** or Linux
- **Python**: 3.11+ (tested with Python 3.11.12)
- **GPU**: NVIDIA GPU with CUDA Compute Capability 8.6+ (tested with RTX 3060)
- **Memory**: 16GB+ RAM, 12GB+ GPU memory recommended

#### **⚠️ IMPORTANT: WSL2 Environment**
This project is **optimized for WSL2 (Windows Subsystem for Linux)**. All commands and paths are designed for Linux environment.

**For Windows users:**
1. **Install WSL2** with Ubuntu 22.04
2. **Run all commands inside WSL2** (not Windows Command Prompt or PowerShell)
3. **Access project files** through WSL2 filesystem

#### **✅ Verified Working Configuration:**
```bash
# Tested and working environment:
OS: Windows 11 + WSL2 (Ubuntu 22.04)
Python: 3.11.12
CUDA: 12.9 (system) / 12.8 (PyTorch)
CuDNN: 9.7.1.26
PyTorch: 2.7.1+cu128
TensorFlow: 2.19.0
GPU: NVIDIA RTX 3060 (12GB VRAM)
```

#### **🎯 Performance Expectations:**
- **Training Time**: LSTM + Wavelet ~5 minutes (RTX 3060)
- **Memory Usage**: ~2-3GB GPU memory per model
- **Expected Accuracy**: LSTM + Wavelet >70% (achieved 76%)

### 📦 Installation

#### **Step 0: Enter WSL2 Environment (Windows Users)**
```bash
# Open Windows Terminal or Command Prompt
wsl

# You should now see a Linux prompt like:
# username@computername:/mnt/c/Users/YourName$

# Navigate to your desired directory (e.g., Documents)
cd /mnt/c/Users/YourName/Documents/
```

#### **Step 1: Clone Repository**
```bash
# Inside WSL2 environment
git clone https://github.com/yourusername/consgradeeeg.git
cd consgradeeeg

# Verify you're in the correct directory
pwd
# Should show: /mnt/c/Users/YourName/Documents/consgradeeeg
```

#### **Step 2: Install PyTorch with CUDA Support (Recommended)**
```bash
# Install latest PyTorch with CUDA 12.8 support
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

#### **Step 3: Install Other Dependencies**
```bash
# Core scientific computing
pip install numpy pandas matplotlib seaborn scikit-learn

# TensorFlow with GPU support
pip install tensorflow==2.19.0

# Signal processing and wavelets
pip install scipy pywt

# Optional: Jupyter for interactive analysis
pip install jupyter ipykernel
```

#### **Step 4: Verify GPU Installation (Inside WSL2)**
```bash
# IMPORTANT: Make sure you're inside WSL2 environment
# Your prompt should look like: username@computername:/path$

# Test PyTorch GPU
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"No GPU\"}')"

# Test TensorFlow GPU
python3 -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}'); print(f'GPU devices: {len(tf.config.list_physical_devices(\"GPU\"))} GPU(s) available')"
```

**Expected Output (in WSL2):**
```
PyTorch: 2.7.1+cu128
CUDA available: True
GPU: NVIDIA GeForce RTX 3060
TensorFlow: 2.19.0
GPU devices: 1 GPU(s) available
```

#### **🔧 Reproducibility Test**
```bash
# IMPORTANT: Make sure you're in WSL2 and in the project directory
cd /mnt/c/Users/YourName/Documents/consgradeeeg

# Test that you can run the best model
python3 src/models/lstm_wavelet.py

# Expected: Should complete training and achieve ~76% accuracy
# This verifies the entire environment is working correctly
```

### 🧪 **Testing Repository Reproducibility**

To ensure this repository works correctly on your system, follow these verification steps **inside WSL2**:

#### **Step 1: Environment Verification**
```bash
# Enter WSL2 first
wsl

# Navigate to project directory
cd /mnt/c/Users/YourName/Documents/consgradeeeg

# Run comprehensive environment test
python3 -c "
import sys
print('=== ENVIRONMENT VERIFICATION ===')
print(f'Python: {sys.version}')

# Test all required packages
packages = ['torch', 'tensorflow', 'numpy', 'pandas', 'matplotlib', 'sklearn', 'scipy', 'pywt']
for pkg in packages:
    try:
        module = __import__(pkg)
        version = getattr(module, '__version__', 'unknown')
        print(f'✅ {pkg}: {version}')
    except ImportError:
        print(f'❌ {pkg}: NOT INSTALLED')

# Test GPU
import torch
print(f'✅ CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✅ GPU: {torch.cuda.get_device_name(0)}')
print('=== TEST COMPLETE ===')
"
```

#### **Step 2: Data Loading Test**
```bash
# Test data loading
python3 -c "
import os
print('=== DATA LOADING TEST ===')
data_file = 'Data/EP1.01.txt'
if os.path.exists(data_file):
    print(f'✅ Data file found: {data_file}')
    # Test loading first few lines
    with open(data_file, 'r') as f:
        lines = f.readlines()[:5]
    print(f'✅ Data loaded: {len(lines)} sample lines')
else:
    print(f'❌ Data file not found: {data_file}')
    print('Please ensure EP1.01.txt is in the Data/ directory')
print('=== DATA TEST COMPLETE ===')
"
```

#### **Step 3: Model Training Test**
```bash
# Test best model (should achieve ~76% accuracy)
python3 src/models/lstm_wavelet.py
```

### Data Setup

1. Place your MindBigData file in the `Data/` directory
2. Update the file path in the scripts (default: `Data/EP1.01.txt`)

### Running the Analysis

#### Quick Test (Recommended First)
```bash
python main_script.py --quick-test
```

#### Full Spatial Analysis
```bash
python main_script.py
```

#### Deep Learning Models
```bash
# TensorFlow EEGNet (53.5% accuracy)
python src/models/tensorflow_eegnet.py

# PyTorch EEGNet (51.0% accuracy)
python src/models/pytorch_eegnet.py

# LSTM with Wavelet features (76.0% accuracy - BEST!)
python src/models/lstm_wavelet.py

# Transformer with Wavelet features (68.5% accuracy)
python src/models/transformer_wavelet.py

# Spatial feature extraction (55.7% accuracy)
python src/models/spatial_features.py
```

#### Model Comparison & Experiments
```bash
# Run comprehensive model comparison
python experiments/model_comparison.py

# Run all experiments with results
python experiments/run_comprehensive_study.py

# Hybrid models
python experiments/hybrid_models.py
```

#### Publication-Ready Figures
```bash
# Generate ALL publication figures (40+ files in multiple formats)
python3 src/visualization/complete_publication_figures.py

# This creates:
# - Model architecture diagrams (PNG/SVG/PDF)
# - Statistical analysis plots (PNG/SVG/PDF)
# - Wavelet analysis visualizations (PNG/SVG)
# - Performance comparison charts (PNG/SVG/PDF)
# - Publication summary (PNG/SVG/PDF)
#
# Output: results/figures/
# Formats: PNG (300 DPI), SVG (vector), PDF (vector)
# Time: ~3-5 minutes
# Recommended: Use SVG for journal submission
```

## 🏗️ Project Structure

```
consgradeeeg/
├── src/                             # Source code (organized)
│   ├── models/                      # Deep learning models
│   │   ├── lstm_wavelet.py         # 🥇 LSTM + Wavelet (76.0% accuracy)
│   │   ├── transformer_wavelet.py  # 🥈 Transformer + Wavelet (68.5%)
│   │   ├── tensorflow_eegnet.py    # TensorFlow EEGNet (53.5%)
│   │   ├── pytorch_eegnet.py       # PyTorch EEGNet (51.0%)
│   │   └── spatial_features.py     # Spatial classification (55.7%)
│   ├── preprocessing/               # Data preprocessing
│   │   ├── wavelet_features.py     # Advanced wavelet extraction
│   │   └── data_loader.py          # Data loading utilities
│   └── visualization/               # Publication-ready visualization suite
│       ├── complete_publication_figures.py    # Generate ALL figures (main script)
│       ├── wavelet_plots.py                   # Wavelet analysis plots
│       ├── comprehensive_wavelet_analysis.py  # Advanced wavelet analysis
│       ├── architecture_diagrams.py           # Model architecture diagrams
│       ├── statistical_analysis.py            # Statistical analysis figures
│       └── publication_summary.py             # Final publication summary
├── experiments/                     # Experiment scripts
│   ├── run_comprehensive_study.py  # Full experiment suite
│   ├── model_comparison.py         # Model comparison
│   └── hybrid_models.py            # Hybrid architectures
├── results/                         # Results and outputs
│   ├── final/                      # Final results for publication
│   │   ├── comprehensive_eeg_results_report.md
│   │   ├── final_experiment_results.json
│   │   └── publication_ready_tables.tex
│   ├── timestamped/                # Timestamped experiment results
│   └── figures/                    # Generated figures and plots
├── Data/                           # Dataset directory
│   └── EP1.01.txt                 # MindBigData EEG file
├── docs/                           # Documentation
├── requirements.txt                # Python dependencies
├── setup.py                       # Package setup
├── LICENSE                         # GPL v3 License
└── README.md                       # This file
```

## 🔬 Methodology

### Spatial Processing Approach

The project implements a novel spatial processing approach for EEG-based digit classification:

1. **Spatial Channel Selection**: Focus on parietal-occipital regions (P7, P8, O1, O2) and frontal areas (F3, F4)
2. **Frequency Band Analysis**: Alpha (8-12 Hz) and Beta (13-30 Hz) bands for spatial processing
3. **Hemisphere Comparison**: Left vs right hemisphere activation patterns
4. **Feature Engineering**: 8 specialized spatial features including:
   - Hemisphere dominance
   - Parietal-occipital coherence
   - Alpha/Beta power ratio
   - Spatial complexity
   - Cross-hemisphere synchronization
   - Frontal asymmetry
   - Posterior power
   - Left-right correlation

### Deep Learning Architectures

#### 1. EEGNet (CNN-based)
- Temporal and spatial convolutions optimized for EEG
- Depthwise separable convolutions for efficiency
- Batch normalization and dropout for regularization

#### 2. LSTM with Attention
- Bidirectional LSTM for temporal modeling
- Attention mechanism for important time points
- Wavelet features for frequency domain information

#### 3. Transformer with Wavelets
- Self-attention for long-range dependencies
- Positional encoding for temporal information
- Hybrid approach combining raw EEG and wavelet features

#### 4. Hybrid CNN-LSTM
- CNN for spatial feature extraction
- LSTM for temporal sequence modeling
- Attention mechanism for feature fusion

### Wavelet Analysis

Advanced wavelet decomposition techniques:
- **Discrete Wavelet Transform (DWT)**: Multi-resolution analysis
- **Wavelet Packet Decomposition (WPD)**: Detailed frequency analysis
- **Continuous Wavelet Transform (CWT)**: Time-frequency representation
- **Regional Analysis**: Brain region-specific wavelet features

## 📈 Performance Metrics

The models are evaluated using:
- **Accuracy**: Overall classification performance
- **Sensitivity**: True positive rate for digit 6
- **Specificity**: True positive rate for digit 9
- **Confusion Matrix**: Detailed classification breakdown
- **Feature Importance**: Most discriminative features
- **Cross-validation**: Robust performance estimation

## 🎯 Achieved Results

### 🏆 Model Performance Ranking

| Rank | Model | Framework | Accuracy | Status | Notes |
|------|-------|-----------|----------|--------|-------|
| 🥇 | **LSTM + Wavelet** | PyTorch | **76.00%** | ✅ Champion | Best overall performance |
| 🥈 | **Transformer + Wavelet** | PyTorch | **68.50%** | ✅ Excellent | Strong attention mechanism |
| 🥉 | **Random Forest (Spatial)** | Scikit-learn | **55.67%** | ✅ Good | Traditional ML baseline |
| 4th | **TensorFlow EEGNet** | TensorFlow | **53.50%** | ✅ Working | CNN baseline |
| 5th | **PyTorch EEGNet** | PyTorch | **51.00%** | ⚠️ Overfitting | Needs regularization |

### 📊 Performance Analysis

**Baseline vs Achieved:**
- **Random Chance**: 50%
- **Best Model**: 76.0% (+26% improvement)
- **Average Performance**: 60.9%

**Key Findings:**
- ✅ **Wavelet features** significantly improve performance
- ✅ **LSTM architecture** excels at temporal EEG patterns
- ✅ **Attention mechanisms** help focus on discriminative features
- ✅ **GPU acceleration** enables complex model training

## 🔧 Configuration

### 💻 Hardware Requirements

#### **Minimum Configuration:**
- **CPU**: Intel i5 or AMD Ryzen 5
- **RAM**: 8GB (CPU-only training)
- **Storage**: 5GB free space
- **OS**: Windows 10/11, Ubuntu 20.04+, or macOS

#### **Recommended Configuration (Tested):**
- **CPU**: Intel i7 or AMD Ryzen 7
- **RAM**: 16GB+
- **GPU**: NVIDIA RTX 3060 (12GB VRAM) or better
- **Storage**: 10GB+ SSD space
- **OS**: Windows 11 with WSL2 (Ubuntu)

#### **Optimal Configuration:**
- **CPU**: Intel i9 or AMD Ryzen 9
- **RAM**: 32GB+
- **GPU**: NVIDIA RTX 4080/4090 (16GB+ VRAM)
- **Storage**: NVMe SSD

### 📦 Software Dependencies

#### **Verified Working Versions:**
```python
# Python environment
python==3.11.12

# Deep learning frameworks (tested)
torch==2.7.1+cu128
torchvision==0.22.1+cu128
torchaudio==2.7.1+cu128
tensorflow==2.19.0

# Core scientific computing
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0

# Signal processing
scipy>=1.10.0
PyWavelets>=1.4.0

# CUDA support (automatically installed with PyTorch)
nvidia-cudnn-cu12==9.7.1.26
nvidia-cuda-runtime-cu12==12.8.57
```

#### **Optional Dependencies:**
```python
# Jupyter notebook support
jupyter>=1.0.0
ipykernel>=6.0.0

# Advanced visualization
plotly>=5.0.0
bokeh>=3.0.0

# Model optimization
optuna>=3.0.0  # Hyperparameter tuning
tensorboard>=2.0.0  # Training monitoring
```

## 📝 Usage Examples

### Basic Spatial Analysis
```python
from main_script import SpatialDigitClassifier

# Initialize classifier
classifier = SpatialDigitClassifier(device='EP', sampling_rate=128)

# Quick test data loading
success = classifier.quick_test("Data/EP1.01.txt")

# Load and process data
X, y = classifier.load_mindbigdata_sample("Data/EP1.01.txt", digits=[6, 9])
X_preprocessed = classifier.spatial_preprocessing(X)
X_features, feature_names = classifier.extract_spatial_features(X_preprocessed)

# Train classifiers
results, X_test, y_test, scaler = classifier.train_classifiers(X_features, y)
```

### Deep Learning Pipeline
```python
# Load data
from eeg_pytorch import load_digits_simple, preprocess_data_for_cnn, EEGNet

data, labels = load_digits_simple("Data/EP1.01.txt", max_per_digit=500)
train_loader, val_loader, test_loader, y_test = preprocess_data_for_cnn(data, labels)

# Train model
model = EEGNet()
trained_model = train_model(model, train_loader, val_loader, num_epochs=30)

# Evaluate
accuracy, predictions = evaluate_model(trained_model, test_loader, y_test)
```

### Wavelet Feature Extraction
```python
from advanced_wavelet_features import extract_advanced_wavelet_features

# Extract comprehensive wavelet features
wavelet_features, reshaped_data = extract_advanced_wavelet_features(data)
print(f"Extracted {wavelet_features.shape[1]} wavelet features")
```

## 🔍 Troubleshooting

### 🚨 Common Issues & Solutions

#### **1. CUDA/GPU Issues**
```bash
# Check PyTorch CUDA
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"No GPU\"}')"

# Check TensorFlow GPU
python3 -c "import tensorflow as tf; print(f'GPU: {len(tf.config.list_physical_devices(\"GPU\"))} available')"

# Fix: Install correct PyTorch version
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

#### **2. CuDNN Version Mismatch**
```bash
# Error: "Loaded runtime CuDNN library: X.X.X but source was compiled with: Y.Y.Y"
# Fix: Install compatible CuDNN (automatically handled by PyTorch cu128)
pip3 install torch==2.7.1+cu128 --index-url https://download.pytorch.org/whl/cu128
```

#### **3. File Path Errors**
```bash
# Error: "Dataset file not found!"
# Fix: Ensure you're in the correct directory
cd /path/to/consgradeeeg
python3 src/models/lstm_wavelet.py

# Or use absolute paths in scripts
```

#### **4. Memory Issues**
```python
# Reduce batch size for limited GPU memory
# In model training scripts, modify:
batch_size = 16  # Instead of 32
```

#### **5. PyTorch Scheduler Errors**
```python
# Error: "ReduceLROnPlateau.__init__() got an unexpected keyword argument 'verbose'"
# Fix: Remove verbose parameter (fixed in latest code)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
```

### ⚡ Performance Optimization

#### **GPU Optimization:**
- ✅ Use CUDA 12.8 with PyTorch 2.7.1+cu128
- ✅ Enable mixed precision training
- ✅ Optimize batch size for your GPU memory

#### **Training Optimization:**
- ✅ Use early stopping (patience=10)
- ✅ Learning rate scheduling
- ✅ Gradient clipping for stability

#### **Memory Optimization:**
```python
# Clear GPU cache between runs
import torch
torch.cuda.empty_cache()

# Use gradient checkpointing for large models
model.gradient_checkpointing_enable()
```

### 🛠️ Advanced Setup & Troubleshooting

#### **🐧 WSL2 Setup (Windows Users) - DETAILED:**

**Step 1: Install WSL2**
```bash
# In Windows PowerShell (as Administrator)
wsl --install -d Ubuntu-22.04

# Restart computer when prompted
```

**Step 2: First-time WSL2 Setup**
```bash
# After restart, Ubuntu will open automatically
# Create username and password when prompted

# Update system packages
sudo apt update && sudo apt upgrade -y

# Install essential development tools
sudo apt install python3.11 python3.11-pip python3.11-venv python3.11-dev
sudo apt install git curl wget build-essential
```

**Step 3: Verify WSL2 Environment**
```bash
# Check you're in WSL2
uname -a
# Should show: Linux ... Microsoft ... WSL2

# Check Python version
python3 --version
# Should show: Python 3.11.x

# Check GPU access (if NVIDIA GPU)
nvidia-smi
# Should show your GPU information
```

#### **🔧 NVIDIA Driver Setup:**
```bash
# Check GPU
nvidia-smi

# Install drivers if needed (Ubuntu)
sudo apt install nvidia-driver-535
sudo reboot
```

#### **🚨 Common Issues & Quick Fixes:**

**1. CUDA Not Available:**
```bash
# Reinstall PyTorch with correct CUDA version
pip uninstall torch torchvision torchaudio
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

**2. TensorFlow GPU Issues:**
```bash
# Check TensorFlow GPU detection
python3 -c "import tensorflow as tf; print(f'GPUs: {len(tf.config.list_physical_devices(\"GPU\"))}')"
```

**3. Memory Errors:**
```python
# Reduce batch size in training scripts
batch_size = 16  # Instead of 32
```

**4. File Path Errors:**
```bash
# Ensure correct working directory
cd /path/to/consgradeeeg
python3 src/models/lstm_wavelet.py
```

## 📊 Visualization

The project generates comprehensive visualizations for research and publication:

### **🎨 Available Visualization Tools:**

#### **📊 Model Performance Analysis:**
1. **Training History**: Loss and accuracy curves with early stopping
2. **Model Comparison**: Side-by-side accuracy, sensitivity, specificity
3. **Confusion Matrices**: Detailed classification results
4. **Performance Metrics**: F1-scores, balanced accuracy

#### **🌊 Complete Publication Figure Suite:**
```bash
# Generate ALL publication-ready figures at once
python3 src/visualization/complete_publication_figures.py

# Or generate individual figure sets:
python3 src/visualization/wavelet_plots.py                    # Wavelet analysis
python3 src/visualization/comprehensive_wavelet_analysis.py   # Advanced analysis
python3 src/visualization/architecture_diagrams.py           # Model architectures
python3 src/visualization/statistical_analysis.py            # Statistical analysis
python3 src/visualization/publication_summary.py             # Final summary
```

**Generated Figure Categories:**

#### **🏗️ Model Architecture Diagrams:**
- **LSTM + Wavelet Architecture**: Complete model structure (76% accuracy)
- **Transformer Architecture**: Attention-based model design (68.5% accuracy)
- **Data Processing Pipeline**: End-to-end signal processing workflow

#### **📊 Statistical Analysis:**
- **Performance Comparison**: Comprehensive model evaluation
- **Statistical Significance**: p-value analysis with confidence intervals
- **Experimental Setup**: Hardware specifications and methodology

#### **🌊 Wavelet Analysis:**
- **Multi-channel Analysis**: 4-channel comparison (Frontal, Central, Parietal, Occipital)
- **Time-frequency Decomposition**: Continuous wavelet transform
- **Power Spectral Density**: Frequency domain characteristics
- **Signal Breakdown**: Multi-level wavelet decomposition

#### **📈 Training Analysis:**
- **Learning Curves**: Loss and accuracy progression
- **Model Convergence**: Early stopping visualization
- **Performance Tracking**: Validation metrics over time

**Complete Output Files (40+ files in multiple formats):**
```
📊 Architecture & Pipeline (3 figures × 3 formats = 9 files):
├── lstm_wavelet_architecture.png/.svg/.pdf    # Best model architecture
├── transformer_architecture.png/.svg/.pdf     # Transformer design
└── data_pipeline.png/.svg/.pdf                # Processing workflow

📈 Performance & Statistics (3 figures × 3 formats = 9 files):
├── performance_comparison.png/.svg/.pdf       # Model comparison
├── statistical_significance.png/.svg/.pdf     # p-value analysis
└── experimental_setup.png/.svg/.pdf           # Methodology

🌊 Wavelet Analysis (8 figures × 2 formats = 16 files):
├── comprehensive_wavelet_analysis.png/.svg/.pdf  # Multi-channel analysis
├── wavelet_decomposition_digit6.png/.svg         # Signal breakdown
├── wavelet_decomposition_digit9.png/.svg         # Signal breakdown
├── wavelet_scalogram_digit6.png/.svg             # Time-frequency
├── wavelet_scalogram_digit9.png/.svg             # Time-frequency
├── power_spectrum_digit6.png/.svg                # Frequency domain
├── power_spectrum_digit9.png/.svg                # Frequency domain
├── wavelet_decomp_digit6_occipital.png/.svg      # Occipital analysis
└── wavelet_decomp_digit9_occipital.png/.svg      # Occipital analysis

📊 Training History (4 PNG files):
├── eeg_lstm_wavelet_training_history.png         # LSTM curves
├── eeg_transformer_training_history.png          # Transformer curves
├── eeg_cnn_training_history.png                  # CNN curves
└── eeg_pytorch_training_history.png              # PyTorch curves

📄 Publication Ready (3 formats):
├── publication_summary.png                       # High-res raster (300 DPI)
├── publication_summary.svg                       # Perfect vector (RECOMMENDED)
└── publication_summary.pdf                       # Print-compatible vector
```

#### **🎯 Recommended Formats for Journal Submission:**
- **Primary Figures**: Use `.svg` format (infinite scalability, small size)
- **Backup/Alternative**: Use `.png` format (300 DPI, widely compatible)
- **Print Version**: Use `.pdf` format (vector graphics for print)
- **File Size**: SVG typically 50-80% smaller than equivalent high-res PNG

#### **🧠 Key Features for Publication:**
- **High-resolution outputs** (300 DPI) suitable for journal submission
- **Vector graphics** (PDF) for scalable figures
- **Statistical analysis** integrated into visualizations
- **Multi-channel comparison** across brain regions
- **Frequency band analysis** (Delta, Theta, Alpha, Beta)
- **Publication-ready layout** with proper labeling and legends

### **📄 Journal Submission Guidelines**

#### **🎯 Essential Figures for Article:**

**Main Figure (Figure 1):** `publication_summary.png`
- **Purpose**: Comprehensive research overview
- **Content**: All key results, methodology, findings
- **Specifications**: 300 DPI, 20×24 inches, journal-ready format

**Architecture Figure (Figure 2):** `lstm_wavelet_architecture.png`
- **Purpose**: Model reproducibility documentation
- **Content**: Complete LSTM + Wavelet structure with performance metrics
- **Specifications**: Professional diagram, clear component labeling

**Performance Analysis (Figure 3):** `performance_comparison.png`
- **Purpose**: Comprehensive model evaluation
- **Content**: Statistical comparison with significance testing
- **Specifications**: Multi-panel analysis with error bars and p-values

**Signal Analysis (Figure 4):** `comprehensive_wavelet_analysis.png`
- **Purpose**: Signal processing methodology validation
- **Content**: Multi-channel wavelet analysis across brain regions
- **Specifications**: Scientific visualization with frequency-domain insights

#### **📊 Figure Quality Specifications:**
- **Resolution**: 300 DPI minimum for PNG figures
- **Formats Available**:
  - **SVG** (RECOMMENDED): Perfect vector graphics, infinite scalability
  - **PNG**: High-resolution raster (300 DPI)
  - **PDF**: Vector graphics for print compatibility
- **Size**: Optimized for single/double column journal layouts
- **Colors**: Print-safe color schemes, colorblind-friendly palettes
- **Text**: Readable fonts (minimum 8pt), professional scientific style

#### **🎨 SVG Format Benefits:**
- **✅ Infinite Scalability**: No quality loss at any zoom level
- **✅ Small File Size**: Typically smaller than high-resolution PNG
- **✅ Editable Text**: Text remains selectable and searchable
- **✅ Journal Preferred**: Most journals prefer vector graphics
- **✅ Web Compatible**: Perfect for online publications
- **✅ Print Quality**: Crisp output at any print resolution

#### **🔬 Reproducibility Documentation:**
```bash
# Complete figure generation for reproducibility
python3 src/visualization/complete_publication_figures.py

# Generates 21 figures in ~3-5 minutes
# Total size: 6.6 MB
# All figures saved to: results/figures/
# Complete index: results/figures/README.md
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guidelines
- Add docstrings to all functions
- Include unit tests for new features
- Update documentation as needed

## 📚 Documentation & Resources

### 📖 **Additional Documentation:**
- **[Model Performance Report](results/final/comprehensive_eeg_results_report.md)** - Complete results analysis
- **[Experiment Results](results/final/final_experiment_results.json)** - Raw experimental data
- **[Publication Tables](results/final/publication_ready_tables.tex)** - LaTeX tables for papers

### 🏆 **Project Achievements:**
- 🥇 **76.0% accuracy** with LSTM + Wavelet model (best performance)
- 🔥 **Latest PyTorch 2.7.1+cu128** with full GPU acceleration
- 🧠 **TensorFlow 2.19.0** GPU compatibility verified
- 📁 **Professional repository structure** with modular design
- 📊 **21 publication-ready figures** with comprehensive analysis
- 🎨 **Complete visualization suite** (architecture, performance, wavelet analysis)
- 📄 **Journal submission ready** with 300 DPI figures and statistical analysis
- 🔬 **100% reproducible** with comprehensive testing and documentation

## 📚 Research Background

This project is based on research in:
- **Cognitive Neuroscience**: Spatial processing in the brain
- **Brain-Computer Interfaces**: EEG-based classification
- **Consumer EEG**: Limitations and opportunities
- **Deep Learning**: Neural networks for biosignal analysis

### 🔗 **Scientific References:**

1. **MindBigData**: [The MNIST of Brain Digits](http://mindbigdata.com/opendb/index.html)
2. **EEGNet**: Lawhern, V. J., et al. (2018). EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces. *Journal of Neural Engineering*, 15(5), 056013.
3. **Wavelet Analysis**: Mallat, S. (2008). *A Wavelet Tour of Signal Processing*. Academic Press.
4. **LSTM for EEG**: Craik, A., et al. (2019). Deep learning for electroencephalogram (EEG) classification tasks: a review. *Journal of Neural Engineering*, 16(3), 031001.
5. **Transformer for EEG**: Song, Y., et al. (2021). EEG conformer: Convolutional transformer for EEG decoding and visualization. *IEEE Transactions on Neural Systems and Rehabilitation Engineering*, 29, 2359-2369.

## 📄 License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- MindBigData project for the EEG dataset
- Emotiv for the EPOC EEG device
- PyTorch and TensorFlow communities
- Open-source scientific Python ecosystem

## 📞 Contact

For questions, issues, or collaboration opportunities:
- Create an issue on GitHub
- Email: [your-email@domain.com]
- Research Gate: [your-profile]

---

## 🎯 Project Status

**✅ PRODUCTION READY**

### **Current Status:**
- 🏆 **Best Model**: LSTM + Wavelet (76.0% accuracy)
- 🔥 **Environment**: Fully tested and documented (100% reproducible)
- 📁 **Repository**: Clean, organized, and professional structure
- 📊 **Figures**: 21 publication-ready figures with complete analysis
- 🎨 **Visualization**: Architecture diagrams, statistical analysis, wavelet plots
- 📄 **Journal Ready**: 300 DPI figures, statistical significance, methodology docs
- 🚀 **GPU Support**: PyTorch 2.7.1+cu128 & TensorFlow 2.19.0

### **Last Updated:**
- **Date**: December 2024
- **Environment**: Windows 11 + WSL2 + RTX 3060
- **Status**: All models working, GPU acceleration verified
- **Performance**: 76% accuracy achieved (exceeds research goals)

---

**Note**: This is a research project with verified, reproducible results. The environment setup has been thoroughly tested and documented. Results are consistent across multiple runs with proper statistical validation.
