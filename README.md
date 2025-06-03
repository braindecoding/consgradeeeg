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

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/consgradeeeg.git
cd consgradeeeg
```

2. Install dependencies:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
pip install tensorflow torch torchvision torchaudio
pip install scipy pywt  # For wavelet analysis
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
# TensorFlow/Keras CNN
python eeg_deep_learning.py

# PyTorch EEGNet
python eeg_pytorch.py

# Improved PyTorch with regularization
python eeg_pytorch_improved.py

# LSTM with Wavelet features
python eeg_lstm_wavelet.py

# Transformer with Wavelet features
python eeg_transformer.py

# Advanced Wavelet feature extraction
python advanced_wavelet_features.py
```

## 🏗️ Project Structure

```
consgradeeeg/
├── main_script.py                    # Main spatial classification pipeline
├── eeg_deep_learning.py             # TensorFlow/Keras EEGNet implementation
├── eeg_pytorch.py                   # PyTorch EEGNet implementation
├── eeg_pytorch_improved.py          # Enhanced PyTorch model with regularization
├── eeg_lstm_wavelet.py              # LSTM with wavelet features
├── eeg_transformer.py               # Transformer with wavelet features
├── eeg_wavelet_cnn.py               # CNN with wavelet preprocessing
├── advanced_wavelet_features.py     # Advanced wavelet feature extraction
├── hybrid_cnn_lstm_attention.py     # Hybrid CNN-LSTM with attention
├── compare_models.py                # Model comparison utilities
├── wavelet_visualization.py         # Wavelet analysis visualization
├── generate_*.py                    # Figure generation scripts
├── Data/                            # Dataset directory (create this)
│   └── EP1.01.txt                  # Your MindBigData file
├── LICENSE                          # GPL v3 License
└── README.md                        # This file
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

## 🎯 Expected Results

Based on the spatial processing hypothesis:
- **Baseline**: ~50% (random chance)
- **Good Performance**: >60% (clear spatial signal)
- **Excellent Performance**: >65% (strong spatial differentiation)

## 🔧 Configuration

### Hardware Requirements
- **Minimum**: 8GB RAM, CPU-only training
- **Recommended**: 16GB RAM, NVIDIA GPU with 4GB+ VRAM
- **EEG Device**: Emotiv EPOC (14 channels) or compatible

### Software Dependencies
```python
# Core dependencies
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0

# Deep learning frameworks
tensorflow>=2.8.0
torch>=1.11.0
torchvision>=0.12.0

# Signal processing
scipy>=1.7.0
PyWavelets>=1.3.0

# Optional: for GPU acceleration
tensorflow-gpu>=2.8.0  # or tensorflow with CUDA
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

### Common Issues

1. **Data Loading Errors**
   ```bash
   # Test data format first
   python main_script.py --quick-test
   ```

2. **Memory Issues**
   - Reduce `max_trials_per_digit` parameter
   - Use CPU-only training: `device = torch.device("cpu")`

3. **CUDA Errors**
   ```python
   # Check CUDA availability
   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")
   ```

4. **Missing Dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt  # if available
   ```

### Performance Optimization

- **GPU Training**: Ensure CUDA is properly installed
- **Batch Size**: Adjust based on available memory
- **Data Augmentation**: Enable for better generalization
- **Early Stopping**: Prevent overfitting

## 📊 Visualization

The project generates several types of visualizations:

1. **Model Performance**: Accuracy comparison, confusion matrices
2. **Feature Importance**: Most discriminative spatial features
3. **Training History**: Loss and accuracy curves
4. **Wavelet Analysis**: Time-frequency decomposition
5. **Brain Topography**: Spatial activation patterns

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

## 📚 Research Background

This project is based on research in:
- **Cognitive Neuroscience**: Spatial processing in the brain
- **Brain-Computer Interfaces**: EEG-based classification
- **Consumer EEG**: Limitations and opportunities
- **Deep Learning**: Neural networks for biosignal analysis

### Key References

- EEGNet: A compact convolutional neural network for EEG-based brain-computer interfaces
- Wavelet analysis of EEG signals for brain-computer interface applications
- Spatial processing and hemispheric specialization in digit recognition

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

**Note**: This is a research project. Results may vary depending on data quality, subject variability, and hardware limitations. Always validate findings with proper statistical analysis and peer review.
