# 📁 Repository Structure

## 🎯 **Overview**
This repository contains a comprehensive study of EEG-based digit classification using consumer-grade hardware. The code is organized into logical modules for better maintainability and reproducibility.

## 📂 **Directory Structure**

```
📁 consgradeeeg/
├── 📁 src/                          # Source code modules
│   ├── 📁 models/                   # Model implementations
│   │   ├── lstm_wavelet.py          # LSTM + Wavelet (Best: 76% accuracy)
│   │   ├── transformer_wavelet.py   # Transformer + Wavelet (68.5% accuracy)
│   │   ├── pytorch_eegnet.py        # PyTorch EEGNet baseline
│   │   ├── tensorflow_eegnet.py     # TensorFlow EEGNet
│   │   └── spatial_features.py     # Spatial feature analysis
│   ├── 📁 preprocessing/            # Data preprocessing
│   │   ├── wavelet_features.py      # Wavelet feature extraction
│   │   └── data_loader.py           # Data loading utilities
│   └── 📁 visualization/            # Plotting & visualization
│       ├── wavelet_plots.py         # Wavelet visualizations
│       ├── create_methodology_figures.py
│       ├── generate_methodology_figures.py
│       ├── generate_advanced_figures.py
│       └── generate_paper_figures.py
├── 📁 experiments/                  # Experiment scripts
│   ├── run_comprehensive_study.py   # Main experiment runner
│   ├── model_comparison.py          # Model comparison utilities
│   └── hybrid_models.py             # Experimental hybrid models
├── 📁 results/                      # Experiment results
│   ├── 📁 final/                    # Final results for publication
│   │   ├── comprehensive_eeg_results_report.md
│   │   ├── EXPERIMENT_SUMMARY.md
│   │   ├── final_experiment_results.json
│   │   └── publication_ready_tables.tex
│   ├── 📁 timestamped/              # Historical experiment runs
│   │   ├── eeg_experiment_results_20250604_054329.json
│   │   ├── eeg_experiment_results_20250604_054329.csv
│   │   └── eeg_results_table_20250604_054329.tex
│   └── 📁 figures/                  # Generated visualizations
│       ├── wavelet_decomposition_digit6.png
│       ├── wavelet_decomposition_digit9.png
│       ├── wavelet_scalogram_digit6.png
│       ├── wavelet_scalogram_digit9.png
│       ├── eeg_lstm_wavelet_training_history.png
│       ├── eeg_transformer_training_history.png
│       └── eeg_pytorch_training_history.png
├── 📁 docs/                         # Documentation
│   └── REPOSITORY_STRUCTURE.md     # This file
├── 📁 Data/                         # EEG dataset (gitignored)
│   └── EP1.01.txt                   # MindBigData EEG file
├── 📄 README.md                     # Main documentation
├── 📄 LICENSE                       # License file
├── 📄 requirements.txt              # Python dependencies
├── 📄 setup.py                      # Package setup
└── 📄 .gitignore                    # Git ignore rules
```

## 🚀 **Quick Start**

### **1. Setup Environment:**
```bash
pip install -r requirements.txt
```

### **2. Run Best Model (LSTM + Wavelet):**
```bash
python src/models/lstm_wavelet.py
```

### **3. Run Comprehensive Study:**
```bash
python experiments/run_comprehensive_study.py
```

### **4. Generate Visualizations:**
```bash
python src/visualization/wavelet_plots.py
```

## 📊 **Model Performance Summary**

| Rank | Model | File | Accuracy | Status |
|------|-------|------|----------|--------|
| 🥇 | LSTM + Wavelet | `src/models/lstm_wavelet.py` | **76.00%** | ✅ Best |
| 🥈 | Transformer + Wavelet | `src/models/transformer_wavelet.py` | **68.50%** | ✅ Good |
| 🥉 | Random Forest (Spatial) | `src/models/spatial_features.py` | **55.67%** | ✅ Baseline |
| 4th | SVM (Spatial) | `src/models/spatial_features.py` | **51.67%** | ✅ Baseline |
| 5th | PyTorch EEGNet | `src/models/pytorch_eegnet.py` | **51.00%** | ⚠️ Overfitting |

## 🔧 **Development Guidelines**

### **Adding New Models:**
1. Create new file in `src/models/`
2. Follow existing naming convention
3. Include GPU configuration
4. Add to experiment runner

### **Adding New Features:**
1. Preprocessing → `src/preprocessing/`
2. Visualization → `src/visualization/`
3. Experiments → `experiments/`

### **Results Management:**
- **Final results** → `results/final/`
- **Timestamped runs** → `results/timestamped/`
- **Figures** → `results/figures/`

## 📝 **File Naming Conventions**

- **Models**: `{framework}_{architecture}.py`
- **Results**: `{experiment_type}_results_{timestamp}.{ext}`
- **Figures**: `{content_type}_{description}.png`
- **Experiments**: `{purpose}_{scope}.py`

## 🎯 **Key Features**

- ✅ **Modular architecture** for easy extension
- ✅ **GPU acceleration** for all deep learning models
- ✅ **Comprehensive logging** and result tracking
- ✅ **Publication-ready** outputs
- ✅ **Reproducible** experiments with fixed seeds
- ✅ **Clean separation** of concerns

## 📚 **Documentation**

- **Main README**: Project overview and setup
- **This file**: Repository structure and guidelines
- **Code comments**: Inline documentation
- **Result reports**: Detailed analysis in `results/final/`

---

**Last Updated**: June 4, 2025  
**Repository Version**: v1.0 (Post-cleanup)  
**Best Model**: LSTM + Wavelet (76% accuracy)
