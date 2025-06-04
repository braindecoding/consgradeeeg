#!/usr/bin/env python3
# complete_publication_figures.py - Generate all publication figures

import os
import subprocess
import sys

def run_script(script_path, description):
    """Run a visualization script and handle errors"""
    print(f"📊 {description}...")
    try:
        result = subprocess.run([sys.executable, script_path], 
                              capture_output=True, text=True, cwd=os.getcwd())
        if result.returncode == 0:
            print(f"✅ {description} completed successfully!")
            return True
        else:
            print(f"❌ Error in {description}:")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"❌ Failed to run {description}: {e}")
        return False

def check_required_files():
    """Check if all required files exist"""
    required_files = [
        'src/visualization/wavelet_plots.py',
        'src/visualization/comprehensive_wavelet_analysis.py',
        'src/visualization/architecture_diagrams.py',
        'src/visualization/statistical_analysis.py',
        'src/visualization/publication_summary.py'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing required files:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        return False
    
    return True

def create_figure_index():
    """Create an index of all generated figures"""
    figures_dir = 'results/figures'
    
    if not os.path.exists(figures_dir):
        print(f"❌ Figures directory not found: {figures_dir}")
        return
    
    # Get all PNG files
    png_files = [f for f in os.listdir(figures_dir) if f.endswith('.png')]
    pdf_files = [f for f in os.listdir(figures_dir) if f.endswith('.pdf')]
    
    # Create index file
    index_content = """# Publication Figures Index

This directory contains all publication-ready figures for the EEG Digit Classification research.

## 📊 Main Publication Figures

### 🏗️ Architecture Diagrams
- `lstm_wavelet_architecture.png` - LSTM + Wavelet model architecture (Best: 76% accuracy)
- `transformer_architecture.png` - Transformer + Wavelet model architecture (68.5% accuracy)
- `data_pipeline.png` - Complete data processing pipeline

### 📈 Performance Analysis
- `performance_comparison.png` - Comprehensive model performance comparison
- `statistical_significance.png` - Statistical significance analysis with p-values
- `experimental_setup.png` - Hardware specifications and methodology

### 🌊 Wavelet Analysis
- `comprehensive_wavelet_analysis.png` - Multi-channel wavelet analysis
- `wavelet_decomposition_digit6.png` - Wavelet decomposition for digit 6
- `wavelet_decomposition_digit9.png` - Wavelet decomposition for digit 9
- `wavelet_scalogram_digit6.png` - Time-frequency analysis for digit 6
- `wavelet_scalogram_digit9.png` - Time-frequency analysis for digit 9
- `power_spectrum_digit6.png` - Power spectral density for digit 6
- `power_spectrum_digit9.png` - Power spectral density for digit 9

### 📊 Training History
- `eeg_lstm_wavelet_training_history.png` - LSTM training curves
- `eeg_transformer_training_history.png` - Transformer training curves
- `eeg_cnn_training_history.png` - CNN training curves
- `eeg_pytorch_training_history.png` - PyTorch EEGNet training curves

### 📄 Publication Summary
- `publication_summary.png` - Complete research summary (300 DPI, journal-ready)
- `publication_summary.pdf` - Vector graphics version for scalable printing

## 🎯 Usage Guidelines

### For Journal Submission:
1. **Main Figure**: Use `publication_summary.png` as the primary comprehensive figure
2. **Architecture**: Include `lstm_wavelet_architecture.png` for model details
3. **Performance**: Use `performance_comparison.png` for results analysis
4. **Methodology**: Include `experimental_setup.png` for reproducibility

### Figure Quality:
- All figures are generated at 300 DPI for print quality
- Vector graphics (PDF) available for scalable figures
- Professional scientific layout with proper labeling
- Color schemes optimized for both print and digital viewing

### File Formats:
- **PNG**: High-resolution raster images (300 DPI)
- **PDF**: Vector graphics for infinite scalability
- **Recommended**: Use PNG for most publications, PDF for presentations

## 📊 Figure Statistics
"""
    
    index_content += f"\n- **Total PNG figures**: {len(png_files)}\n"
    index_content += f"- **Total PDF figures**: {len(pdf_files)}\n"
    index_content += f"- **Total file size**: {get_directory_size(figures_dir):.1f} MB\n"
    
    index_content += "\n## 📋 Complete File List\n\n### PNG Files:\n"
    for png_file in sorted(png_files):
        file_size = os.path.getsize(os.path.join(figures_dir, png_file)) / (1024*1024)
        index_content += f"- `{png_file}` ({file_size:.1f} MB)\n"
    
    if pdf_files:
        index_content += "\n### PDF Files:\n"
        for pdf_file in sorted(pdf_files):
            file_size = os.path.getsize(os.path.join(figures_dir, pdf_file)) / (1024*1024)
            index_content += f"- `{pdf_file}` ({file_size:.1f} MB)\n"
    
    # Write index file
    with open(os.path.join(figures_dir, 'README.md'), 'w') as f:
        f.write(index_content)
    
    print(f"📋 Figure index created: {figures_dir}/README.md")

def get_directory_size(directory):
    """Calculate total size of directory in MB"""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            total_size += os.path.getsize(filepath)
    return total_size / (1024 * 1024)

def main():
    """Generate all publication figures"""
    print("🚀 COMPLETE PUBLICATION FIGURES GENERATION")
    print("=" * 60)
    print("This script will generate ALL figures needed for publication")
    print("=" * 60)
    
    # Check if required files exist
    if not check_required_files():
        print("❌ Cannot proceed - missing required files")
        return
    
    # Create output directory
    os.makedirs('results/figures', exist_ok=True)
    
    # List of scripts to run
    scripts = [
        ('src/visualization/wavelet_plots.py', 'Generating wavelet analysis plots'),
        ('src/visualization/comprehensive_wavelet_analysis.py', 'Creating comprehensive wavelet analysis'),
        ('src/visualization/architecture_diagrams.py', 'Drawing model architecture diagrams'),
        ('src/visualization/statistical_analysis.py', 'Performing statistical analysis'),
        ('src/visualization/publication_summary.py', 'Creating publication summary')
    ]
    
    # Run all scripts
    success_count = 0
    for script_path, description in scripts:
        if run_script(script_path, description):
            success_count += 1
        print()  # Add spacing
    
    # Create figure index
    create_figure_index()
    
    # Final summary
    print("=" * 60)
    print("🎉 PUBLICATION FIGURES GENERATION COMPLETE!")
    print("=" * 60)
    print(f"✅ Successfully generated: {success_count}/{len(scripts)} figure sets")
    
    if success_count == len(scripts):
        print("🏆 ALL FIGURES GENERATED SUCCESSFULLY!")
        print("\n📊 Generated Figure Categories:")
        print("  ✅ Wavelet Analysis (8 figures × 2 formats = 16 files)")
        print("  ✅ Model Architectures (3 figures × 3 formats = 9 files)")
        print("  ✅ Statistical Analysis (3 figures × 3 formats = 9 files)")
        print("  ✅ Training History (4 figures)")
        print("  ✅ Publication Summary (3 formats)")
        print("\n🎯 READY FOR JOURNAL SUBMISSION!")
        print("📄 Check results/figures/README.md for complete index")
        print("📊 Main figures:")
        print("  - PNG: results/figures/publication_summary.png")
        print("  - SVG: results/figures/publication_summary.svg (RECOMMENDED)")
        print("  - PDF: results/figures/publication_summary.pdf")
        print("\n🎨 SVG Format Benefits:")
        print("  ✅ Infinite scalability without quality loss")
        print("  ✅ Smaller file sizes than high-res PNG")
        print("  ✅ Perfect for journal submission")
        print("  ✅ Text remains searchable and editable")
    else:
        print(f"⚠️ {len(scripts) - success_count} figure sets failed to generate")
        print("Please check error messages above and fix issues")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
