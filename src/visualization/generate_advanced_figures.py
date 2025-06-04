#!/usr/bin/env python3
"""
Advanced Figure Generation for Spatial Digit Classification Paper
================================================================

This script generates additional advanced figures for the paper:
- Statistical Analysis with Box Plots
- Feature Importance Analysis
- Channel-wise EEG Analysis
- Confusion Matrix Enhancement

Author: Research Team
Date: 2024
"""

import os
import numpy as np
import matplotlib.pyplot as plt
try:
    import seaborn as sns
except ImportError:
    sns = None

try:
    from PIL import Image
except ImportError:
    Image = None

import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality figures
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    plt.style.use('default')

plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

class AdvancedFigureGenerator:
    def __init__(self, base_path="paper_components"):
        self.base_path = base_path
        self.output_path = "enhanced_figures"
        
        print("🔬 Advanced Figure Generator Initialized")
        print(f"📁 Base path: {self.base_path}")
        print(f"📁 Output path: {self.output_path}")
        
    def load_data(self):
        """Load core datasets"""
        print("\n📊 Loading datasets for advanced analysis...")
        
        try:
            self.data = np.load(f"{self.base_path}/core_data/EP1.01.npy")
            self.labels = np.load(f"{self.base_path}/core_data/EP1.01_labels.npy")
            self.reshaped_data = np.load(f"{self.base_path}/core_data/reshaped_data.npy")
            self.wavelet_features = np.load(f"{self.base_path}/core_data/advanced_wavelet_features.npy")
            
            print(f"✅ Advanced data loaded successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error loading data: {str(e)}")
            return False
    
    def generate_statistical_analysis(self):
        """Generate statistical analysis with box plots"""
        print("\n📊 Generating Statistical Analysis...")
        
        if not hasattr(self, 'wavelet_features'):
            print("❌ Wavelet features not loaded.")
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Statistical Analysis of EEG Features', fontsize=18, fontweight='bold')
        
        # Separate features by class
        class0_features = self.wavelet_features[self.labels == 0]
        class1_features = self.wavelet_features[self.labels == 1]
        
        # Feature statistics
        feature_means_0 = np.mean(class0_features, axis=0)
        feature_means_1 = np.mean(class1_features, axis=0)
        feature_stds_0 = np.std(class0_features, axis=0)
        feature_stds_1 = np.std(class1_features, axis=0)
        
        # Plot 1: Feature means comparison
        axes[0, 0].plot(feature_means_0[:100], 'b-', label='Digit 6', alpha=0.7)
        axes[0, 0].plot(feature_means_1[:100], 'r-', label='Digit 9', alpha=0.7)
        axes[0, 0].set_title('Feature Means Comparison (First 100 Features)', fontweight='bold')
        axes[0, 0].set_xlabel('Feature Index')
        axes[0, 0].set_ylabel('Mean Value')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Feature standard deviations
        axes[0, 1].plot(feature_stds_0[:100], 'b-', label='Digit 6', alpha=0.7)
        axes[0, 1].plot(feature_stds_1[:100], 'r-', label='Digit 9', alpha=0.7)
        axes[0, 1].set_title('Feature Standard Deviations (First 100 Features)', fontweight='bold')
        axes[0, 1].set_xlabel('Feature Index')
        axes[0, 1].set_ylabel('Standard Deviation')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Box plot of selected features
        selected_features = [0, 10, 20, 30, 40]  # Select representative features
        box_data = []
        labels_box = []
        
        for feat_idx in selected_features:
            box_data.extend([class0_features[:, feat_idx], class1_features[:, feat_idx]])
            labels_box.extend([f'F{feat_idx}\nDigit 6', f'F{feat_idx}\nDigit 9'])
        
        axes[1, 0].boxplot(box_data, labels=labels_box)
        axes[1, 0].set_title('Feature Distribution Box Plots', fontweight='bold')
        axes[1, 0].set_ylabel('Feature Value')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Feature correlation heatmap (simplified)
        # Calculate correlation between first 20 features
        corr_features = self.wavelet_features[:, :20]
        correlation_matrix = np.corrcoef(corr_features.T)
        
        im = axes[1, 1].imshow(correlation_matrix, cmap='coolwarm', aspect='auto')
        axes[1, 1].set_title('Feature Correlation Matrix (First 20 Features)', fontweight='bold')
        axes[1, 1].set_xlabel('Feature Index')
        axes[1, 1].set_ylabel('Feature Index')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=axes[1, 1])
        cbar.set_label('Correlation Coefficient')
        
        plt.tight_layout()
        output_file = f"{self.output_path}/statistical_analysis.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Statistical analysis saved: {output_file}")
        return output_file
    
    def generate_channel_analysis(self):
        """Generate channel-wise EEG analysis"""
        print("\n🧠 Generating Channel-wise EEG Analysis...")
        
        if not hasattr(self, 'reshaped_data'):
            print("❌ Reshaped data not loaded.")
            return None
        
        # Channel names for Emotiv EPOC
        channels = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
        
        fig, axes = plt.subplots(3, 2, figsize=(16, 18))
        fig.suptitle('Channel-wise EEG Analysis: Digit 6 vs Digit 9', fontsize=18, fontweight='bold')
        
        # Separate data by class
        class0_data = self.reshaped_data[self.labels == 0]  # Digit 6
        class1_data = self.reshaped_data[self.labels == 1]  # Digit 9
        
        # Plot 1: Average signal per channel
        mean_class0 = np.mean(class0_data, axis=0)
        mean_class1 = np.mean(class1_data, axis=0)
        
        channel_means_0 = np.mean(mean_class0, axis=1)
        channel_means_1 = np.mean(mean_class1, axis=1)
        
        x = np.arange(len(channels))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, channel_means_0, width, label='Digit 6', alpha=0.8, color='skyblue')
        axes[0, 0].bar(x + width/2, channel_means_1, width, label='Digit 9', alpha=0.8, color='lightcoral')
        axes[0, 0].set_title('Average Signal Amplitude by Channel', fontweight='bold')
        axes[0, 0].set_xlabel('EEG Channels')
        axes[0, 0].set_ylabel('Average Amplitude')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(channels, rotation=45)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Signal variance per channel
        var_class0 = np.var(mean_class0, axis=1)
        var_class1 = np.var(mean_class1, axis=1)
        
        axes[0, 1].bar(x - width/2, var_class0, width, label='Digit 6', alpha=0.8, color='skyblue')
        axes[0, 1].bar(x + width/2, var_class1, width, label='Digit 9', alpha=0.8, color='lightcoral')
        axes[0, 1].set_title('Signal Variance by Channel', fontweight='bold')
        axes[0, 1].set_xlabel('EEG Channels')
        axes[0, 1].set_ylabel('Variance')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(channels, rotation=45)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Sample signals from key channels
        key_channels = [2, 5, 6, 7, 11]  # F3, P7, O1, O2, F4
        key_names = [channels[i] for i in key_channels]
        
        for i, (ch_idx, ch_name) in enumerate(zip(key_channels[:4], key_names[:4])):
            row = 1 + i // 2
            col = i % 2
            
            # Plot sample signals
            sample_0 = class0_data[0, ch_idx, :]  # First sample from class 0
            sample_1 = class1_data[0, ch_idx, :]  # First sample from class 1
            
            axes[row, col].plot(sample_0, 'b-', label='Digit 6', alpha=0.7)
            axes[row, col].plot(sample_1, 'r-', label='Digit 9', alpha=0.7)
            axes[row, col].set_title(f'Sample Signals - Channel {ch_name}', fontweight='bold')
            axes[row, col].set_xlabel('Time Points')
            axes[row, col].set_ylabel('Amplitude')
            axes[row, col].legend()
            axes[row, col].grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_file = f"{self.output_path}/channel_analysis.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Channel analysis saved: {output_file}")
        return output_file

def main():
    """Main function to generate advanced figures"""
    print("🔬 STARTING ADVANCED FIGURE GENERATION")
    print("=" * 50)
    
    # Initialize generator
    generator = AdvancedFigureGenerator()
    
    # Load data
    if not generator.load_data():
        print("❌ Failed to load data. Exiting.")
        return
    
    # Generate advanced figures
    figures_generated = []
    
    # 1. Statistical Analysis
    try:
        fig1 = generator.generate_statistical_analysis()
        if fig1:
            figures_generated.append(fig1)
    except Exception as e:
        print(f"❌ Error generating statistical analysis: {str(e)}")
    
    # 2. Channel Analysis
    try:
        fig2 = generator.generate_channel_analysis()
        if fig2:
            figures_generated.append(fig2)
    except Exception as e:
        print(f"❌ Error generating channel analysis: {str(e)}")
    
    # Summary
    print("\n" + "=" * 50)
    print("🎉 ADVANCED FIGURE GENERATION COMPLETE!")
    print(f"📊 Total advanced figures generated: {len(figures_generated)}")
    for fig in figures_generated:
        print(f"   ✅ {fig}")
    
    print(f"\n📁 All advanced figures saved in: enhanced_figures/")
    print("🚀 Ready for high-impact journal submission!")

if __name__ == "__main__":
    main()
