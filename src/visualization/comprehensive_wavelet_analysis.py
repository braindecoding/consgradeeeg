#!/usr/bin/env python3
# comprehensive_wavelet_analysis.py - Advanced wavelet analysis for publication

import numpy as np
import matplotlib.pyplot as plt
import pywt
import seaborn as sns
from scipy import stats
from scipy.signal import welch
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

def load_data_for_analysis():
    """Load and preprocess data for wavelet analysis"""
    file_path = "Data/EP1.01.txt"
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return None, None
    
    data = []
    labels = []
    digit_counts = {6: 0, 9: 0}
    max_per_digit = 100  # Use smaller sample for detailed analysis
    
    print(f"📂 Loading data from: {file_path}")
    
    with open(file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            parts = line.strip().split('\t')
            if len(parts) < 7:
                continue
            
            try:
                digit = int(parts[4])
                if digit not in [6, 9]:
                    continue
                
                if digit_counts[digit] >= max_per_digit:
                    continue
                
                # Parse EEG data and normalize to 1792 length
                eeg_values = [float(x) for x in parts[6].split(',')]
                
                if len(eeg_values) >= 1792:
                    normalized_values = eeg_values[:1792]
                else:
                    normalized_values = eeg_values + [0.0] * (1792 - len(eeg_values))
                
                data.append(normalized_values)
                labels.append(digit)
                digit_counts[digit] += 1
                
                if digit_counts[6] >= max_per_digit and digit_counts[9] >= max_per_digit:
                    break
                    
            except (ValueError, IndexError):
                continue
    
    print(f"✅ Loaded {len(data)} trials: {digit_counts[6]} digit-6, {digit_counts[9]} digit-9")
    
    return np.array(data), np.array(labels)

def extract_wavelet_features_detailed(signal, wavelet='db4', level=4):
    """Extract detailed wavelet features for analysis"""
    # Perform wavelet decomposition
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    
    features = {}
    
    # Approximation coefficients (low frequency)
    features['approx_energy'] = np.sum(coeffs[0]**2)
    features['approx_mean'] = np.mean(coeffs[0])
    features['approx_std'] = np.std(coeffs[0])
    features['approx_max'] = np.max(np.abs(coeffs[0]))
    
    # Detail coefficients (high frequency)
    detail_energies = []
    detail_means = []
    detail_stds = []
    detail_maxs = []
    
    for i in range(1, len(coeffs)):
        detail_energies.append(np.sum(coeffs[i]**2))
        detail_means.append(np.mean(coeffs[i]))
        detail_stds.append(np.std(coeffs[i]))
        detail_maxs.append(np.max(np.abs(coeffs[i])))
    
    features['detail_energies'] = detail_energies
    features['detail_means'] = detail_means
    features['detail_stds'] = detail_stds
    features['detail_maxs'] = detail_maxs
    
    # Total energy
    features['total_energy'] = features['approx_energy'] + sum(detail_energies)
    
    # Energy ratios
    features['approx_ratio'] = features['approx_energy'] / features['total_energy']
    features['detail_ratios'] = [e / features['total_energy'] for e in detail_energies]
    
    return features

def create_comprehensive_comparison_plot(data, labels):
    """Create comprehensive comparison plot for publication"""
    # Reshape data
    reshaped_data = []
    for trial in data:
        reshaped_data.append(trial.reshape(14, 128))
    
    reshaped_data = np.array(reshaped_data)
    
    # Select channels of interest
    channels = {
        'Frontal (F3)': 2,
        'Central (FC5)': 3,
        'Parietal (P7)': 5,
        'Occipital (O1)': 6
    }
    
    # Create figure with subplots
    fig, axes = plt.subplots(4, 3, figsize=(18, 16))
    fig.suptitle('Comprehensive Wavelet Analysis: Digit 6 vs Digit 9', fontsize=16, fontweight='bold')
    
    for row, (channel_name, channel_idx) in enumerate(channels.items()):
        # Get sample signals for each digit
        digit6_indices = np.where(labels == 6)[0]
        digit9_indices = np.where(labels == 9)[0]
        
        # Select representative signals
        np.random.seed(42)
        sample_6 = reshaped_data[np.random.choice(digit6_indices)][channel_idx]
        sample_9 = reshaped_data[np.random.choice(digit9_indices)][channel_idx]
        
        # Plot 1: Raw signals comparison
        axes[row, 0].plot(sample_6, label='Digit 6', color='blue', alpha=0.7)
        axes[row, 0].plot(sample_9, label='Digit 9', color='red', alpha=0.7)
        axes[row, 0].set_title(f'{channel_name} - Raw Signals')
        axes[row, 0].set_ylabel('Amplitude (μV)')
        axes[row, 0].legend()
        axes[row, 0].grid(True, alpha=0.3)
        
        # Plot 2: Power spectral density
        freqs_6, psd_6 = welch(sample_6, fs=128, nperseg=64)
        freqs_9, psd_9 = welch(sample_9, fs=128, nperseg=64)
        
        axes[row, 1].semilogy(freqs_6, psd_6, label='Digit 6', color='blue', alpha=0.7)
        axes[row, 1].semilogy(freqs_9, psd_9, label='Digit 9', color='red', alpha=0.7)
        axes[row, 1].set_title(f'{channel_name} - Power Spectral Density')
        axes[row, 1].set_ylabel('PSD (μV²/Hz)')
        axes[row, 1].set_xlim(0, 30)
        axes[row, 1].legend()
        axes[row, 1].grid(True, alpha=0.3)
        
        # Add frequency band markers
        axes[row, 1].axvline(x=4, color='gray', linestyle='--', alpha=0.5)
        axes[row, 1].axvline(x=8, color='gray', linestyle='--', alpha=0.5)
        axes[row, 1].axvline(x=13, color='gray', linestyle='--', alpha=0.5)
        
        # Plot 3: Wavelet energy comparison
        # Extract features for multiple trials
        features_6 = []
        features_9 = []
        
        for idx in digit6_indices[:20]:  # Use 20 trials for statistics
            signal = reshaped_data[idx][channel_idx]
            features_6.append(extract_wavelet_features_detailed(signal))
        
        for idx in digit9_indices[:20]:
            signal = reshaped_data[idx][channel_idx]
            features_9.append(extract_wavelet_features_detailed(signal))
        
        # Plot energy ratios
        approx_ratios_6 = [f['approx_ratio'] for f in features_6]
        approx_ratios_9 = [f['approx_ratio'] for f in features_9]
        
        detail1_ratios_6 = [f['detail_ratios'][0] for f in features_6]
        detail1_ratios_9 = [f['detail_ratios'][0] for f in features_9]
        
        x_pos = [0, 1]
        approx_means = [np.mean(approx_ratios_6), np.mean(approx_ratios_9)]
        detail1_means = [np.mean(detail1_ratios_6), np.mean(detail1_ratios_9)]
        
        width = 0.35
        axes[row, 2].bar([x - width/2 for x in x_pos], approx_means, width, 
                        label='Approximation', alpha=0.7, color='lightblue')
        axes[row, 2].bar([x + width/2 for x in x_pos], detail1_means, width,
                        label='Detail Level 1', alpha=0.7, color='lightcoral')
        
        axes[row, 2].set_title(f'{channel_name} - Wavelet Energy Ratios')
        axes[row, 2].set_ylabel('Energy Ratio')
        axes[row, 2].set_xticks(x_pos)
        axes[row, 2].set_xticklabels(['Digit 6', 'Digit 9'])
        axes[row, 2].legend()
        axes[row, 2].grid(True, alpha=0.3)
    
    # Set x-labels for bottom row
    for col in range(3):
        if col == 0:
            axes[-1, col].set_xlabel('Time (samples)')
        elif col == 1:
            axes[-1, col].set_xlabel('Frequency (Hz)')
        else:
            axes[-1, col].set_xlabel('Digit Class')
    
    plt.tight_layout()
    return fig

def main():
    """Main function for comprehensive wavelet analysis"""
    print("🚀 Comprehensive Wavelet Analysis for Publication")
    print("=" * 60)

    # Load data
    data, labels = load_data_for_analysis()
    if data is None:
        return

    # Create output directory
    os.makedirs('results/figures', exist_ok=True)

    # Generate comprehensive comparison plot
    print("📊 Generating comprehensive wavelet analysis plot...")
    fig = create_comprehensive_comparison_plot(data, labels)

    # Save in multiple formats
    fig.savefig('results/figures/comprehensive_wavelet_analysis.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig('results/figures/comprehensive_wavelet_analysis.svg',
                format='svg', bbox_inches='tight', facecolor='white')
    fig.savefig('results/figures/comprehensive_wavelet_analysis.pdf',
                format='pdf', bbox_inches='tight', facecolor='white')
    plt.close(fig)

    print("✅ Comprehensive wavelet analysis completed!")
    print("📊 Files saved in multiple formats:")
    print("  PNG: results/figures/comprehensive_wavelet_analysis.png")
    print("  SVG: results/figures/comprehensive_wavelet_analysis.svg")
    print("  PDF: results/figures/comprehensive_wavelet_analysis.pdf")
    print("🎯 SVG format is perfect for journal submission!")

if __name__ == "__main__":
    main()
