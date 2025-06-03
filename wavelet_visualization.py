#!/usr/bin/env python3
# wavelet_visualization.py - Script to visualize wavelet decomposition of EEG signals

import numpy as np
import os
import matplotlib.pyplot as plt
import pywt
from scipy.signal import welch
from best_model import load_digits_simple

def plot_wavelet_decomposition(signal, wavelet='db4', level=4, title='Wavelet Decomposition'):
    """Plot wavelet decomposition of a signal"""
    # Ensure level is not too high
    max_level = pywt.dwt_max_level(len(signal), pywt.Wavelet(wavelet).dec_len)
    level = min(level, max_level)

    # Perform wavelet decomposition
    coeffs = pywt.wavedec(signal, wavelet, level=level)

    # Create figure
    fig, axes = plt.subplots(level + 2, 1, figsize=(12, 8), sharex=True)

    # Plot original signal
    axes[0].plot(signal)
    axes[0].set_title('Original Signal')
    axes[0].set_ylabel('Amplitude')

    # Plot approximation coefficients
    axes[1].plot(coeffs[0])
    axes[1].set_title(f'Approximation Coefficients (Level {level})')
    axes[1].set_ylabel('Amplitude')

    # Plot detail coefficients
    for i in range(level):
        axes[i+2].plot(coeffs[i+1])
        axes[i+2].set_title(f'Detail Coefficients (Level {level-i})')
        axes[i+2].set_ylabel('Amplitude')

    # Set x-label for the last subplot
    axes[-1].set_xlabel('Sample')

    # Set main title
    fig.suptitle(title, fontsize=16)

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)

    return fig

def plot_wavelet_scalogram(signal, fs=128, wavelet='cmor1.5-1.0', title='Wavelet Scalogram'):
    """Plot wavelet scalogram of a signal"""
    # Calculate scales for frequencies of interest (1-30 Hz)
    scales = pywt.scale2frequency(wavelet, np.arange(1, 100)) / (1.0 / fs)
    scales = scales[(scales >= 1) & (scales <= 30)]  # Keep scales for 1-30 Hz

    # Perform continuous wavelet transform
    coeffs, freqs = pywt.cwt(signal, scales, wavelet)

    # Create figure
    plt.figure(figsize=(12, 8))

    # Plot scalogram
    plt.imshow(np.abs(coeffs), aspect='auto', cmap='jet',
               extent=[0, len(signal)/fs, freqs[-1], freqs[0]])

    # Add colorbar
    plt.colorbar(label='Magnitude')

    # Set labels
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title(title)

    # Add frequency bands
    plt.axhline(y=4, color='w', linestyle='--', alpha=0.5)
    plt.axhline(y=8, color='w', linestyle='--', alpha=0.5)
    plt.axhline(y=13, color='w', linestyle='--', alpha=0.5)
    plt.axhline(y=30, color='w', linestyle='--', alpha=0.5)

    # Add frequency band labels
    plt.text(len(signal)/fs + 0.5, 2, 'Delta (1-4 Hz)', color='w', ha='left', va='center')
    plt.text(len(signal)/fs + 0.5, 6, 'Theta (4-8 Hz)', color='w', ha='left', va='center')
    plt.text(len(signal)/fs + 0.5, 10, 'Alpha (8-13 Hz)', color='w', ha='left', va='center')
    plt.text(len(signal)/fs + 0.5, 20, 'Beta (13-30 Hz)', color='w', ha='left', va='center')

    plt.tight_layout()

    return plt.gcf()

def plot_power_spectrum(signal, fs=128, title='Power Spectrum'):
    """Plot power spectrum of a signal"""
    # Calculate power spectrum
    freqs, psd = welch(signal, fs=fs, nperseg=min(256, len(signal)))

    # Create figure
    plt.figure(figsize=(10, 6))

    # Plot power spectrum
    plt.semilogy(freqs, psd)

    # Add frequency bands
    plt.axvline(x=4, color='r', linestyle='--', alpha=0.5)
    plt.axvline(x=8, color='r', linestyle='--', alpha=0.5)
    plt.axvline(x=13, color='r', linestyle='--', alpha=0.5)
    plt.axvline(x=30, color='r', linestyle='--', alpha=0.5)

    # Add frequency band labels
    plt.text(2, np.max(psd), 'Delta', color='r', ha='center', va='bottom')
    plt.text(6, np.max(psd), 'Theta', color='r', ha='center', va='bottom')
    plt.text(10, np.max(psd), 'Alpha', color='r', ha='center', va='bottom')
    plt.text(20, np.max(psd), 'Beta', color='r', ha='center', va='bottom')

    # Set labels
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density (V^2/Hz)')
    plt.title(title)
    plt.grid(True)

    plt.tight_layout()

    return plt.gcf()

def compare_digits(data, labels, digit1=6, digit2=9):
    """Compare wavelet features between two digits"""
    # Reshape data to 14 channels x 128 timepoints
    reshaped_data = []
    for trial in data:
        try:
            # Reshape to 14 x 128
            reshaped = trial.reshape(14, 128)
            reshaped_data.append(reshaped)
        except ValueError:
            print(f"  ⚠️ Reshape failed for trial with length {len(trial)}")
            continue

    # Get indices for each digit
    digit1_indices = np.where(labels == digit1)[0]
    digit2_indices = np.where(labels == digit2)[0]

    # Select a random trial for each digit
    np.random.seed(42)  # For reproducibility
    digit1_idx = np.random.choice(digit1_indices)
    digit2_idx = np.random.choice(digit2_indices)

    # Get the trials
    trial1 = reshaped_data[digit1_idx]
    trial2 = reshaped_data[digit2_idx]

    # Define channels of interest
    frontal_channel = 2  # F3
    occipital_channel = 6  # O1

    # Create output directory
    os.makedirs('wavelet_plots', exist_ok=True)

    # Plot wavelet decomposition for frontal channel
    fig1 = plot_wavelet_decomposition(trial1[frontal_channel],
                                     title=f'Wavelet Decomposition - Digit {digit1} - Frontal Channel (F3)')
    fig1.savefig(f'wavelet_plots/wavelet_decomp_digit{digit1}_frontal.png')

    fig2 = plot_wavelet_decomposition(trial2[frontal_channel],
                                     title=f'Wavelet Decomposition - Digit {digit2} - Frontal Channel (F3)')
    fig2.savefig(f'wavelet_plots/wavelet_decomp_digit{digit2}_frontal.png')

    # Plot wavelet scalogram for frontal channel
    fig3 = plot_wavelet_scalogram(trial1[frontal_channel],
                                 title=f'Wavelet Scalogram - Digit {digit1} - Frontal Channel (F3)')
    fig3.savefig(f'wavelet_plots/wavelet_scalogram_digit{digit1}_frontal.png')

    fig4 = plot_wavelet_scalogram(trial2[frontal_channel],
                                 title=f'Wavelet Scalogram - Digit {digit2} - Frontal Channel (F3)')
    fig4.savefig(f'wavelet_plots/wavelet_scalogram_digit{digit2}_frontal.png')

    # Plot power spectrum for frontal channel
    fig5 = plot_power_spectrum(trial1[frontal_channel],
                              title=f'Power Spectrum - Digit {digit1} - Frontal Channel (F3)')
    fig5.savefig(f'wavelet_plots/power_spectrum_digit{digit1}_frontal.png')

    fig6 = plot_power_spectrum(trial2[frontal_channel],
                              title=f'Power Spectrum - Digit {digit2} - Frontal Channel (F3)')
    fig6.savefig(f'wavelet_plots/power_spectrum_digit{digit2}_frontal.png')

    # Plot wavelet decomposition for occipital channel
    fig7 = plot_wavelet_decomposition(trial1[occipital_channel],
                                     title=f'Wavelet Decomposition - Digit {digit1} - Occipital Channel (O1)')
    fig7.savefig(f'wavelet_plots/wavelet_decomp_digit{digit1}_occipital.png')

    fig8 = plot_wavelet_decomposition(trial2[occipital_channel],
                                     title=f'Wavelet Decomposition - Digit {digit2} - Occipital Channel (O1)')
    fig8.savefig(f'wavelet_plots/wavelet_decomp_digit{digit2}_occipital.png')

    print(f"✅ Wavelet visualizations saved to 'wavelet_plots' directory")

def main():
    """Main function"""
    print("🚀 Wavelet Visualization for EEG Digit Classification")
    print("=" * 50)

    # Load data
    file_path = "Data/EP1.01.txt"
    data, labels = load_digits_simple(file_path, max_per_digit=500)

    if data is None:
        print("❌ Failed to load data")
        return

    # Compare digits
    compare_digits(data, labels)

    print("\n✅ Analysis completed!")

if __name__ == "__main__":
    main()
