#!/usr/bin/env python3
# architecture_diagrams.py - Create model architecture diagrams for publication

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np
import os

def create_lstm_wavelet_architecture():
    """Create LSTM + Wavelet architecture diagram"""
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    
    # Define colors
    colors = {
        'input': '#E8F4FD',
        'wavelet': '#FFE6CC',
        'lstm': '#D4E6F1',
        'attention': '#F8D7DA',
        'dense': '#D5F4E6',
        'output': '#FFF2CC'
    }
    
    # Input layer
    input_box = FancyBboxPatch((0.5, 8), 2, 1, boxstyle="round,pad=0.1", 
                               facecolor=colors['input'], edgecolor='black', linewidth=2)
    ax.add_patch(input_box)
    ax.text(1.5, 8.5, 'EEG Input\n(14 channels × 128 timepoints)', 
            ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Wavelet decomposition branch
    wavelet_box = FancyBboxPatch((0.5, 6), 2, 1, boxstyle="round,pad=0.1",
                                facecolor=colors['wavelet'], edgecolor='black', linewidth=2)
    ax.add_patch(wavelet_box)
    ax.text(1.5, 6.5, 'Wavelet Decomposition\n(db4, 4 levels)', 
            ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Wavelet features
    wavelet_feat_box = FancyBboxPatch((0.5, 4), 2, 1, boxstyle="round,pad=0.1",
                                     facecolor=colors['wavelet'], edgecolor='black', linewidth=2)
    ax.add_patch(wavelet_feat_box)
    ax.text(1.5, 4.5, 'Wavelet Features\n(280 features)', 
            ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Raw EEG processing branch
    reshape_box = FancyBboxPatch((4, 6), 2, 1, boxstyle="round,pad=0.1",
                                facecolor=colors['input'], edgecolor='black', linewidth=2)
    ax.add_patch(reshape_box)
    ax.text(5, 6.5, 'Reshape\n(128 × 14)', 
            ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Bidirectional LSTM
    lstm_box = FancyBboxPatch((4, 4), 2, 1, boxstyle="round,pad=0.1",
                             facecolor=colors['lstm'], edgecolor='black', linewidth=2)
    ax.add_patch(lstm_box)
    ax.text(5, 4.5, 'Bidirectional LSTM\n(64 units)', 
            ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Attention mechanism
    attention_box = FancyBboxPatch((4, 2), 2, 1, boxstyle="round,pad=0.1",
                                  facecolor=colors['attention'], edgecolor='black', linewidth=2)
    ax.add_patch(attention_box)
    ax.text(5, 2.5, 'Attention Mechanism\n(Temporal weighting)', 
            ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Feature fusion
    fusion_box = FancyBboxPatch((7.5, 3), 2, 1, boxstyle="round,pad=0.1",
                               facecolor=colors['dense'], edgecolor='black', linewidth=2)
    ax.add_patch(fusion_box)
    ax.text(8.5, 3.5, 'Feature Fusion\n(Concatenation)', 
            ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Dense layers
    dense1_box = FancyBboxPatch((11, 4), 2, 1, boxstyle="round,pad=0.1",
                               facecolor=colors['dense'], edgecolor='black', linewidth=2)
    ax.add_patch(dense1_box)
    ax.text(12, 4.5, 'Dense Layer\n(64 units, ReLU)', 
            ha='center', va='center', fontsize=10, fontweight='bold')
    
    dense2_box = FancyBboxPatch((11, 2), 2, 1, boxstyle="round,pad=0.1",
                               facecolor=colors['dense'], edgecolor='black', linewidth=2)
    ax.add_patch(dense2_box)
    ax.text(12, 2.5, 'Dense Layer\n(32 units, ReLU)', 
            ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Output
    output_box = FancyBboxPatch((14.5, 3), 2, 1, boxstyle="round,pad=0.1",
                               facecolor=colors['output'], edgecolor='black', linewidth=2)
    ax.add_patch(output_box)
    ax.text(15.5, 3.5, 'Output\n(2 classes: 6, 9)', 
            ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Add arrows
    arrows = [
        # Main flow
        ((1.5, 8), (1.5, 7)),      # Input to wavelet
        ((1.5, 6), (1.5, 5)),      # Wavelet to features
        ((2.5, 8.5), (4, 6.5)),    # Input to reshape
        ((5, 6), (5, 5)),          # Reshape to LSTM
        ((5, 4), (5, 3)),          # LSTM to attention
        # Fusion
        ((2.5, 4.5), (7.5, 3.5)),  # Wavelet to fusion
        ((6, 2.5), (7.5, 3.5)),    # Attention to fusion
        # Dense layers
        ((9.5, 3.5), (11, 4.5)),   # Fusion to dense1
        ((12, 4), (12, 3)),        # Dense1 to dense2
        ((13, 2.5), (14.5, 3.5)),  # Dense2 to output
    ]
    
    for start, end in arrows:
        arrow = ConnectionPatch(start, end, "data", "data",
                              arrowstyle="->", shrinkA=5, shrinkB=5,
                              mutation_scale=20, fc="black", lw=2)
        ax.add_patch(arrow)
    
    # Add title and labels
    ax.set_title('LSTM + Wavelet Architecture (Best Model: 76% Accuracy)', 
                fontsize=16, fontweight='bold', pad=20)
    
    # Add performance metrics
    metrics_text = """
Performance Metrics:
• Test Accuracy: 76.00%
• Sensitivity (Digit 6): 83.00%
• Specificity (Digit 9): 69.00%
• Training Time: ~5 minutes
• Parameters: ~50K
    """
    ax.text(0.5, 1, metrics_text, fontsize=10, 
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    ax.set_xlim(0, 17)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    return fig

def create_transformer_architecture():
    """Create Transformer + Wavelet architecture diagram"""
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    
    colors = {
        'input': '#E8F4FD',
        'wavelet': '#FFE6CC',
        'transformer': '#E1D5E7',
        'attention': '#F8D7DA',
        'dense': '#D5F4E6',
        'output': '#FFF2CC'
    }
    
    # Input
    input_box = FancyBboxPatch((1, 10), 2, 1, boxstyle="round,pad=0.1",
                               facecolor=colors['input'], edgecolor='black', linewidth=2)
    ax.add_patch(input_box)
    ax.text(2, 10.5, 'EEG Input\n(14 × 128)', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Wavelet branch
    wavelet_box = FancyBboxPatch((1, 8), 2, 1, boxstyle="round,pad=0.1",
                                facecolor=colors['wavelet'], edgecolor='black', linewidth=2)
    ax.add_patch(wavelet_box)
    ax.text(2, 8.5, 'Wavelet Features\n(280 features)', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Transformer branch
    projection_box = FancyBboxPatch((5, 8), 2, 1, boxstyle="round,pad=0.1",
                                   facecolor=colors['transformer'], edgecolor='black', linewidth=2)
    ax.add_patch(projection_box)
    ax.text(6, 8.5, 'Input Projection\n(14 → 64)', ha='center', va='center', fontsize=10, fontweight='bold')
    
    pos_encoding_box = FancyBboxPatch((5, 6), 2, 1, boxstyle="round,pad=0.1",
                                     facecolor=colors['transformer'], edgecolor='black', linewidth=2)
    ax.add_patch(pos_encoding_box)
    ax.text(6, 6.5, 'Positional\nEncoding', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Multi-head attention
    mha_box = FancyBboxPatch((8.5, 6), 3, 1, boxstyle="round,pad=0.1",
                            facecolor=colors['attention'], edgecolor='black', linewidth=2)
    ax.add_patch(mha_box)
    ax.text(10, 6.5, 'Multi-Head Attention\n(8 heads, 64 dim)', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Feed forward
    ff_box = FancyBboxPatch((8.5, 4), 3, 1, boxstyle="round,pad=0.1",
                           facecolor=colors['transformer'], edgecolor='black', linewidth=2)
    ax.add_patch(ff_box)
    ax.text(10, 4.5, 'Feed Forward\n(64 → 256 → 64)', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Global average pooling
    pool_box = FancyBboxPatch((8.5, 2), 3, 1, boxstyle="round,pad=0.1",
                             facecolor=colors['transformer'], edgecolor='black', linewidth=2)
    ax.add_patch(pool_box)
    ax.text(10, 2.5, 'Global Average\nPooling', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Feature fusion
    fusion_box = FancyBboxPatch((12.5, 5), 2, 1, boxstyle="round,pad=0.1",
                               facecolor=colors['dense'], edgecolor='black', linewidth=2)
    ax.add_patch(fusion_box)
    ax.text(13.5, 5.5, 'Feature Fusion\n(128 features)', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Output layers
    dense_box = FancyBboxPatch((12.5, 3), 2, 1, boxstyle="round,pad=0.1",
                              facecolor=colors['dense'], edgecolor='black', linewidth=2)
    ax.add_patch(dense_box)
    ax.text(13.5, 3.5, 'Dense Layer\n(64 units)', ha='center', va='center', fontsize=10, fontweight='bold')
    
    output_box = FancyBboxPatch((12.5, 1), 2, 1, boxstyle="round,pad=0.1",
                               facecolor=colors['output'], edgecolor='black', linewidth=2)
    ax.add_patch(output_box)
    ax.text(13.5, 1.5, 'Output\n(2 classes)', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Add arrows
    arrows = [
        ((2, 10), (2, 9)),         # Input to wavelet
        ((3, 10.5), (5, 8.5)),     # Input to projection
        ((6, 8), (6, 7)),          # Projection to pos encoding
        ((7, 6.5), (8.5, 6.5)),    # To attention
        ((10, 6), (10, 5)),        # Attention to FF
        ((10, 4), (10, 3)),        # FF to pooling
        ((3, 8.5), (12.5, 5.5)),   # Wavelet to fusion
        ((11.5, 2.5), (12.5, 5.5)), # Pooling to fusion
        ((13.5, 5), (13.5, 4)),    # Fusion to dense
        ((13.5, 3), (13.5, 2)),    # Dense to output
    ]
    
    for start, end in arrows:
        arrow = ConnectionPatch(start, end, "data", "data",
                              arrowstyle="->", shrinkA=5, shrinkB=5,
                              mutation_scale=20, fc="black", lw=2)
        ax.add_patch(arrow)
    
    ax.set_title('Transformer + Wavelet Architecture (68.5% Accuracy)', 
                fontsize=16, fontweight='bold', pad=20)
    
    # Add performance metrics
    metrics_text = """
Performance Metrics:
• Test Accuracy: 68.50%
• Balanced Performance
• Self-attention mechanism
• Early stopping: Epoch 21
• Parameters: ~75K
    """
    ax.text(1, 0.5, metrics_text, fontsize=10,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    return fig

def create_data_pipeline_diagram():
    """Create data processing pipeline diagram"""
    fig, ax = plt.subplots(1, 1, figsize=(18, 8))
    
    colors = {
        'raw': '#FFE6E6',
        'preprocess': '#E6F3FF',
        'feature': '#E6FFE6',
        'model': '#FFF0E6',
        'output': '#F0E6FF'
    }
    
    # Pipeline stages
    stages = [
        ('Raw EEG Data\n(MindBigData)', 1, colors['raw']),
        ('Data Loading\n& Validation', 3.5, colors['preprocess']),
        ('Preprocessing\n(Normalization)', 6, colors['preprocess']),
        ('Wavelet\nDecomposition', 8.5, colors['feature']),
        ('Feature\nExtraction', 11, colors['feature']),
        ('Model Training\n(LSTM/Transformer)', 13.5, colors['model']),
        ('Classification\nResults', 16, colors['output'])
    ]
    
    # Draw stages
    for i, (label, x, color) in enumerate(stages):
        box = FancyBboxPatch((x-0.75, 3), 1.5, 2, boxstyle="round,pad=0.1",
                            facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(box)
        ax.text(x, 4, label, ha='center', va='center', fontsize=10, fontweight='bold')
        
        # Add arrows between stages
        if i < len(stages) - 1:
            next_x = stages[i+1][1]
            arrow = ConnectionPatch((x+0.75, 4), (next_x-0.75, 4), "data", "data",
                                  arrowstyle="->", shrinkA=5, shrinkB=5,
                                  mutation_scale=20, fc="black", lw=2)
            ax.add_patch(arrow)
    
    # Add detailed annotations
    annotations = [
        (1, 2, "• 14-channel EEG\n• 128 Hz sampling\n• Digit 6 vs 9"),
        (3.5, 2, "• Format validation\n• Quality checks\n• Balance classes"),
        (6, 2, "• Length normalization\n• Amplitude scaling\n• Artifact removal"),
        (8.5, 2, "• Daubechies-4 wavelet\n• 4 decomposition levels\n• Multi-resolution"),
        (11, 2, "• Energy features\n• Statistical moments\n• Frequency bands"),
        (13.5, 2, "• Deep learning\n• GPU acceleration\n• Cross-validation"),
        (16, 2, "• 76% accuracy\n• Confusion matrix\n• Performance metrics")
    ]
    
    for x, y, text in annotations:
        ax.text(x, y, text, ha='center', va='top', fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    ax.set_title('EEG Signal Processing Pipeline', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlim(0, 17.5)
    ax.set_ylim(1, 6)
    ax.axis('off')
    
    return fig

def main():
    """Generate all architecture diagrams in multiple formats"""
    print("🏗️ Generating Model Architecture Diagrams")
    print("=" * 50)

    os.makedirs('results/figures', exist_ok=True)

    # Generate LSTM architecture
    print("📊 Creating LSTM + Wavelet architecture diagram...")
    fig1 = create_lstm_wavelet_architecture()
    # Save in multiple formats
    fig1.savefig('results/figures/lstm_wavelet_architecture.png', dpi=300, bbox_inches='tight')
    fig1.savefig('results/figures/lstm_wavelet_architecture.svg', format='svg', bbox_inches='tight')
    fig1.savefig('results/figures/lstm_wavelet_architecture.pdf', format='pdf', bbox_inches='tight')
    plt.close(fig1)

    # Generate Transformer architecture
    print("📊 Creating Transformer + Wavelet architecture diagram...")
    fig2 = create_transformer_architecture()
    # Save in multiple formats
    fig2.savefig('results/figures/transformer_architecture.png', dpi=300, bbox_inches='tight')
    fig2.savefig('results/figures/transformer_architecture.svg', format='svg', bbox_inches='tight')
    fig2.savefig('results/figures/transformer_architecture.pdf', format='pdf', bbox_inches='tight')
    plt.close(fig2)

    # Generate data pipeline
    print("📊 Creating data processing pipeline diagram...")
    fig3 = create_data_pipeline_diagram()
    # Save in multiple formats
    fig3.savefig('results/figures/data_pipeline.png', dpi=300, bbox_inches='tight')
    fig3.savefig('results/figures/data_pipeline.svg', format='svg', bbox_inches='tight')
    fig3.savefig('results/figures/data_pipeline.pdf', format='pdf', bbox_inches='tight')
    plt.close(fig3)

    print("✅ Architecture diagrams generated successfully!")
    print("📄 Files saved in multiple formats:")
    print("  PNG (300 DPI):")
    print("    - results/figures/lstm_wavelet_architecture.png")
    print("    - results/figures/transformer_architecture.png")
    print("    - results/figures/data_pipeline.png")
    print("  SVG (Vector):")
    print("    - results/figures/lstm_wavelet_architecture.svg")
    print("    - results/figures/transformer_architecture.svg")
    print("    - results/figures/data_pipeline.svg")
    print("  PDF (Vector):")
    print("    - results/figures/lstm_wavelet_architecture.pdf")
    print("    - results/figures/transformer_architecture.pdf")
    print("    - results/figures/data_pipeline.pdf")
    print("🎯 SVG files are perfect for journal submission!")

if __name__ == "__main__":
    main()
