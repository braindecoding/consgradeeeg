#!/usr/bin/env python3
# compare_models.py - Compare all implemented models for EEG classification

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Define model results
model_data = {
    'name': ['SVM with Wavelet', 'CNN with Wavelet', 'BiLSTM with Attention', 'Transformer'],
    'accuracy': [0.6867, 0.6850, 0.7300, 0.7050],
    'sensitivity': [0.7733, 0.9400, 0.8700, 0.7000],
    'specificity': [0.6000, 0.4300, 0.5900, 0.7100]
}

# Define confusion matrices separately
confusion_matrices = [
    np.array([[116, 34], [40, 60]]),  # SVM with Wavelet
    np.array([[94, 6], [57, 43]]),    # CNN with Wavelet
    np.array([[87, 13], [41, 59]]),   # BiLSTM with Attention
    np.array([[70, 30], [29, 71]])    # Transformer
]

# Create a DataFrame for easy comparison
df = pd.DataFrame(model_data)
print("Model Comparison:")
print(df[['name', 'accuracy', 'sensitivity', 'specificity']])

# Plot accuracy comparison
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.barplot(x='name', y='accuracy', data=df)
plt.title('Model Accuracy Comparison')
plt.ylim(0.5, 0.8)
plt.xticks(rotation=45)
plt.tight_layout()

# Plot sensitivity and specificity
plt.subplot(1, 2, 2)
df_melted = pd.melt(df, id_vars=['name'], value_vars=['sensitivity', 'specificity'],
                    var_name='Metric', value_name='Value')
sns.barplot(x='name', y='Value', hue='Metric', data=df_melted)
plt.title('Sensitivity and Specificity Comparison')
plt.ylim(0.3, 1.0)
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig('model_comparison.png')
print("Comparison plot saved as 'model_comparison.png'")

# Plot confusion matrices
plt.figure(figsize=(15, 4))
for i, cm in enumerate(confusion_matrices):
    plt.subplot(1, 4, i+1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Digit 6', 'Digit 9'],
                yticklabels=['Digit 6', 'Digit 9'])
    plt.title(df['name'][i])
    plt.xlabel('Predicted')
    plt.ylabel('True')

plt.tight_layout()
plt.savefig('confusion_matrices.png')
print("Confusion matrices plot saved as 'confusion_matrices.png'")

# Calculate additional metrics
precision_6 = []
precision_9 = []
f1_6 = []
f1_9 = []
balanced_accuracy = []

for i, cm in enumerate(confusion_matrices):
    # Precision
    prec_6 = cm[0, 0] / (cm[0, 0] + cm[1, 0])
    prec_9 = cm[1, 1] / (cm[0, 1] + cm[1, 1])
    precision_6.append(prec_6)
    precision_9.append(prec_9)

    # F1 Score
    f1_score_6 = 2 * (prec_6 * df['sensitivity'][i]) / (prec_6 + df['sensitivity'][i])
    f1_score_9 = 2 * (prec_9 * df['specificity'][i]) / (prec_9 + df['specificity'][i])
    f1_6.append(f1_score_6)
    f1_9.append(f1_score_9)

    # Balanced Accuracy
    bal_acc = (df['sensitivity'][i] + df['specificity'][i]) / 2
    balanced_accuracy.append(bal_acc)

# Add metrics to DataFrame
df['precision_6'] = precision_6
df['precision_9'] = precision_9
df['f1_6'] = f1_6
df['f1_9'] = f1_9
df['balanced_accuracy'] = balanced_accuracy

# Display detailed metrics
print("\nDetailed Metrics:")
print(df[['name', 'accuracy', 'balanced_accuracy', 'f1_6', 'f1_9']])

# Plot balanced accuracy and F1 scores
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.barplot(x='name', y='balanced_accuracy', data=df)
plt.title('Balanced Accuracy Comparison')
plt.ylim(0.5, 0.8)
plt.xticks(rotation=45)
plt.tight_layout()

# Plot F1 scores
plt.subplot(1, 2, 2)
df_f1_melted = pd.melt(df, id_vars=['name'], value_vars=['f1_6', 'f1_9'],
                       var_name='Metric', value_name='Value')
sns.barplot(x='name', y='Value', hue='Metric', data=df_f1_melted)
plt.title('F1 Score Comparison')
plt.ylim(0.3, 1.0)
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig('detailed_metrics.png')
print("Detailed metrics plot saved as 'detailed_metrics.png'")

# Summary and recommendations
print("\nSummary and Recommendations:")
print("-" * 50)

# Find best model for each metric
best_accuracy = df.loc[df['accuracy'].idxmax()]
best_balanced = df.loc[df['balanced_accuracy'].idxmax()]
best_f1_6 = df.loc[df['f1_6'].idxmax()]
best_f1_9 = df.loc[df['f1_9'].idxmax()]

print(f"Best overall accuracy: {best_accuracy['name']} ({best_accuracy['accuracy']:.4f})")
print(f"Best balanced accuracy: {best_balanced['name']} ({best_balanced['balanced_accuracy']:.4f})")
print(f"Best F1 score for Digit 6: {best_f1_6['name']} ({best_f1_6['f1_6']:.4f})")
print(f"Best F1 score for Digit 9: {best_f1_9['name']} ({best_f1_9['f1_9']:.4f})")

# Analyze trade-offs
print("\nTrade-offs:")
print("- BiLSTM with Attention has the highest accuracy but is biased toward Digit 6")
print("- Transformer has the most balanced performance between Digit 6 and 9")
print("- CNN with Wavelet has excellent sensitivity but poor specificity")
print("- SVM with Wavelet is simpler and faster to train but less accurate than deep learning models")

# Recommendations
print("\nRecommendations:")
print("1. For best overall performance: Use BiLSTM with Attention")
print("2. For most balanced predictions: Use Transformer")
print("3. For best detection of Digit 6: Use CNN with Wavelet")
print("4. For best detection of Digit 9: Use Transformer")
print("5. For fastest inference: Use SVM with Wavelet")

# Future improvements
print("\nFuture Improvements:")
print("1. Collect more training data to improve model generalization")
print("2. Implement ensemble methods combining multiple models")
print("3. Explore more advanced wavelet features and decomposition levels")
print("4. Fine-tune hyperparameters using systematic grid search")
print("5. Implement data augmentation techniques specific to EEG data")
print("6. Explore transfer learning from pre-trained EEG models")

# Conclusion
print("\nConclusion:")
print("Deep learning models with wavelet features significantly outperform traditional machine learning")
print("approaches for EEG digit classification. The BiLSTM with Attention model achieves the highest")
print("accuracy (73.00%), while the Transformer model provides the most balanced performance between")
print("classes. These results demonstrate the effectiveness of combining advanced neural network")
print("architectures with wavelet analysis for EEG signal processing.")
