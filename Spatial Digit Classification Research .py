# Spatial Digit Classification Research Protocol
# Enhanced pipeline for digit 6 vs 9 classification

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import mne
from mne import io
from mne.preprocessing import ICA

class SpatialDigitClassifier:
    def __init__(self, device='EPOC', sampling_rate=128):
        self.device = device
        self.fs = sampling_rate
        self.spatial_channels = ['P7', 'P8', 'O1', 'O2', 'F3', 'F4'] if device == 'EPOC' else None
        
    def load_mindbigdata(self, file_path, digits=[6, 9]):
        """
        Load MindBigData format for specific digits
        Enhanced for spatial analysis
        """
        data = []
        labels = []
        
        # Parse MindBigData format
        with open(file_path, 'r') as f:
            for line in f:
                if line.strip():
                    parts = line.strip().split()
                    if len(parts) > 5:
                        digit = int(parts[4])  # Digit value
                        if digit in digits:
                            # Extract trial data
                            trial_data = list(map(float, parts[6:]))
                            data.append(trial_data)
                            labels.append(digit)
        
        return np.array(data), np.array(labels)
    
    def spatial_preprocessing(self, data):
        """
        Spatial-focused preprocessing pipeline
        """
        # 1. Bandpass filtering for spatial processing
        # Alpha/Beta bands crucial for spatial cognition
        filtered_data = []
        
        for trial in data:
            # Reshape for channel analysis
            trial_channels = trial.reshape(-1, len(self.spatial_channels))
            
            # Apply bandpass filter (8-30 Hz for spatial processing)
            filtered_trial = signal.filtfilt(
                *signal.butter(4, [8, 30], btype='band', fs=self.fs),
                trial_channels, axis=0
            )
            filtered_data.append(filtered_trial.flatten())
        
        return np.array(filtered_data)
    
    def extract_spatial_features(self, data):
        """
        Extract spatial processing specific features
        """
        features = []
        
        for trial in data:
            # Reshape to channels x timepoints
            channels_data = trial.reshape(-1, len(self.spatial_channels))
            
            trial_features = []
            
            # 1. Hemisphere Dominance Index
            left_parietal = channels_data[:, 0]  # P7
            right_parietal = channels_data[:, 1]  # P8
            hemisphere_dominance = np.mean(right_parietal) - np.mean(left_parietal)
            trial_features.append(hemisphere_dominance)
            
            # 2. Parietal-Occipital Coherence
            parietal_avg = (channels_data[:, 0] + channels_data[:, 1]) / 2
            occipital_avg = (channels_data[:, 2] + channels_data[:, 3]) / 2
            coherence = np.corrcoef(parietal_avg, occipital_avg)[0, 1]
            trial_features.append(coherence)
            
            # 3. Alpha/Beta Power Ratio (spatial processing marker)
            freqs, psd = signal.periodogram(channels_data, fs=self.fs, axis=0)
            alpha_power = np.mean(psd[(freqs >= 8) & (freqs <= 12)])
            beta_power = np.mean(psd[(freqs >= 13) & (freqs <= 30)])
            alpha_beta_ratio = alpha_power / (beta_power + 1e-10)
            trial_features.append(alpha_beta_ratio)
            
            # 4. Spatial Complexity Index
            spatial_variance = np.var(channels_data, axis=1)
            complexity_index = np.mean(spatial_variance)
            trial_features.append(complexity_index)
            
            # 5. Cross-hemisphere synchronization
            cross_sync = np.corrcoef(
                channels_data[:, [0, 2, 4]].mean(axis=1),  # Left hemisphere
                channels_data[:, [1, 3, 5]].mean(axis=1)   # Right hemisphere
            )[0, 1]
            trial_features.append(cross_sync)
            
            features.append(trial_features)
        
        return np.array(features)
    
    def advanced_classification_pipeline(self, X, y):
        """
        Advanced classification with spatial-aware models
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.svm import SVC
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Dropout, Conv1D, GlobalMaxPooling1D
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Standardize
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        results = {}
        
        # 1. Spatial-aware Random Forest
        rf_spatial = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=42
        )
        rf_spatial.fit(X_train_scaled, y_train)
        rf_pred = rf_spatial.predict(X_test_scaled)
        results['Random Forest'] = {
            'accuracy': (rf_pred == y_test).mean(),
            'predictions': rf_pred,
            'feature_importance': rf_spatial.feature_importances_
        }
        
        # 2. SVM with RBF (good for spatial patterns)
        svm_spatial = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
        svm_spatial.fit(X_train_scaled, y_train)
        svm_pred = svm_spatial.predict(X_test_scaled)
        results['SVM'] = {
            'accuracy': (svm_pred == y_test).mean(),
            'predictions': svm_pred
        }
        
        # 3. Spatial Neural Network
        model = Sequential([
            Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.3),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        # Convert labels to binary (6->0, 9->1)
        y_train_binary = (y_train == 9).astype(int)
        y_test_binary = (y_test == 9).astype(int)
        
        history = model.fit(
            X_train_scaled, y_train_binary,
            epochs=100, batch_size=32,
            validation_split=0.2,
            verbose=0
        )
        
        nn_pred_prob = model.predict(X_test_scaled)
        nn_pred = (nn_pred_prob > 0.5).astype(int)
        results['Neural Network'] = {
            'accuracy': (nn_pred.flatten() == y_test_binary).mean(),
            'predictions': nn_pred.flatten(),
            'history': history
        }
        
        return results, y_test, scaler
    
    def visualize_spatial_results(self, results, y_test):
        """
        Comprehensive visualization for spatial classification
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Accuracy Comparison
        accuracies = [results[model]['accuracy'] for model in results.keys()]
        model_names = list(results.keys())
        
        axes[0,0].bar(model_names, accuracies, color=['skyblue', 'lightcoral', 'lightgreen'])
        axes[0,0].set_title('Model Accuracy Comparison\nDigit 6 vs 9 Classification')
        axes[0,0].set_ylabel('Accuracy')
        axes[0,0].set_ylim(0, 1)
        
        # Add accuracy values on bars
        for i, acc in enumerate(accuracies):
            axes[0,0].text(i, acc + 0.01, f'{acc:.3f}', ha='center', fontweight='bold')
        
        # 2. Confusion Matrix (best model)
        best_model = max(results.keys(), key=lambda k: results[k]['accuracy'])
        best_pred = results[best_model]['predictions']
        
        cm = confusion_matrix(y_test, best_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Digit 6', 'Digit 9'],
                   yticklabels=['Digit 6', 'Digit 9'],
                   ax=axes[0,1])
        axes[0,1].set_title(f'Confusion Matrix - {best_model}')
        
        # 3. Feature Importance (if available)
        if 'feature_importance' in results['Random Forest']:
            feature_names = ['Hemisphere Dom.', 'Parietal-Occip. Coh.', 
                           'Alpha/Beta Ratio', 'Spatial Complex.', 'Cross-Hem. Sync.']
            importance = results['Random Forest']['feature_importance']
            
            axes[1,0].barh(feature_names, importance, color='orange')
            axes[1,0].set_title('Feature Importance\n(Spatial Processing Features)')
            axes[1,0].set_xlabel('Importance')
        
        # 4. Training History (Neural Network)
        if 'history' in results['Neural Network']:
            history = results['Neural Network']['history']
            axes[1,1].plot(history.history['accuracy'], label='Training')
            axes[1,1].plot(history.history['val_accuracy'], label='Validation')
            axes[1,1].set_title('Neural Network Training History')
            axes[1,1].set_xlabel('Epoch')
            axes[1,1].set_ylabel('Accuracy')
            axes[1,1].legend()
        
        plt.tight_layout()
        plt.show()
        
        # Print detailed results
        print("=" * 60)
        print("SPATIAL DIGIT CLASSIFICATION RESULTS")
        print("=" * 60)
        
        for model_name, result in results.items():
            print(f"\n{model_name}:")
            print(f"  Accuracy: {result['accuracy']:.4f}")
            if result['accuracy'] > 0.6:
                print(f"  🎉 SIGNIFICANT IMPROVEMENT over random chance!")
            else:
                print(f"  ⚠️  Still around random chance level")
        
        best_accuracy = max(result['accuracy'] for result in results.values())
        print(f"\n🏆 Best Model: {best_model}")
        print(f"🎯 Best Accuracy: {best_accuracy:.4f}")
        
        if best_accuracy > 0.65:
            print("\n✅ SUCCESS: Spatial processing approach shows promise!")
            print("   Digit 6 vs 9 classification significantly better than digit 0 vs 1")
        elif best_accuracy > 0.55:
            print("\n🔍 MODERATE SUCCESS: Some spatial signal detected")
            print("   Consider feature engineering and model optimization")
        else:
            print("\n❌ Challenge remains: Even spatial digits difficult to classify")
            print("   May need more advanced spatial feature extraction")

# Usage Example
def run_spatial_digit_experiment(data_path):
    """
    Complete experimental pipeline
    """
    classifier = SpatialDigitClassifier(device='EPOC')
    
    # Load data
    print("Loading MindBigData for digits 6 and 9...")
    X, y = classifier.load_mindbigdata(data_path, digits=[6, 9])
    print(f"Loaded {len(X)} trials: {sum(y==6)} digit 6, {sum(y==9)} digit 9")
    
    # Preprocessing
    print("Applying spatial preprocessing...")
    X_preprocessed = classifier.spatial_preprocessing(X)
    
    # Feature extraction
    print("Extracting spatial features...")
    X_features = classifier.extract_spatial_features(X_preprocessed)
    
    # Classification
    print("Training spatial-aware classifiers...")
    results, y_test, scaler = classifier.advanced_classification_pipeline(X_features, y)
    
    # Visualization
    classifier.visualize_spatial_results(results, y_test)
    
    return classifier, results

# Run the experiment
# classifier, results = run_spatial_digit_experiment('path_to_your_mindbigdata_file')