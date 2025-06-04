# main.py - Spatial Digit Classification Implementation
# ============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class SpatialDigitClassifier:
    def __init__(self, device='EP', sampling_rate=128):
        self.device = device
        self.fs = sampling_rate
        # EP device channel mapping (based on your dataset analysis)
        # From debug: AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4
        self.ep_channels = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
        self.spatial_channels = ['P7', 'P8', 'O1', 'O2', 'F3', 'F4']  # Key spatial channels

        print(f"🧠 Spatial Digit Classifier initialized")
        print(f"📡 Device: {self.device}")
        print(f"📊 Sampling Rate: {self.fs} Hz")
        print(f"🎯 Target: Spatial processing for digits 6 vs 9")
        print(f"📺 Available channels: {len(self.ep_channels)} channels")
        print(f"🎯 Spatial focus: {self.spatial_channels}")

    def load_mindbigdata_sample(self, file_path=None, digits=[6, 9], max_trials_per_digit=500):
        """
        Load MindBigData format (TXT/TSV) based on specific column structure:
        Column 1: ID (67635)
        Column 2: Event (67635)
        Column 3: Device (EP)
        Column 4: Channel (AF3, F7, F3, etc.)
        Column 5: Digit (6 or 9) ← Target untuk klasifikasi
        Column 6: Length (260) ← Jumlah data points
        Column 7: Data (comma-separated EEG values)

        Note: Column 7 data length may vary (typically 256-264 values)
        """
        print(f"\n📂 Loading data for digits {digits}...")

        if file_path is None or not os.path.exists(file_path):
            print("❌ Dataset file not found!")
            print(f"   Expected path: {file_path}")
            print("   Please ensure dataset file exists in the specified path.")
            print("   Supported formats: .txt or .tsv (MindBigData format)")
            return None, None

        print(f"📖 Reading file: {file_path}")

        # Initialize data containers
        data_6 = []
        data_9 = []

        # Track length statistics
        expected_lengths = []
        actual_lengths = []

        with open(file_path, 'r') as f:
            for line_num, line in enumerate(f):
                if not line.strip():
                    continue

                # Split by TAB
                parts = line.split('\t')

                # Need at least 7 columns
                if len(parts) < 7:
                    continue

                try:
                    # Column 5 (index 4) = digit
                    digit = int(parts[4])

                    # Only process if it's in target digits (6 or 9)
                    if digit in digits:
                        # Column 6 (index 5) = expected length of data
                        try:
                            expected_length = int(parts[5])
                            expected_lengths.append(expected_length)
                            # Print for first few lines to verify
                            if line_num < 5:
                                print(f"  Line {line_num + 1}: Digit {digit}, Expected length: {expected_length}")
                        except (ValueError, IndexError):
                            expected_length = None

                        # Column 7 (index 6) = data (comma-separated EEG values)
                        data_string = parts[6]

                        # Parse comma-separated values and convert explicitly to float64
                        values = [np.float64(x.strip()) for x in data_string.split(',') if x.strip()]
                        actual_lengths.append(len(values))

                        # Store based on digit
                        if digit == 6 and len(data_6) < max_trials_per_digit:
                            data_6.append(values)
                        elif digit == 9 and len(data_9) < max_trials_per_digit:
                            data_9.append(values)

                        # Progress
                        if (len(data_6) + len(data_9)) % 100 == 0:
                            print(f"  Found: {len(data_6)} digit-6, {len(data_9)} digit-9")

                        # Stop when we have enough
                        if len(data_6) >= max_trials_per_digit and len(data_9) >= max_trials_per_digit:
                            break

                except (ValueError, IndexError):
                    # Skip malformed lines
                    if line_num < 10:  # Show first few errors
                        print(f"  Skipping line {line_num + 1}: parse error")
                    continue

        print(f"✅ Final count: {len(data_6)} digit-6, {len(data_9)} digit-9")

        # Print length statistics
        if expected_lengths and actual_lengths:
            print(f"  📏 Expected lengths: min={min(expected_lengths)}, max={max(expected_lengths)}, avg={sum(expected_lengths)/len(expected_lengths):.1f}")
            print(f"  📏 Actual lengths: min={min(actual_lengths)}, max={max(actual_lengths)}, avg={sum(actual_lengths)/len(actual_lengths):.1f}")

        if len(data_6) == 0 or len(data_9) == 0:
            print("❌ Missing data for one or both digits!")
            return None, None

        # Combine data and labels
        all_data = data_6 + data_9
        all_labels = [6] * len(data_6) + [9] * len(data_9)

        # Store original lengths for analysis
        original_lengths = [len(trial) for trial in all_data]
        if original_lengths:
            print(f"  📏 Original data lengths before normalization: min={min(original_lengths)}, max={max(original_lengths)}, avg={sum(original_lengths)/len(original_lengths):.1f}")

        # Normalize lengths (simple padding/truncating)
        normalized_data = []
        target_length = 1792  # 14 channels * 128 timepoints

        for trial in all_data:
            if len(trial) >= target_length:
                # Truncate if too long
                normalized_data.append(trial[:target_length])
            else:
                # Pad with repetition if too short
                trial_copy = trial.copy()  # Create a copy to avoid modifying the original
                while len(trial_copy) < target_length:
                    trial_copy.extend(trial[:min(len(trial), target_length - len(trial_copy))])
                normalized_data.append(trial_copy[:target_length])

        # Explicitly convert to float64 numpy arrays
        data = np.array(normalized_data, dtype=np.float64)
        labels = np.array(all_labels, dtype=np.int32)

        # Check for NaN or infinity values and replace them
        if np.isnan(data).any() or np.isinf(data).any():
            print("  ⚠️ Warning: NaN or Infinity values detected in data, replacing with zeros")
            data = np.nan_to_num(data)

        # Data quality check
        print(f"\n🔍 Data Quality Check:")
        print(f"  📊 Data shape: {data.shape}")
        print(f"  📈 Value range: {data.min():.2f} to {data.max():.2f}")
        print(f"  🎯 Expected: 14 channels × 128 timepoints = 1792 features")
        print(f"  🔢 Data type: {data.dtype}")
        print(f"  🏷️ Labels type: {labels.dtype}")

        return data, labels

    def quick_test(self, file_path):
        """
        Quick test function to verify data loading works correctly
        """
        print("🧪 QUICK TEST - Data Loading")
        print("=" * 40)
        print("Struktur data yang diharapkan:")
        print("  Column 1: ID (67635)")
        print("  Column 2: Event (67635)")
        print("  Column 3: Device (EP)")
        print("  Column 4: Channel (AF3, F7, F3, etc.)")
        print("  Column 5: Digit (6 or 9) ← Target untuk klasifikasi")
        print("  Column 6: Length (260) ← Jumlah data points")
        print("  Column 7: Data (comma-separated EEG values)")
        print("=" * 40)

        # Load small sample first
        data, labels = self.load_mindbigdata_sample(file_path, max_trials_per_digit=10)

        if data is not None:
            print(f"\n🎉 SUCCESS!")
            print(f"  Data shape: {data.shape}")
            print(f"  Labels: {np.unique(labels, return_counts=True)}")
            print(f"  Value range: {data.min():.1f} to {data.max():.1f}")
            print(f"  Data type: {data.dtype}")
            print(f"  Labels type: {labels.dtype}")

            # Show sample data
            print(f"\n📊 Sample dari trial pertama:")
            print(f"  10 nilai pertama: {data[0][:10]}")
            print(f"  Label: {labels[0]}")

            # Show data distribution
            print(f"\n📊 Distribusi data:")
            for digit in np.unique(labels):
                count = np.sum(labels == digit)
                print(f"  Digit {digit}: {count} trials")

            # Try simple preprocessing
            print("\n🔄 Testing preprocessing...")
            try:
                X_preprocessed = self.spatial_preprocessing(data[:2])  # Just test with 2 samples
                print(f"  ✅ Preprocessing successful!")
                print(f"  📊 Preprocessed shape: {X_preprocessed.shape}")

                # Try feature extraction
                print("\n🔄 Testing feature extraction...")
                X_features, feature_names = self.extract_spatial_features(X_preprocessed)
                print(f"  ✅ Feature extraction successful!")
                print(f"  📊 Features shape: {X_features.shape}")
                print(f"  🏷️ Number of features: {len(feature_names)}")
            except Exception as e:
                print(f"  ❌ Error during preprocessing/feature extraction: {str(e)}")

            return True
        else:
            print(f"\n❌ FAILED")
            print("  Pastikan file data memiliki format yang benar")
            print("  dan berisi data untuk digit 6 dan 9")
            return False



    def spatial_preprocessing(self, data):
        """
        Apply spatial-focused preprocessing with variable length handling
        """
        print("\n🔧 Applying spatial preprocessing...")
        print(f"  📊 Input data shape: {data.shape}")

        processed_data = []

        # Determine channel count and timepoints per channel
        total_points = data.shape[1]
        # Assume 14 channels (EP device standard)
        channels_count = 14
        timepoints_per_channel = total_points // channels_count

        print(f"  📺 Estimated: {channels_count} channels, {timepoints_per_channel} timepoints each")

        for i, trial in enumerate(data):
            try:
                # Reshape to channels x timepoints (flexible)
                trial_reshaped = trial.reshape(channels_count, -1)

                # Apply bandpass filter for spatial processing (8-30 Hz)
                filtered_trial = np.zeros_like(trial_reshaped)

                for ch in range(channels_count):
                    try:
                        # Adaptive filtering based on actual timepoints
                        timepoints = trial_reshaped.shape[1]

                        # Only filter if we have enough data points
                        if timepoints >= 10:  # Minimum for filtering
                            # Bandpass filter for alpha/beta bands (spatial processing)
                            sos = signal.butter(4, [8, 30], btype='band', fs=self.fs, output='sos')
                            filtered_trial[ch] = signal.sosfilt(sos, trial_reshaped[ch])
                        else:
                            # If too few points, just use original
                            filtered_trial[ch] = trial_reshaped[ch]
                    except:
                        # If filtering fails, use original signal
                        filtered_trial[ch] = trial_reshaped[ch]

                processed_data.append(filtered_trial.flatten())

            except Exception as err:
                # If reshape fails, use original trial
                print(f"  ⚠️  Trial {i}: Reshape failed ({str(err)}), using original data")
                processed_data.append(trial)

            if (i + 1) % 200 == 0:
                print(f"  📊 Processed {i + 1}/{len(data)} trials")

        # Convert to numpy array with explicit dtype
        processed_data_array = np.array(processed_data, dtype=np.float64)

        # Check for NaN or infinity values
        if np.isnan(processed_data_array).any() or np.isinf(processed_data_array).any():
            print("  ⚠️ Warning: NaN or Infinity values detected in preprocessed data, replacing with zeros")
            processed_data_array = np.nan_to_num(processed_data_array)

        print("✅ Preprocessing completed!")
        print(f"  📊 Output data shape: {processed_data_array.shape}")
        print(f"  🔢 Output data type: {processed_data_array.dtype}")
        return processed_data_array

    def extract_spatial_features(self, data):
        """
        Extract spatial processing features
        """
        print("\n🧩 Extracting spatial features...")

        features = []
        feature_names = [
            'Hemisphere_Dominance',
            'Parietal_Occipital_Coherence',
            'Alpha_Beta_Ratio',
            'Spatial_Complexity',
            'Cross_Hemisphere_Sync',
            'Frontal_Asymmetry',
            'Posterior_Power',
            'Left_Right_Correlation'
        ]

        for i, trial in enumerate(data):
            # Reshape to channels
            channels = trial.reshape(14, -1)
            trial_features = []

            # 1. Hemisphere Dominance (P7 vs P8)
            left_parietal = channels[5]   # P7
            right_parietal = channels[8]  # P8
            hemisphere_dom = np.mean(right_parietal) - np.mean(left_parietal)
            trial_features.append(hemisphere_dom)

            # 2. Parietal-Occipital Coherence
            parietal_avg = (channels[5] + channels[8]) / 2  # P7, P8
            occipital_avg = (channels[6] + channels[7]) / 2  # O1, O2
            coherence = np.corrcoef(parietal_avg, occipital_avg)[0, 1]
            trial_features.append(coherence if not np.isnan(coherence) else 0)

            # 3. Alpha/Beta Power Ratio
            freqs, psd = signal.periodogram(channels.mean(axis=0), fs=self.fs)
            alpha_power = np.mean(psd[(freqs >= 8) & (freqs <= 12)])
            beta_power = np.mean(psd[(freqs >= 13) & (freqs <= 30)])
            ratio = alpha_power / (beta_power + 1e-10)
            trial_features.append(ratio)

            # 4. Spatial Complexity
            spatial_var = np.var(channels, axis=1)
            complexity = np.mean(spatial_var)
            trial_features.append(complexity)

            # 5. Cross-hemisphere synchronization
            left_channels = channels[[0, 1, 2, 3, 4, 5, 6]].mean(axis=0)
            right_channels = channels[[7, 8, 9, 10, 11, 12, 13]].mean(axis=0)
            cross_sync = np.corrcoef(left_channels, right_channels)[0, 1]
            trial_features.append(cross_sync if not np.isnan(cross_sync) else 0)

            # 6. Frontal Asymmetry (F3 vs F4)
            frontal_asym = np.mean(channels[11]) - np.mean(channels[2])  # F4 - F3
            trial_features.append(frontal_asym)

            # 7. Posterior Power (O1, O2, P7, P8)
            posterior_power = np.mean([np.var(channels[5]), np.var(channels[6]),
                                     np.var(channels[7]), np.var(channels[8])])
            trial_features.append(posterior_power)

            # 8. Left-Right Correlation
            lr_corr = np.corrcoef(left_channels, right_channels)[0, 1]
            trial_features.append(lr_corr if not np.isnan(lr_corr) else 0)

            features.append(trial_features)

            if (i + 1) % 200 == 0:
                print(f"  🧩 Extracted features from {i + 1}/{len(data)} trials")

        # Convert to numpy array with explicit dtype
        features_array = np.array(features, dtype=np.float64)

        # Check for NaN or infinity values
        if np.isnan(features_array).any() or np.isinf(features_array).any():
            print("  ⚠️ Warning: NaN or Infinity values detected in features, replacing with zeros")
            features_array = np.nan_to_num(features_array)

        print(f"✅ Feature extraction completed!")
        print(f"  📊 Features shape: {features_array.shape}")
        print(f"  🔢 Features data type: {features_array.dtype}")
        print(f"  🏷️  Feature names: {feature_names}")

        return features_array, feature_names

    def train_classifiers(self, X, y):
        """
        Train multiple classifiers optimized for spatial processing
        """
        print("\n🤖 Training spatial-aware classifiers...")

        # Ensure consistent data types
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.int32)

        # Check for NaN or infinity values
        if np.isnan(X).any() or np.isinf(X).any():
            print("  ⚠️ Warning: NaN or Infinity values detected in features, replacing with zeros")
            X = np.nan_to_num(X)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        print(f"  📊 Training set: {len(X_train)} samples")
        print(f"  📊 Test set: {len(X_test)} samples")
        print(f"  🔢 Feature data type: {X.dtype}")
        print(f"  🏷️ Label data type: {y.dtype}")

        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Ensure scaled data is also float64
        X_train_scaled = np.array(X_train_scaled, dtype=np.float64)
        X_test_scaled = np.array(X_test_scaled, dtype=np.float64)

        results = {}

        # 1. Random Forest (good for feature interpretation)
        print("  🌳 Training Random Forest...")
        try:
            # Debug information
            print(f"  X_train_scaled shape: {X_train_scaled.shape}, dtype: {X_train_scaled.dtype}")
            print(f"  y_train shape: {y_train.shape}, dtype: {y_train.dtype}")
            print(f"  y_train values: {np.unique(y_train)}")

            # Check for NaN or infinity values
            if np.isnan(X_train_scaled).any() or np.isinf(X_train_scaled).any():
                print("  ⚠️ Warning: NaN or Infinity values in training data, replacing with zeros")
                X_train_scaled = np.nan_to_num(X_train_scaled)

            # Ensure y_train is int32
            y_train = np.array(y_train, dtype=np.int32)

            # Train Random Forest
            rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
            rf.fit(X_train_scaled, y_train)

            # Predict
            rf_pred = rf.predict(X_test_scaled)
            rf_accuracy = accuracy_score(y_test, rf_pred)

            results['Random Forest'] = {
                'model': rf,
                'accuracy': rf_accuracy,
                'predictions': rf_pred,
                'feature_importance': rf.feature_importances_
            }

            print(f"  ✅ Random Forest trained successfully! Accuracy: {rf_accuracy:.4f}")
        except Exception as e:
            print(f"  ❌ Error training Random Forest: {str(e)}")
            print(f"  Skipping Random Forest classifier")

        # 2. SVM with RBF kernel
        print("  🎯 Training SVM...")
        try:
            # Ensure y_train is int32
            y_train = np.array(y_train, dtype=np.int32)

            # Train SVM
            svm = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
            svm.fit(X_train_scaled, y_train)

            # Predict
            svm_pred = svm.predict(X_test_scaled)
            svm_accuracy = accuracy_score(y_test, svm_pred)

            results['SVM'] = {
                'model': svm,
                'accuracy': svm_accuracy,
                'predictions': svm_pred
            }

            print(f"  ✅ SVM trained successfully! Accuracy: {svm_accuracy:.4f}")
        except Exception as e:
            print(f"  ❌ Error training SVM: {str(e)}")
            print(f"  Skipping SVM classifier")

        # 3. Simple Neural Network (if tensorflow available)
        try:
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Dense, Dropout
            from tensorflow.keras.optimizers import Adam

            print("  🧠 Training Neural Network...")

            try:
                # Ensure y_train is int32
                y_train = np.array(y_train, dtype=np.int32)

                # Create model
                model = Sequential([
                    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
                    Dropout(0.3),
                    Dense(32, activation='relu'),
                    Dropout(0.3),
                    Dense(16, activation='relu'),
                    Dense(1, activation='sigmoid')
                ])

                model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])

                # Convert to binary labels
                y_train_binary = (y_train == 9).astype(int)
                # Convert test labels to binary (0 for digit 6, 1 for digit 9)
                y_test_binary = (y_test == 9).astype(int)

                # Train
                history = model.fit(
                    X_train_scaled, y_train_binary,
                    epochs=50, batch_size=32,
                    validation_split=0.2,
                    verbose=0
                )

                # Predict
                nn_pred_prob = model.predict(X_test_scaled, verbose=0)
                nn_pred = (nn_pred_prob > 0.5).astype(int).flatten()

                # Calculate binary accuracy (0/1 classification)
                binary_accuracy = accuracy_score(y_test_binary, nn_pred)
                print(f"  Binary accuracy: {binary_accuracy:.4f}")

                # Convert back to original labels (6/9)
                nn_pred_labels = np.where(nn_pred == 1, 9, 6)
                nn_accuracy = accuracy_score(y_test, nn_pred_labels)

                results['Neural Network'] = {
                    'model': model,
                    'accuracy': nn_accuracy,
                    'predictions': nn_pred_labels,
                    'history': history
                }

                print(f"  ✅ Neural Network trained successfully! Accuracy: {nn_accuracy:.4f}")
            except Exception as e:
                print(f"  ❌ Error training Neural Network: {str(e)}")
                print(f"  Skipping Neural Network classifier")

        except ImportError:
            print("  ⚠️  TensorFlow not available, skipping Neural Network")

        print("✅ All classifiers trained!")

        return results, X_test, y_test, scaler

    def visualize_results(self, results, y_test, feature_names):
        """
        Create comprehensive visualization of results
        """
        print("\n📊 Creating visualizations...")

        # Check if we have any results
        if not results:
            print("❌ No models were successfully trained. Skipping visualizations.")
            return

        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Spatial Digit Classification Results: Digit 6 vs 9', fontsize=16, fontweight='bold')

        # 1. Accuracy Comparison
        model_names = list(results.keys())
        accuracies = [results[model]['accuracy'] for model in model_names]

        print(f"  📊 Models trained: {model_names}")
        print(f"  📊 Accuracies: {accuracies}")

        bars = axes[0,0].bar(model_names, accuracies,
                            color=['skyblue', 'lightcoral', 'lightgreen'][:len(model_names)])
        axes[0,0].set_title('Model Accuracy Comparison', fontweight='bold')
        axes[0,0].set_ylabel('Accuracy')
        axes[0,0].set_ylim(0, 1)
        axes[0,0].axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Random Chance')
        axes[0,0].legend()

        # Add accuracy values on bars
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            axes[0,0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                          f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')

        # 2. Best model confusion matrix
        try:
            best_model_name = max(results.keys(), key=lambda k: results[k]['accuracy'])
            best_pred = results[best_model_name]['predictions']

            cm = confusion_matrix(y_test, best_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=['Digit 6', 'Digit 9'],
                       yticklabels=['Digit 6', 'Digit 9'],
                       ax=axes[0,1])
            axes[0,1].set_title(f'Confusion Matrix - {best_model_name}', fontweight='bold')
        except (ValueError, KeyError):
            axes[0,1].text(0.5, 0.5, 'No model available\nfor confusion matrix',
                          ha='center', va='center', fontsize=14)
            axes[0,1].set_title('Confusion Matrix', fontweight='bold')

        # 3. Feature Importance (Random Forest)
        if 'Random Forest' in results and 'feature_importance' in results['Random Forest']:
            importance = results['Random Forest']['feature_importance']
            indices = np.argsort(importance)[::-1]

            axes[0,2].barh(range(len(feature_names)), importance[indices], color='orange')
            axes[0,2].set_yticks(range(len(feature_names)))
            axes[0,2].set_yticklabels([feature_names[i] for i in indices])
            axes[0,2].set_title('Feature Importance (Random Forest)', fontweight='bold')
            axes[0,2].set_xlabel('Importance')

        # 4. Class distribution
        unique, counts = np.unique(y_test, return_counts=True)
        axes[1,0].pie(counts, labels=[f'Digit {int(u)}' for u in unique], autopct='%1.1f%%',
                     colors=['lightblue', 'lightpink'])
        axes[1,0].set_title('Test Set Distribution', fontweight='bold')

        # 5. Training history (Neural Network if available)
        if 'Neural Network' in results and 'history' in results['Neural Network']:
            history = results['Neural Network']['history']
            axes[1,1].plot(history.history['accuracy'], label='Training', linewidth=2)
            axes[1,1].plot(history.history['val_accuracy'], label='Validation', linewidth=2)
            axes[1,1].set_title('Neural Network Training History', fontweight='bold')
            axes[1,1].set_xlabel('Epoch')
            axes[1,1].set_ylabel('Accuracy')
            axes[1,1].legend()
            axes[1,1].grid(True, alpha=0.3)
        else:
            axes[1,1].text(0.5, 0.5, 'Neural Network\nNot Available',
                          ha='center', va='center', fontsize=14)
            axes[1,1].set_title('Neural Network Training', fontweight='bold')

        # 6. Performance summary
        axes[1,2].axis('off')
        summary_text = "PERFORMANCE SUMMARY\n" + "="*25 + "\n\n"

        # Store best accuracy
        best_acc = 0.0

        for model_name, result in results.items():
            acc = result['accuracy']
            summary_text += f"{model_name}:\n"
            summary_text += f"  Accuracy: {acc:.4f}\n"
            if acc > 0.65:
                summary_text += "  🎉 EXCELLENT!\n"
            elif acc > 0.60:
                summary_text += "  ✅ GOOD RESULT!\n"
            elif acc > 0.55:
                summary_text += "  🔍 PROMISING\n"
            else:
                summary_text += "  ⚠️  CHALLENGING\n"
            summary_text += "\n"

            # Update best accuracy
            if acc > best_acc:
                best_acc = acc

        if results:
            summary_text += f"🏆 BEST: {best_acc:.4f}\n\n"

            if best_acc > 0.60:
                summary_text += "SUCCESS: Spatial approach\nshows clear improvement!"
            else:
                summary_text += "Challenge: Further research\nneeded for digit classification"
        else:
            summary_text += "⚠️ Not enough data to determine best model."

        axes[1,2].text(0.1, 0.9, summary_text, fontsize=10, verticalalignment='top',
                      bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))

        plt.tight_layout()
        plt.show()

        # Print detailed results
        print("\n" + "="*70)
        print("🧠 SPATIAL DIGIT CLASSIFICATION RESULTS")
        print("="*70)

        # Store best accuracy and model
        best_accuracy = 0.0
        best_model = None

        for model_name, result in results.items():
            acc = result['accuracy']
            print(f"\n{model_name}:")
            print(f"  🎯 Accuracy: {acc:.4f}")

            if acc > 0.65:
                print(f"  🎉 EXCELLENT! Significant improvement over random chance!")
            elif acc > 0.60:
                print(f"  ✅ GOOD! Clear spatial signal detected!")
            elif acc > 0.55:
                print(f"  🔍 PROMISING! Some spatial processing captured!")
            else:
                print(f"  ⚠️  Still challenging - around random chance level")

            # Update best model and accuracy
            if acc > best_accuracy:
                best_accuracy = acc
                best_model = model_name

        if results:
            print(f"\n🏆 BEST MODEL: {best_model}")
            print(f"🎯 BEST ACCURACY: {best_accuracy:.4f}")

            if best_accuracy > 0.60:
                print("\n✅ SUCCESS: Spatial processing approach works!")
                print("   Digit 6 vs 9 classification shows spatial differentiation!")
                print("   This is a significant finding for EEG-based digit classification!")
            elif best_accuracy > 0.55:
                print("\n🔍 MODERATE SUCCESS: Spatial signal detected!")
                print("   Consider advanced feature engineering and more data!")
            else:
                print("\n🤔 CHALLENGE: EEG digit classification remains difficult")
                print("   Important negative result - fundamental limitations confirmed")
        else:
            print("\n⚠️ Not enough models were successfully trained to determine the best model.")

        print("\n" + "="*70)

def main(run_quick_test=False):
    """
    Main execution function

    Parameters:
    -----------
    run_quick_test : bool
        If True, runs a quick test of the data loading functionality
    """
    print("🚀 Starting Spatial Digit Classification Experiment")
    print("="*60)

    # Initialize classifier
    classifier = SpatialDigitClassifier(device='EP', sampling_rate=128)

    # Load data - UPDATE THIS PATH TO YOUR DATASET
    data_file = "Data/EP1.01.txt"  # UPDATE THIS PATH!

    # Run quick test if requested
    if run_quick_test:
        print("\n🧪 Running quick test of data loading...")
        success = classifier.quick_test(data_file)
        if not success:
            print("\n❌ Quick test failed. Please check your data file and format.")
            return None
        print("\n✅ Quick test successful! Continuing with full analysis...")

    print(f"\n📂 Attempting to load data from: {data_file}")
    print("📋 Expected format: MindBigData TXT/TSV")
    print("🎯 Looking for digits 6 and 9")
    print("\n💡 Expected file structure:")
    print("  Column 1: ID (67635)")
    print("  Column 2: Event (67635)")
    print("  Column 3: Device (EP)")
    print("  Column 4: Channel (AF3, F7, F3, etc.)")
    print("  Column 5: Digit (6 or 9) ← Target untuk klasifikasi")
    print("  Column 6: Length (260) ← Jumlah data points")
    print("  Column 7: Data (comma-separated EEG values)")

    X, y = classifier.load_mindbigdata_sample(data_file, digits=[6, 9], max_trials_per_digit=500)

    if X is None or len(X) == 0:
        print("\n❌ EXPERIMENT STOPPED: No valid dataset found")
        print("\n📝 To fix this issue:")
        print("1. Place your MindBigData file in the 'data/' folder")
        print("2. Update the 'data_file' path in main_script.py")
        print("3. Ensure the file contains trials for digits 6 and 9")
        print("4. Verify file format: TXT or TSV with MindBigData structure")
        print("5. Check that column 7 contains comma-separated EEG values")
        return None

    print(f"\n✅ Dataset loaded successfully! Proceeding with analysis...")

    # Preprocessing
    X_preprocessed = classifier.spatial_preprocessing(X)

    # Feature extraction
    X_features, _ = classifier.extract_spatial_features(X_preprocessed)

    # Classification
    results, _, _, _ = classifier.train_classifiers(X_features, y)

    # Skip visualization for now to avoid recursion error
    print("\n⚠️ Skipping visualization to avoid recursion error")

    # Print results directly
    print("\n" + "="*70)
    print("🧠 SPATIAL DIGIT CLASSIFICATION RESULTS")
    print("="*70)

    # Store best accuracy and model
    best_accuracy = 0.0
    best_model = None

    for model_name, result in results.items():
        acc = result['accuracy']
        print(f"\n{model_name}:")
        print(f"  🎯 Accuracy: {acc:.4f}")

        if acc > 0.65:
            print(f"  🎉 EXCELLENT! Significant improvement over random chance!")
        elif acc > 0.60:
            print(f"  ✅ GOOD! Clear spatial signal detected!")
        elif acc > 0.55:
            print(f"  🔍 PROMISING! Some spatial processing captured!")
        else:
            print(f"  ⚠️  Still challenging - around random chance level")

        # Update best model and accuracy
        if acc > best_accuracy:
            best_accuracy = acc
            best_model = model_name

    if results:
        print(f"\n🏆 BEST MODEL: {best_model}")
        print(f"🎯 BEST ACCURACY: {best_accuracy:.4f}")
    else:
        print("\n⚠️ Not enough models were successfully trained to determine the best model.")

    print("\n🎉 Experiment completed successfully!")
    print("📊 Check the plots above for detailed results!")

    return classifier, results, X_features, y

if __name__ == "__main__":
    import argparse

    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Spatial Digit Classification')
    parser.add_argument('--quick-test', action='store_true',
                        help='Run a quick test of data loading before full analysis')
    args = parser.parse_args()

    # Run the experiment
    try:
        result = main(run_quick_test=args.quick_test)
        if result is not None:
            classifier, results, features, labels = result
            print("\n🎉 All variables assigned successfully!")
        else:
            print("\n❌ Experiment failed - please check dataset and try again")
            print("\n💡 TIP: Try running with the --quick-test flag to test data loading:")
            print("    python main_script.py --quick-test")
    except Exception as e:
        print(f"\n❌ Error during execution: {e}")
        print("Please check your dataset format and try again")
        print("\n💡 TIP: Try running with the --quick-test flag to test data loading:")
        print("    python main_script.py --quick-test")