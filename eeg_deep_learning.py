#!/usr/bin/env python3
# eeg_deep_learning.py - Deep Learning model for EEG digit classification

import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D, Input
from tensorflow.keras.layers import BatchNormalization, Activation, AveragePooling1D
from tensorflow.keras.layers import Reshape, Concatenate, LSTM, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

# Suppress TensorFlow warnings
tf.get_logger().setLevel('ERROR')

def load_digits_simple(file_path, target_digits=[6, 9], max_per_digit=500):
    """
    Load MindBigData format (TXT/TSV) based on specific column structure:
    Column 1: ID (67635)
    Column 2: Event (67635)
    Column 3: Device (EP)
    Column 4: Channel (AF3, F7, F3, etc.)
    Column 5: Digit (6 or 9) ← Target untuk klasifikasi
    Column 6: Length (260) ← Jumlah data points
    Column 7: Data (comma-separated EEG values)
    """
    print(f"📂 Loading data for digits {target_digits}...")
    
    if file_path is None or not os.path.exists(file_path):
        print("❌ Dataset file not found!")
        return None, None
    
    print(f"📖 Reading file: {file_path}")
    
    # Initialize data containers
    data_6 = []
    data_9 = []
    
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
                
                # Only process if it's in target digits
                if digit in target_digits:
                    # Column 7 (index 6) = data
                    data_string = parts[6]
                    
                    # Parse comma-separated values
                    values = [np.float64(x.strip()) for x in data_string.split(',') if x.strip()]
                    
                    # Store based on digit
                    if digit == 6 and len(data_6) < max_per_digit:
                        data_6.append(values)
                    elif digit == 9 and len(data_9) < max_per_digit:
                        data_9.append(values)
                    
                    # Progress
                    if (len(data_6) + len(data_9)) % 100 == 0:
                        print(f"  Found: {len(data_6)} digit-6, {len(data_9)} digit-9")
                    
                    # Stop when we have enough
                    if len(data_6) >= max_per_digit and len(data_9) >= max_per_digit:
                        break
                        
            except (ValueError, IndexError):
                continue
    
    print(f"✅ Final count: {len(data_6)} digit-6, {len(data_9)} digit-9")
    
    if len(data_6) == 0 or len(data_9) == 0:
        print("❌ Missing data for one or both digits!")
        return None, None
    
    # Combine data and labels
    all_data = data_6 + data_9
    all_labels = [0] * len(data_6) + [1] * len(data_9)  # 0 for digit 6, 1 for digit 9
    
    # Normalize lengths (simple padding/truncating)
    normalized_data = []
    target_length = 1792  # 14 channels * 128 timepoints
    
    for trial in all_data:
        if len(trial) >= target_length:
            # Truncate if too long
            normalized_data.append(trial[:target_length])
        else:
            # Pad with repetition if too short
            trial_copy = trial.copy()
            while len(trial_copy) < target_length:
                trial_copy.extend(trial[:min(len(trial), target_length - len(trial_copy))])
            normalized_data.append(trial_copy[:target_length])
    
    data = np.array(normalized_data, dtype=np.float64)
    labels = np.array(all_labels, dtype=np.int32)
    
    # Check for NaN or infinity values
    if np.isnan(data).any() or np.isinf(data).any():
        print("  ⚠️ Warning: NaN or Infinity values detected in data, replacing with zeros")
        data = np.nan_to_num(data)
    
    print(f"  📊 Data shape: {data.shape}")
    
    return data, labels

def preprocess_data_for_cnn(data, labels, test_size=0.2, val_size=0.2):
    """Preprocess data for CNN model"""
    print("\n🔄 Preprocessing data for CNN...")
    
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
    
    # Convert to numpy array
    X = np.array(reshaped_data)
    y = labels
    
    # Split into train, validation, and test sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size/(1-test_size), 
        random_state=42, stratify=y_train_val
    )
    
    # Standardize each channel separately
    for i in range(X_train.shape[0]):
        for j in range(X_train.shape[1]):  # For each channel
            scaler = StandardScaler()
            X_train[i, j, :] = scaler.fit_transform(X_train[i, j, :].reshape(-1, 1)).flatten()
            
            # Use the same scaler for validation and test data
            if i < X_val.shape[0] and j < X_val.shape[1]:
                X_val[i, j, :] = scaler.transform(X_val[i, j, :].reshape(-1, 1)).flatten()
            if i < X_test.shape[0] and j < X_test.shape[1]:
                X_test[i, j, :] = scaler.transform(X_test[i, j, :].reshape(-1, 1)).flatten()
    
    # Convert labels to categorical
    y_train_cat = to_categorical(y_train, num_classes=2)
    y_val_cat = to_categorical(y_val, num_classes=2)
    y_test_cat = to_categorical(y_test, num_classes=2)
    
    print(f"  📊 Training set: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"  📊 Validation set: X_val={X_val.shape}, y_val={y_val.shape}")
    print(f"  📊 Test set: X_test={X_test.shape}, y_test={y_test.shape}")
    
    return X_train, X_val, X_test, y_train_cat, y_val_cat, y_test_cat, y_test

def build_eegnet_model(input_shape):
    """Build EEGNet model - a specialized CNN for EEG data"""
    print("\n🏗️ Building EEGNet model...")
    
    # Model parameters
    nb_classes = 2  # Binary classification (digit 6 vs 9)
    F1 = 8  # Number of temporal filters
    F2 = 16  # Number of pointwise filters
    D = 2  # Depth multiplier
    
    # Input layer
    input_layer = Input(shape=input_shape)
    
    # Reshape to (channels, samples, 1)
    reshaped = Reshape((input_shape[0], input_shape[1], 1))(input_layer)
    
    # Block 1: Temporal Convolution
    block1 = Conv1D(F1, kernel_size=64, padding='same', 
                   input_shape=(input_shape[0], input_shape[1], 1))(reshaped)
    block1 = BatchNormalization()(block1)
    
    # Block 2: Spatial Convolution
    block2 = Conv1D(F2, kernel_size=32, padding='same')(block1)
    block2 = BatchNormalization()(block2)
    block2 = Activation('elu')(block2)
    block2 = MaxPooling1D(pool_size=4)(block2)
    block2 = Dropout(0.5)(block2)
    
    # Block 3: Separable Convolution
    block3 = Conv1D(F2, kernel_size=16, padding='same')(block2)
    block3 = BatchNormalization()(block3)
    block3 = Activation('elu')(block3)
    block3 = MaxPooling1D(pool_size=4)(block3)
    block3 = Dropout(0.5)(block3)
    
    # Classification
    flatten = Flatten()(block3)
    dense = Dense(nb_classes, activation='softmax')(flatten)
    
    model = Model(inputs=input_layer, outputs=dense)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001),
                 metrics=['accuracy'])
    
    print(model.summary())
    
    return model

def train_and_evaluate_model(model, X_train, X_val, X_test, y_train, y_val, y_test, y_test_orig):
    """Train and evaluate the model"""
    print("\n🚀 Training model...")
    
    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001, verbose=1)
    
    # Train model
    history = model.fit(
        X_train, y_train,
        batch_size=32,
        epochs=50,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    # Evaluate model
    print("\n📊 Evaluating model...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"  ✅ Test accuracy: {test_acc:.4f}")
    
    # Predictions
    y_pred_prob = model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)
    
    # Classification report
    print("\n📊 Classification Report:")
    print(classification_report(y_test_orig, y_pred, target_names=['Digit 6', 'Digit 9']))
    
    # Confusion matrix
    cm = confusion_matrix(y_test_orig, y_pred)
    print(f"  Confusion Matrix:")
    print(f"  {cm[0][0]:4d} {cm[0][1]:4d} | Digit 6")
    print(f"  {cm[1][0]:4d} {cm[1][1]:4d} | Digit 9")
    print(f"    6    9   <- Predicted")
    
    # Calculate sensitivity and specificity
    sensitivity = cm[0][0] / (cm[0][0] + cm[0][1])  # True positive rate for digit 6
    specificity = cm[1][1] / (cm[1][0] + cm[1][1])  # True positive rate for digit 9
    
    print(f"  Sensitivity (Digit 6): {sensitivity:.4f}")
    print(f"  Specificity (Digit 9): {specificity:.4f}")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('eeg_cnn_training_history.png')
    print("\n📊 Training history plot saved as 'eeg_cnn_training_history.png'")
    
    return model, history

def main():
    """Main function"""
    print("🚀 Deep Learning for EEG Digit Classification")
    print("=" * 50)
    
    # Load data
    file_path = "Data/EP1.01.txt"
    data, labels = load_digits_simple(file_path, max_per_digit=500)
    
    if data is None:
        print("❌ Failed to load data")
        return
    
    # Preprocess data for CNN
    X_train, X_val, X_test, y_train, y_val, y_test, y_test_orig = preprocess_data_for_cnn(data, labels)
    
    # Build model
    model = build_eegnet_model(input_shape=(X_train.shape[1], X_train.shape[2]))
    
    # Train and evaluate model
    model, history = train_and_evaluate_model(
        model, X_train, X_val, X_test, y_train, y_val, y_test, y_test_orig
    )
    
    # Save model
    model.save('eeg_cnn_model.h5')
    print("\n💾 Model saved as 'eeg_cnn_model.h5'")
    
    print("\n✅ Analysis completed!")

if __name__ == "__main__":
    main()
