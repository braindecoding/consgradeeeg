#!/usr/bin/env python3
# eeg_lstm_wavelet.py - Advanced Bidirectional LSTM model with wavelet features for EEG classification

import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

import pywt  # PyWavelets library for wavelet analysis

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def load_digits_simple(file_path, target_digits=[6, 9], max_per_digit=500):
    """Load EEG data for digit classification"""
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

def extract_wavelet_features(data):
    """Extract wavelet features from EEG data"""
    print("\n🧩 Extracting wavelet features...")
    
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
    
    # Define wavelet parameters
    wavelet = 'db4'  # Daubechies wavelet with 4 vanishing moments
    level = 4        # Decomposition level
    
    # Extract wavelet features
    wavelet_features = []
    
    for trial in reshaped_data:
        trial_features = []
        
        # Process each channel
        for channel in range(trial.shape[0]):
            # Perform wavelet decomposition
            coeffs = pywt.wavedec(trial[channel], wavelet, level=level)
            
            # Extract features from each level
            for i in range(level + 1):
                # Calculate energy
                energy = np.sum(coeffs[i]**2)
                
                # Calculate entropy
                entropy = -np.sum(coeffs[i]**2 * np.log(coeffs[i]**2 + 1e-10))
                
                # Calculate mean
                mean = np.mean(coeffs[i])
                
                # Calculate standard deviation
                std = np.std(coeffs[i])
                
                # Add features
                trial_features.extend([energy, entropy, mean, std])
        
        wavelet_features.append(trial_features)
    
    wavelet_features = np.array(wavelet_features, dtype=np.float64)
    
    # Check for NaN or infinity values
    if np.isnan(wavelet_features).any() or np.isinf(wavelet_features).any():
        print("  ⚠️ Warning: NaN or Infinity values in features, replacing with zeros")
        wavelet_features = np.nan_to_num(wavelet_features)
    
    print(f"  ✅ Wavelet features extracted: {wavelet_features.shape}")
    
    return wavelet_features, reshaped_data

def preprocess_data(data, labels, wavelet_features, test_size=0.2, val_size=0.2):
    """Preprocess data for LSTM model with wavelet features"""
    print("\n🔄 Preprocessing data...")
    
    # Convert to numpy array
    X_raw = np.array(data)
    X_wavelet = np.array(wavelet_features)
    y = labels
    
    # Split into train, validation, and test sets
    X_raw_train_val, X_raw_test, X_wavelet_train_val, X_wavelet_test, y_train_val, y_test = train_test_split(
        X_raw, X_wavelet, y, test_size=test_size, random_state=42, stratify=y
    )
    
    X_raw_train, X_raw_val, X_wavelet_train, X_wavelet_val, y_train, y_val = train_test_split(
        X_raw_train_val, X_wavelet_train_val, y_train_val, 
        test_size=val_size/(1-test_size), random_state=42, stratify=y_train_val
    )
    
    # Standardize raw data
    for i in range(X_raw_train.shape[0]):
        for j in range(X_raw_train.shape[1]):  # For each channel
            scaler = StandardScaler()
            X_raw_train[i, j, :] = scaler.fit_transform(X_raw_train[i, j, :].reshape(-1, 1)).flatten()
            
            # Use the same scaler for validation and test data
            if i < X_raw_val.shape[0] and j < X_raw_val.shape[1]:
                X_raw_val[i, j, :] = scaler.transform(X_raw_val[i, j, :].reshape(-1, 1)).flatten()
            if i < X_raw_test.shape[0] and j < X_raw_test.shape[1]:
                X_raw_test[i, j, :] = scaler.transform(X_raw_test[i, j, :].reshape(-1, 1)).flatten()
    
    # Standardize wavelet features
    scaler_wavelet = StandardScaler()
    X_wavelet_train = scaler_wavelet.fit_transform(X_wavelet_train)
    X_wavelet_val = scaler_wavelet.transform(X_wavelet_val)
    X_wavelet_test = scaler_wavelet.transform(X_wavelet_test)
    
    # Convert to PyTorch tensors
    # For LSTM, we need shape [batch_size, sequence_length, features]
    # Our data is [batch_size, channels, timepoints], so we'll transpose to [batch_size, timepoints, channels]
    X_raw_train_tensor = torch.FloatTensor(X_raw_train.transpose(0, 2, 1))
    X_raw_val_tensor = torch.FloatTensor(X_raw_val.transpose(0, 2, 1))
    X_raw_test_tensor = torch.FloatTensor(X_raw_test.transpose(0, 2, 1))
    
    X_wavelet_train_tensor = torch.FloatTensor(X_wavelet_train)
    X_wavelet_val_tensor = torch.FloatTensor(X_wavelet_val)
    X_wavelet_test_tensor = torch.FloatTensor(X_wavelet_test)
    
    y_train_tensor = torch.LongTensor(y_train)
    y_val_tensor = torch.LongTensor(y_val)
    y_test_tensor = torch.LongTensor(y_test)
    
    print(f"  📊 Training set: X_raw={X_raw_train_tensor.shape}, X_wavelet={X_wavelet_train_tensor.shape}, y={y_train_tensor.shape}")
    print(f"  📊 Validation set: X_raw={X_raw_val_tensor.shape}, X_wavelet={X_wavelet_val_tensor.shape}, y={y_val_tensor.shape}")
    print(f"  📊 Test set: X_raw={X_raw_test_tensor.shape}, X_wavelet={X_wavelet_test_tensor.shape}, y={y_test_tensor.shape}")
    
    # Create DataLoaders
    train_dataset = TensorDataset(X_raw_train_tensor, X_wavelet_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_raw_val_tensor, X_wavelet_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_raw_test_tensor, X_wavelet_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    return train_loader, val_loader, test_loader, y_test

class AttentionLayer(nn.Module):
    """Attention mechanism for LSTM"""
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Linear(hidden_size, 1)
        
    def forward(self, lstm_output):
        # lstm_output shape: [batch_size, seq_len, hidden_size]
        attention_weights = F.softmax(self.attention(lstm_output), dim=1)
        # attention_weights shape: [batch_size, seq_len, 1]
        
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)
        # context_vector shape: [batch_size, hidden_size]
        
        return context_vector, attention_weights

class BidirectionalLSTMWithWavelet(nn.Module):
    """Bidirectional LSTM with attention and wavelet features"""
    def __init__(self, input_size, hidden_size, wavelet_features_dim, num_layers=2, dropout_rate=0.5):
        super(BidirectionalLSTMWithWavelet, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        
        # Bidirectional LSTM for raw EEG data
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention mechanism
        self.attention = AttentionLayer(hidden_size * 2)  # *2 for bidirectional
        
        # Fully connected layers for wavelet features
        self.fc_wavelet1 = nn.Linear(wavelet_features_dim, hidden_size)
        self.batch_norm_wavelet = nn.BatchNorm1d(hidden_size)
        
        # Combined fully connected layers
        self.fc_combined = nn.Linear(hidden_size * 2 + hidden_size, hidden_size)  # LSTM + wavelet
        self.fc_out = nn.Linear(hidden_size, 2)  # 2 classes
    
    def forward(self, x_raw, x_wavelet):
        # Process raw EEG data with LSTM
        # x_raw shape: [batch_size, seq_len, input_size]
        lstm_output, _ = self.lstm(x_raw)
        # lstm_output shape: [batch_size, seq_len, hidden_size*2]
        
        # Apply attention
        context_vector, _ = self.attention(lstm_output)
        # context_vector shape: [batch_size, hidden_size*2]
        
        # Process wavelet features
        x_wavelet = self.fc_wavelet1(x_wavelet)
        x_wavelet = self.batch_norm_wavelet(x_wavelet)
        x_wavelet = F.elu(x_wavelet)
        x_wavelet = F.dropout(x_wavelet, self.dropout_rate, training=self.training)
        
        # Combine features
        x_combined = torch.cat((context_vector, x_wavelet), dim=1)
        
        # Final classification
        x_combined = self.fc_combined(x_combined)
        x_combined = F.elu(x_combined)
        x_combined = F.dropout(x_combined, self.dropout_rate, training=self.training)
        x_combined = self.fc_out(x_combined)
        
        return x_combined

def train_model(model, train_loader, val_loader, num_epochs=50, weight_decay=1e-4):
    """Train the model with regularization and early stopping"""
    print("\n🚀 Training model...")
    
    # Loss function and optimizer with weight decay (L2 regularization)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # Training loop
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    best_val_loss = float('inf')
    best_model_state = None
    patience = 15
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs_raw, inputs_wavelet, labels in train_loader:
            inputs_raw, inputs_wavelet, labels = inputs_raw.to(device), inputs_wavelet.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs_raw, inputs_wavelet)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Statistics
            train_loss += loss.item() * inputs_raw.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = train_correct / train_total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs_raw, inputs_wavelet, labels in val_loader:
                inputs_raw, inputs_wavelet, labels = inputs_raw.to(device), inputs_wavelet.to(device), labels.to(device)
                
                # Forward pass
                outputs = model(inputs_raw, inputs_wavelet)
                loss = criterion(outputs, labels)
                
                # Statistics
                val_loss += loss.item() * inputs_raw.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_correct / val_total
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f"  ⚠️ Early stopping at epoch {epoch+1}")
            break
        
        # Print statistics
        print(f"Epoch {epoch+1}/{num_epochs}: "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(train_accs, label='Train')
    plt.plot(val_accs, label='Validation')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Validation')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('eeg_lstm_wavelet_training_history.png')
    print("\n📊 Training history plot saved as 'eeg_lstm_wavelet_training_history.png'")
    
    return model

def evaluate_model(model, test_loader, y_test):
    """Evaluate the model"""
    print("\n📊 Evaluating model...")
    
    model.eval()
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for inputs_raw, inputs_wavelet, _ in test_loader:
            inputs_raw, inputs_wavelet = inputs_raw.to(device), inputs_wavelet.to(device)
            
            # Forward pass
            outputs = model(inputs_raw, inputs_wavelet)
            probs = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, all_preds)
    print(f"  ✅ Test accuracy: {accuracy:.4f}")
    
    # Classification report
    print("\n📊 Classification Report:")
    print(classification_report(y_test, all_preds, target_names=['Digit 6', 'Digit 9']))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, all_preds)
    print(f"  Confusion Matrix:")
    print(f"  {cm[0][0]:4d} {cm[0][1]:4d} | Digit 6")
    print(f"  {cm[1][0]:4d} {cm[1][1]:4d} | Digit 9")
    print(f"    6    9   <- Predicted")
    
    # Calculate sensitivity and specificity
    sensitivity = cm[0][0] / (cm[0][0] + cm[0][1])  # True positive rate for digit 6
    specificity = cm[1][1] / (cm[1][0] + cm[1][1])  # True positive rate for digit 9
    
    print(f"  Sensitivity (Digit 6): {sensitivity:.4f}")
    print(f"  Specificity (Digit 9): {specificity:.4f}")
    
    return accuracy, all_preds, all_probs

def main():
    """Main function"""
    print("🚀 Bidirectional LSTM with Attention and Wavelet Features for EEG Classification")
    print("=" * 50)
    
    # Load data
    file_path = "Data/EP1.01.txt"
    data, labels = load_digits_simple(file_path, max_per_digit=500)
    
    if data is None:
        print("❌ Failed to load data")
        return
    
    # Extract wavelet features
    wavelet_features, reshaped_data = extract_wavelet_features(data)
    
    # Preprocess data
    train_loader, val_loader, test_loader, y_test = preprocess_data(reshaped_data, labels, wavelet_features)
    
    # Build model
    input_size = 14  # Number of EEG channels
    hidden_size = 64  # LSTM hidden size
    wavelet_features_dim = wavelet_features.shape[1]
    
    model = BidirectionalLSTMWithWavelet(
        input_size=input_size,
        hidden_size=hidden_size,
        wavelet_features_dim=wavelet_features_dim,
        num_layers=2,
        dropout_rate=0.5
    ).to(device)
    
    print(model)
    
    # Train model
    model = train_model(model, train_loader, val_loader, num_epochs=100, weight_decay=1e-4)
    
    # Evaluate model
    accuracy, predictions, probabilities = evaluate_model(model, test_loader, y_test)
    
    # Save model
    torch.save(model.state_dict(), 'eeg_lstm_wavelet_model.pth')
    print("\n💾 Model saved as 'eeg_lstm_wavelet_model.pth'")
    
    print("\n✅ Analysis completed!")

if __name__ == "__main__":
    main()
