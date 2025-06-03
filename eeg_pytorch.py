#!/usr/bin/env python3
# eeg_pytorch.py - Deep Learning model for EEG digit classification using PyTorch

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

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train).unsqueeze(1)  # Add channel dimension
    X_val_tensor = torch.FloatTensor(X_val).unsqueeze(1)
    X_test_tensor = torch.FloatTensor(X_test).unsqueeze(1)
    
    y_train_tensor = torch.LongTensor(y_train)
    y_val_tensor = torch.LongTensor(y_val)
    y_test_tensor = torch.LongTensor(y_test)
    
    print(f"  📊 Training set: X_train={X_train_tensor.shape}, y_train={y_train_tensor.shape}")
    print(f"  📊 Validation set: X_val={X_val_tensor.shape}, y_val={y_val_tensor.shape}")
    print(f"  📊 Test set: X_test={X_test_tensor.shape}, y_test={y_test_tensor.shape}")
    
    # Create DataLoaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    return train_loader, val_loader, test_loader, y_test

class EEGNet(nn.Module):
    """EEGNet model for EEG classification"""
    def __init__(self):
        super(EEGNet, self).__init__()
        
        # Layer 1: Temporal Convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, 64), padding=(0, 32)),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.Dropout(0.25)
        )
        
        # Layer 2: Spatial Convolution
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(14, 1)),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Dropout(0.25),
            nn.AvgPool2d(kernel_size=(1, 4))
        )
        
        # Layer 3: Separable Convolution
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 16), padding=(0, 8)),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Dropout(0.25),
            nn.AvgPool2d(kernel_size=(1, 8))
        )
        
        # Fully Connected Layer
        self.fc = nn.Linear(32 * 1 * 4, 2)  # 2 classes (digit 6 vs 9)
    
    def forward(self, x):
        # Layer 1
        x = self.conv1(x)
        
        # Layer 2
        x = self.conv2(x)
        
        # Layer 3
        x = self.conv3(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully Connected Layer
        x = self.fc(x)
        
        return x

def train_model(model, train_loader, val_loader, num_epochs=50):
    """Train the model"""
    print("\n🚀 Training model...")
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # Training loop
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            train_loss += loss.item() * inputs.size(0)
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
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Statistics
                val_loss += loss.item() * inputs.size(0)
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
    plt.savefig('eeg_pytorch_training_history.png')
    print("\n📊 Training history plot saved as 'eeg_pytorch_training_history.png'")
    
    # Create history dictionary
    history = {
        'train_loss': train_losses,
        'val_loss': val_losses,
        'train_acc': train_accs,
        'val_acc': val_accs
    }

    return model, history

def evaluate_model(model, test_loader, y_test):
    """Evaluate the model"""
    print("\n📊 Evaluating model...")
    
    model.eval()
    all_preds = []
    
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            
            # Forward pass
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
    
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
    
    return accuracy, all_preds

def main():
    """Main function"""
    print("🚀 Deep Learning (PyTorch) for EEG Digit Classification")
    print("=" * 50)
    
    # Load data
    file_path = "Data/EP1.01.txt"
    data, labels = load_digits_simple(file_path, max_per_digit=500)
    
    if data is None:
        print("❌ Failed to load data")
        return
    
    # Preprocess data for CNN
    train_loader, val_loader, test_loader, y_test = preprocess_data_for_cnn(data, labels)
    
    # Build model
    model = EEGNet().to(device)
    print(model)
    
    # Train model
    model, history = train_model(model, train_loader, val_loader, num_epochs=30)
    
    # Evaluate model
    accuracy, predictions = evaluate_model(model, test_loader, y_test)
    
    # Save model
    torch.save(model.state_dict(), 'eeg_pytorch_model.pth')
    print("\n💾 Model saved as 'eeg_pytorch_model.pth'")
    
    print("\n✅ Analysis completed!")

if __name__ == "__main__":
    main()
