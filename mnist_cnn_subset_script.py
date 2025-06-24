#!/usr/bin/env python3
"""
MNIST Digit Recognition using CNN with Keras - Incremental Learning
Achieves 95% accuracy with minimal training samples through incremental learning

This script implements:
- Incremental learning: add 100 cases at a time until 95% accuracy
- Multiple trials with different random subsets
- Best subset selection and saving
- Convolutional Neural Network (CNN) architecture
- Learning rate scheduling
- Early stopping
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from datetime import datetime

from sklearn.model_selection import train_test_split

from tensorflow.keras.utils import to_categorical

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import EarlyStopping

def load_data():
    """
    Load MNIST training and test data from DR CSV files
    """
    print("Loading data from DR CSV files...")
    
    # Load data from the data directory (DR version)
    train = pd.read_csv('data/mnist_train_dr.csv')
    test = pd.read_csv('data/mnist_test_dr.csv')
    df = train.copy()
    df_test = test.copy()
    
    print(f"Training data shape: {df.shape}")
    print(f"Test data shape: {df_test.shape}")
    
    return df, df_test

def check_missing_values(df, df_test):
    """Check for missing values in the datasets"""
    print("\nChecking for missing values...")
    
    missing_train = df.isnull().any().sum()
    missing_test = df_test.isnull().any().sum()
    
    print(f"Missing values in training data: {missing_train}")
    print(f"Missing values in test data: {missing_test}")
    
    return missing_train == 0 and missing_test == 0

def preprocess_data(df, df_test, seed=3141):
    """
    Preprocess the data including:
    - Train/test split
    - Reshaping images (for DR data, we'll reshape to approximate square dimensions)
    - Normalization
    - One-hot encoding
    """
    print("\nPreprocessing data...")
    
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Split training data - handle both 'label' and 'class' column names
    if 'label' in df.columns:
        X = df.iloc[:, 1:]  # All columns except the first (labels)
        Y = df.iloc[:, 0]   # First column (labels)
    elif 'class' in df.columns:
        X = df.iloc[:, 1:]  # All columns except the first (class)
        Y = df.iloc[:, 0]   # First column (class)
    else:
        raise ValueError("Neither 'label' nor 'class' column found in training data")
    
    x_train, x_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.1, random_state=seed
    )
    
    print(f"Training set: {x_train.shape}")
    print(f"Validation set: {x_test.shape}")
    
    # For test data, separate features and labels
    if 'label' in df_test.columns:
        df_test_features = df_test.iloc[:, 1:]  # Features (exclude label column)
        df_test_labels = df_test.iloc[:, 0]     # Labels (first column)
    elif 'class' in df_test.columns:
        df_test_features = df_test.iloc[:, 1:]  # Features (exclude class column)
        df_test_labels = df_test.iloc[:, 0]     # Class (first column)
    else:
        df_test_features = df_test  # No label column to exclude
        df_test_labels = None
    
    print(f"Test data features shape: {df_test_features.shape}")
    if df_test_labels is not None:
        print(f"Test data labels shape: {df_test_labels.shape}")
    
    # Calculate image dimensions for reshaping
    # For DR data, we'll reshape to approximate square dimensions
    n_features = x_train.shape[1]
    img_size = int(np.sqrt(n_features))
    
    # If not a perfect square, pad to the next square
    if img_size * img_size != n_features:
        img_size = int(np.ceil(np.sqrt(n_features)))
        print(f"Features ({n_features}) not a perfect square, using {img_size}x{img_size} = {img_size*img_size}")
    
    # Reshape images to square dimensions
    x_train = x_train.values.reshape(-1, img_size, img_size, 1)
    x_test = x_test.values.reshape(-1, img_size, img_size, 1)
    df_test = df_test_features.values.reshape(-1, img_size, img_size, 1)
    
    print(f"Reshaped training data: {x_train.shape}")
    print(f"Reshaped validation data: {x_test.shape}")
    print(f"Reshaped test data: {df_test.shape}")
    
    # Normalize pixel values to [0, 1]
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    df_test = df_test.astype("float32") / 255
    
    # One-hot encode labels
    y_train = to_categorical(y_train, num_classes=10)
    y_test = to_categorical(y_test, num_classes=10)
    
    print(f"One-hot encoded training labels: {y_train.shape}")
    print(f"One-hot encoded validation labels: {y_test.shape}")
    
    return x_train, x_test, y_train, y_test, df_test, df_test_labels, img_size

def preprocess_incremental_data(df_subset, df_test, seed=3141):
    """
    Preprocess data for incremental learning without splitting training data
    """
    print(f"\nPreprocessing incremental data with {len(df_subset)} samples...")
    
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Handle both 'label' and 'class' column names for training data
    if 'label' in df_subset.columns:
        X_train = df_subset.iloc[:, 1:]  # All columns except the first (labels)
        Y_train = df_subset.iloc[:, 0]   # First column (labels)
    elif 'class' in df_subset.columns:
        X_train = df_subset.iloc[:, 1:]  # All columns except the first (class)
        Y_train = df_subset.iloc[:, 0]   # First column (class)
    else:
        raise ValueError("Neither 'label' nor 'class' column found in training data")
    
    print(f"Training set: {X_train.shape}")
    
    # For test data, separate features and labels
    if 'label' in df_test.columns:
        df_test_features = df_test.iloc[:, 1:]  # Features (exclude label column)
        df_test_labels = df_test.iloc[:, 0]     # Labels (first column)
    elif 'class' in df_test.columns:
        df_test_features = df_test.iloc[:, 1:]  # Features (exclude class column)
        df_test_labels = df_test.iloc[:, 0]     # Class (first column)
    else:
        df_test_features = df_test  # No label column to exclude
        df_test_labels = None
    
    print(f"Test data features shape: {df_test_features.shape}")
    if df_test_labels is not None:
        print(f"Test data labels shape: {df_test_labels.shape}")
    
    # Calculate number of classes dynamically
    num_classes = len(np.unique(Y_train))
    print(f"Number of classes detected: {num_classes}")
    
    # Calculate image dimensions for reshaping
    n_features = X_train.shape[1]
    img_size = int(np.sqrt(n_features))
    
    # If not a perfect square, pad to the next square
    if img_size * img_size != n_features:
        img_size = int(np.ceil(np.sqrt(n_features)))
        print(f"Features ({n_features}) not a perfect square, using {img_size}x{img_size} = {img_size*img_size}")
    
    # Reshape images to square dimensions
    x_train = X_train.values.reshape(-1, img_size, img_size, 1)
    df_test_reshaped = df_test_features.values.reshape(-1, img_size, img_size, 1)
    
    print(f"Reshaped training data: {x_train.shape}")
    print(f"Reshaped test data: {df_test_reshaped.shape}")
    
    # Normalize pixel values to [0, 1]
    x_train = x_train.astype("float32") / 255
    df_test_reshaped = df_test_reshaped.astype("float32") / 255
    
    # One-hot encode labels
    y_train = to_categorical(Y_train, num_classes=num_classes)
    
    print(f"One-hot encoded training labels: {y_train.shape}")
    
    return x_train, y_train, df_test_reshaped, df_test_labels, img_size, num_classes

def build_model(img_size=28, num_classes=10):
    """
    Build the CNN model architecture
    """
    print(f"\nBuilding CNN model for {img_size}x{img_size} images with {num_classes} classes...")
    
    model = Sequential()
    
    # Add Input layer as first layer
    model.add(Input(shape=(img_size, img_size, 1)))
    
    # First Convolutional Block
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', 
                     strides=1, padding='same', data_format='channels_last'))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', 
                     strides=1, padding='same', data_format='channels_last'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid'))
    model.add(Dropout(0.25))
    
    # Second Convolutional Block
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', 
                     strides=1, padding='same', data_format='channels_last'))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=1, 
                     padding='same', activation='relu', data_format='channels_last'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding='valid', strides=2))
    model.add(Dropout(0.25))
    
    # Flatten and Dense Layers
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(Dense(1024, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    
    return model

def compile_model(model):
    """
    Compile the model with optimizer, loss function, and metrics
    """
    print("\nCompiling model...")
    
    # Optimizer
    optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
    
    # Compile the model
    model.compile(optimizer=optimizer, 
                  loss="categorical_crossentropy", 
                  metrics=["accuracy"])
    
    # Print model summary
    model.summary()
    
    return model

def create_callbacks():
    """
    Create callbacks for training with smart learning rate scheduling
    """
    print("\nCreating callbacks...")
    
    # Smart Learning Rate Scheduler with warmup and plateau detection
    def smart_lr_schedule(epoch):
        initial_lr = 0.001
        
        # Warmup phase (first 3 epochs)
        if epoch < 3:
            return initial_lr * (epoch + 1) / 3
        
        # Main training phase with cosine annealing
        warmup_epochs = 3
        total_epochs = 30
        decay_epochs = total_epochs - warmup_epochs
        current_decay_epoch = epoch - warmup_epochs
        
        # Cosine annealing with restarts
        cosine_decay = 0.5 * (1 + np.cos(np.pi * current_decay_epoch / decay_epochs))
        return initial_lr * cosine_decay
    
    # Learning Rate Scheduler
    reduce_lr = LearningRateScheduler(smart_lr_schedule)
    
    # Early Stopping with more patience since we don't have data augmentation
    early_stopping = EarlyStopping(
        min_delta=0.001,  # minimum amount of change to count as an improvement
        patience=15,  # increased patience since no data augmentation
        restore_best_weights=True,
        monitor='val_accuracy'  # monitor validation accuracy
    )
    
    # Visualize learning rate decay
    print("Smart learning rate schedule:")
    for i in range(10):
        lr = smart_lr_schedule(i)
        print(f"Epoch {i+1} Learning Rate: {lr:.6f}")
    
    return [reduce_lr, early_stopping]

def train_model(model, x_train, y_train, x_test, y_test, callbacks, epochs=50):
    """
    Train the model
    """
    print(f"\nTraining model with {len(x_train)} samples...")
    
    # Calculate batch size as percentage of dataset size
    # Start with 20% for small datasets, scale down to 5% for large datasets
    if len(x_train) <= 100:
        batch_percentage = 0.20  # 20% of dataset
    elif len(x_train) <= 1000:
        batch_percentage = 0.15  # 15% of dataset
    elif len(x_train) <= 10000:
        batch_percentage = 0.10  # 10% of dataset
    else:
        batch_percentage = 0.05  # 5% of dataset
    
    batch_size = max(1, int(len(x_train) * batch_percentage))
    
    print(f"Using batch size {batch_size} ({batch_percentage*100:.0f}% of {len(x_train)} samples)")
    
    history = model.fit(
        x_train, y_train,
        epochs=epochs,
        validation_data=(x_test, y_test),
        verbose=1,
        batch_size=batch_size,
        callbacks=callbacks,
        shuffle=True
    )
    
    return history

def evaluate_model(model, x_test, y_test, history, plot=False):
    """
    Evaluate the model and optionally plot training history
    """
    print("\nEvaluating model...")
    
    # Evaluate on test set
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    
    # Plot training history only if requested
    if plot:
        plt.figure(figsize=(13, 5))
        
        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title("Model Loss")
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend(['Train', 'Validation'])
        plt.grid()
        
        # Plot accuracy
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend(['Train', 'Validation'])
        plt.grid()
        
        plt.tight_layout()
        plt.show()
    
    return test_accuracy

def plot_best_trial(best_trial):
    """
    Plot the training history for the best trial only
    """
    if best_trial['best_history'] is not None:
        history = best_trial['best_history']
        
        plt.figure(figsize=(13, 5))
        
        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title(f"Best Trial {best_trial['trial_num']} - Model Loss")
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend(['Train', 'Validation'])
        plt.grid()
        
        # Plot accuracy
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title(f"Best Trial {best_trial['trial_num']} - Model Accuracy")
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend(['Train', 'Validation'])
        plt.grid()
        
        plt.tight_layout()
        plt.show()

def make_predictions(model, df_test, df_test_labels=None):
    """
    Make predictions on test data and save to CSV
    Also calculate accuracy if test labels are provided
    """
    print("\nMaking predictions...")
    
    # Make predictions
    pred_digits_test = np.argmax(model.predict(df_test), axis=1)
    
    # Calculate accuracy if test labels are provided
    if df_test_labels is not None:
        test_accuracy = np.mean(pred_digits_test == df_test_labels)
        print(f"Test set accuracy: {test_accuracy:.4f}")
    
    # Create submission file
    image_id_test = list(range(1, len(pred_digits_test) + 1))
    
    submission = pd.DataFrame({
        'ImageId': image_id_test,
        'Label': pred_digits_test
    })
    
    # Save to CSV
    submission.to_csv('mnist_predictions.csv', index=False)
    print(f"Predictions saved to 'mnist_predictions.csv'")
    print(f"Number of predictions: {len(pred_digits_test)}")
    
    return submission

def save_model(model, filename='mnist_cnn_model.h5'):
    """
    Save the trained model
    """
    print(f"\nSaving model to {filename}...")
    model.save(filename)
    print("Model saved successfully!")

def incremental_learning_trial(df, df_test, trial_num, target_accuracy=0.95, batch_size=100, max_samples=None):
    """
    Perform one incremental learning trial
    Start with 100 samples, add 100 more until target accuracy is reached
    Uses stratified sampling to ensure representative class distribution
    Backtracks when accuracy deteriorates and tries different samples
    """
    print(f"\n{'='*60}")
    print(f"TRIAL {trial_num}")
    print(f"{'='*60}")
    
    # Set unique random seed for this trial
    trial_seed = random.randint(1, 100000)
    np.random.seed(trial_seed)
    random.seed(trial_seed)
    print(f"Trial {trial_num} using seed: {trial_seed}")
    
    # Get total available samples
    total_samples = len(df)
    if max_samples is None:
        max_samples = total_samples
    
    # Get class labels for stratified sampling
    if 'label' in df.columns:
        class_labels = df.iloc[:, 0]  # First column (labels)
    elif 'class' in df.columns:
        class_labels = df.iloc[:, 0]  # First column (class)
    else:
        raise ValueError("Neither 'label' nor 'class' column found in training data")
    
    # Initialize variables
    current_samples = batch_size
    best_accuracy = 0
    best_model = None
    best_history = None
    best_sample_indices = None
    consecutive_deteriorations = 0
    max_deteriorations = 3  # Maximum consecutive deteriorations before giving up
    
    # Use stratified sampling to get representative indices
    from sklearn.model_selection import StratifiedShuffleSplit
    
    while current_samples <= max_samples and best_accuracy < target_accuracy:
        print(f"\n--- Training with {current_samples} samples ---")
        
        # Try different sample combinations if accuracy deteriorated
        attempts = 0
        max_attempts = 5  # Try up to 5 different sample combinations
        
        while attempts < max_attempts:
            # Use stratified sampling to select current subset
            sss = StratifiedShuffleSplit(n_splits=1, test_size=1-current_samples/total_samples, 
                                       random_state=trial_seed + attempts)  # Different seed for each attempt
            
            # Get stratified indices
            for train_idx, _ in sss.split(df, class_labels):
                current_indices = train_idx
                break
            
            # Ensure we get exactly the number of samples we want
            if len(current_indices) > current_samples:
                current_indices = current_indices[:current_samples]
            elif len(current_indices) < current_samples:
                # If we don't have enough, add more random samples
                remaining_indices = np.setdiff1d(np.arange(total_samples), current_indices)
                additional_needed = current_samples - len(current_indices)
                additional_indices = np.random.choice(remaining_indices, additional_needed, replace=False)
                current_indices = np.concatenate([current_indices, additional_indices])
            
            df_subset = df.iloc[current_indices].copy()
            
            # Print class distribution for verification
            subset_classes = df_subset.iloc[:, 0]
            unique_classes, class_counts = np.unique(subset_classes, return_counts=True)
            print(f"Attempt {attempts + 1}: Class distribution: {dict(zip(unique_classes, class_counts))}")
            
            # Preprocess this subset
            try:
                x_train, y_train, df_test_processed, df_test_labels, img_size, num_classes = preprocess_incremental_data(
                    df_subset, df_test, seed=trial_seed + attempts
                )
                
                # Build and compile model
                model = build_model(img_size, num_classes)
                model = compile_model(model)
                
                # Create callbacks
                callbacks = create_callbacks()
                
                # Train model (fewer epochs for incremental learning)
                history = train_model(model, x_train, y_train, df_test_processed, to_categorical(df_test_labels, num_classes=num_classes), callbacks, epochs=30)
                
                # Evaluate model on the complete test set
                test_accuracy = evaluate_model(model, df_test_processed, to_categorical(df_test_labels, num_classes=num_classes), history, plot=False)
                
                print(f"Attempt {attempts + 1} accuracy: {test_accuracy:.4f}")
                
                # Check if this is better than our best
                if test_accuracy > best_accuracy:
                    best_accuracy = test_accuracy
                    best_model = model
                    best_history = history
                    best_sample_indices = current_indices.copy()
                    consecutive_deteriorations = 0  # Reset deterioration counter
                    print(f"New best accuracy: {best_accuracy:.4f} with {current_samples} samples")
                    break  # Found a good combination, move to next sample size
                else:
                    # Accuracy deteriorated or didn't improve
                    deterioration = best_accuracy - test_accuracy
                    print(f"Accuracy deteriorated by {deterioration:.4f}, trying different samples...")
                    attempts += 1
                    
            except Exception as e:
                print(f"Error in attempt {attempts + 1}: {e}")
                attempts += 1
        
        # If we couldn't find a better combination after all attempts
        if attempts >= max_attempts:
            consecutive_deteriorations += 1
            print(f"Could not improve accuracy after {max_attempts} attempts")
            print(f"Consecutive deteriorations: {consecutive_deteriorations}")
            
            if consecutive_deteriorations >= max_deteriorations:
                print(f"Stopping trial {trial_num} after {max_deteriorations} consecutive deteriorations")
                break
        
        # Check if target reached
        if best_accuracy >= target_accuracy:
            print(f"Target accuracy {target_accuracy} reached with {current_samples} samples!")
            break
        
        # Add more samples for next iteration
        current_samples += batch_size
    
    return {
        'trial_num': trial_num,
        'best_accuracy': best_accuracy,
        'samples_used': len(best_sample_indices) if best_sample_indices is not None else 0,
        'best_model': best_model,
        'best_history': best_history,
        'best_sample_indices': best_sample_indices,
        'target_reached': best_accuracy >= target_accuracy,
        'seed_used': trial_seed
    }

def run_multiple_trials(df, df_test, num_trials=10, target_accuracy=0.95, batch_size=100, max_samples=None):
    """
    Run multiple incremental learning trials and find the best performing subset
    """
    print(f"\n{'='*80}")
    print(f"RUNNING {num_trials} INCREMENTAL LEARNING TRIALS")
    print(f"Target accuracy: {target_accuracy}")
    print(f"Batch size: {batch_size}")
    print(f"{'='*80}")
    
    trials_results = []
    
    for trial in range(1, num_trials + 1):
        result = incremental_learning_trial(df, df_test, trial, target_accuracy, batch_size, max_samples)
        trials_results.append(result)
        
        print(f"\nTrial {trial} completed:")
        print(f"  Best accuracy: {result['best_accuracy']:.4f}")
        print(f"  Samples used: {result['samples_used']}")
        print(f"  Target reached: {result['target_reached']}")
    
    # Find the best trial
    best_trial = max(trials_results, key=lambda x: x['best_accuracy'])
    
    print(f"\n{'='*80}")
    print(f"BEST TRIAL RESULTS")
    print(f"{'='*80}")
    print(f"Trial number: {best_trial['trial_num']}")
    print(f"Best accuracy: {best_trial['best_accuracy']:.4f}")
    print(f"Samples used: {best_trial['samples_used']}")
    print(f"Target reached: {best_trial['target_reached']}")
    
    return best_trial, trials_results

def save_all_trials_subsets(df, all_trials, base_filename=None):
    """
    Save all trial subsets to individual CSV files with original row indices
    """
    if base_filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"trial_subset_{timestamp}"
    
    saved_files = []
    
    for trial in all_trials:
        if trial['best_sample_indices'] is not None:
            trial_subset = df.iloc[trial['best_sample_indices']].copy()
            
            # Add original row index column
            trial_subset['original_row_index'] = trial['best_sample_indices']
            
            filename = f"{base_filename}_trial_{trial['trial_num']}_acc_{trial['best_accuracy']:.4f}_samples_{trial['samples_used']}.csv"
            trial_subset.to_csv(filename, index=False)
            saved_files.append(filename)
            print(f"Trial {trial['trial_num']} subset saved to: {filename}")
            print(f"  - Original row indices: {trial['best_sample_indices'][:10]}..." if len(trial['best_sample_indices']) > 10 else f"  - Original row indices: {trial['best_sample_indices']}")
    
    return saved_files

def save_best_subset(df, best_trial, filename=None):
    """
    Save the best performing subset to a new CSV file with original row indices
    """
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"best_mnist_subset_{timestamp}.csv"
    
    if best_trial['best_sample_indices'] is not None:
        best_subset = df.iloc[best_trial['best_sample_indices']].copy()
        
        # Add original row index column
        best_subset['original_row_index'] = best_trial['best_sample_indices']
        
        best_subset.to_csv(filename, index=False)
        print(f"\nBest subset saved to: {filename}")
        print(f"Subset shape: {best_subset.shape}")
        print(f"Original row indices: {best_trial['best_sample_indices'][:10]}..." if len(best_trial['best_sample_indices']) > 10 else f"Original row indices: {best_trial['best_sample_indices']}")
        return filename
    else:
        print("No best subset to save")
        return None

def main():
    """
    Main function to run the incremental learning pipeline
    """
    print("=" * 80)
    print("MNIST Digit Recognition - Incremental Learning")
    print("=" * 80)
    
    try:
        # Load data
        df, df_test = load_data()
        
        # Check for missing values
        if not check_missing_values(df, df_test):
            print("Warning: Missing values detected in the data!")
            return
        
        # Run multiple trials
        best_trial, all_trials = run_multiple_trials(
            df, df_test, 
            num_trials=10,  # Number of trials to run
            target_accuracy=0.95,  # Target accuracy
            batch_size=100,  # Samples to add each iteration
            max_samples=None  # Maximum samples to use (None for all)
        )
        
        # Save all trial subsets
        saved_files = save_all_trials_subsets(df, all_trials)
        
        # Save best model if target was reached
        if best_trial['target_reached'] and best_trial['best_model'] is not None:
            save_model(best_trial['best_model'], 'best_incremental_model.h5')
        
        # Plot the best trial
        plot_best_trial(best_trial)
        
        # Print summary of all trials
        print(f"\n{'='*80}")
        print(f"SUMMARY OF ALL TRIALS")
        print(f"{'='*80}")
        for trial in all_trials:
            print(f"Trial {trial['trial_num']}: Accuracy={trial['best_accuracy']:.4f}, "
                  f"Samples={trial['samples_used']}, Target_reached={trial['target_reached']}")
        
        print(f"\n{'='*80}")
        print(f"PIPELINE COMPLETED!")
        print(f"Best accuracy achieved: {best_trial['best_accuracy']:.4f}")
        print(f"All trial subsets saved to: {', '.join(saved_files)}")
        print(f"Best subset saved to: {save_best_subset(df, best_trial)}")
        print(f"{'='*80}")
        
    except FileNotFoundError as e:
        print(f"Error: Could not find data files. Please update the file paths in the load_data() function.")
        print(f"Original error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main() 