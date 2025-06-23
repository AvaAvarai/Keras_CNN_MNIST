#!/usr/bin/env python3
"""
MNIST Digit Recognition using CNN with Keras
Achieves 99.6% accuracy on the MNIST dataset

This script is converted from a Jupyter notebook that implements:
- Convolutional Neural Network (CNN) architecture
- Data augmentation and preprocessing
- Learning rate scheduling
- Early stopping
- Model evaluation and prediction

Based on techniques from various Kaggle notebooks and external resources.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical

from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.optimizers import Adam

from keras.callbacks import LearningRateScheduler
from keras.callbacks import EarlyStopping

def load_data():
    """
    Load MNIST training and test data
    Note: Update the file paths according to your data location
    """
    print("Loading data...")
    
    # Update these paths to match your data location
    train = pd.read_csv('../input/digit-recognizer/train.csv')
    test = pd.read_csv('../input/digit-recognizer/test.csv')
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
    - Reshaping images
    - Normalization
    - Data augmentation
    - One-hot encoding
    """
    print("\nPreprocessing data...")
    
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Split training data
    X = df.iloc[:, 1:]  # All columns except the first (labels)
    Y = df.iloc[:, 0]   # First column (labels)
    
    x_train, x_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.1, random_state=seed
    )
    
    print(f"Training set: {x_train.shape}")
    print(f"Validation set: {x_test.shape}")
    
    # Reshape images from 1D to 3D (28x28x1)
    # -1 means let numpy figure out the number of examples
    x_train = x_train.values.reshape(-1, 28, 28, 1)
    x_test = x_test.values.reshape(-1, 28, 28, 1)
    df_test = df_test.values.reshape(-1, 28, 28, 1)
    
    print(f"Reshaped training data: {x_train.shape}")
    print(f"Reshaped validation data: {x_test.shape}")
    print(f"Reshaped test data: {df_test.shape}")
    
    # Normalize pixel values to [0, 1]
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    df_test = df_test.astype("float32") / 255
    
    # Create data augmentation generator
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range=0.1,  # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False  # randomly flip images
    )
    
    # Fit the data generator on training data
    datagen.fit(x_train)
    
    # One-hot encode labels
    y_train = to_categorical(y_train, num_classes=10)
    y_test = to_categorical(y_test, num_classes=10)
    
    print(f"One-hot encoded training labels: {y_train.shape}")
    print(f"One-hot encoded validation labels: {y_test.shape}")
    
    return x_train, x_test, y_train, y_test, df_test, datagen

def build_model():
    """
    Build the CNN model architecture
    """
    print("\nBuilding CNN model...")
    
    model = Sequential()
    
    # First Convolutional Block
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', 
                     strides=1, padding='same', data_format='channels_last',
                     input_shape=(28, 28, 1)))
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
    model.add(Dense(10, activation='softmax'))
    
    return model

def compile_model(model):
    """
    Compile the model with optimizer, loss function, and metrics
    """
    print("\nCompiling model...")
    
    # Optimizer
    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
    
    # Compile the model
    model.compile(optimizer=optimizer, 
                  loss="categorical_crossentropy", 
                  metrics=["accuracy"])
    
    # Print model summary
    model.summary()
    
    return model

def create_callbacks():
    """
    Create callbacks for training
    """
    print("\nCreating callbacks...")
    
    # Learning Rate Scheduler
    reduce_lr = LearningRateScheduler(lambda x: 1e-3 * 0.9 ** x)
    
    # Early Stopping
    early_stopping = EarlyStopping(
        min_delta=0.001,  # minimum amount of change to count as an improvement
        patience=20,  # how many epochs to wait before stopping
        restore_best_weights=True,
    )
    
    # Visualize learning rate decay
    print("Learning rate decay schedule:")
    decays = [(lambda x: 1e-3 * 0.9 ** x)(x) for x in range(10)]
    for i, lr in enumerate(decays, 1):
        print(f"Epoch {i} Learning Rate: {lr}")
    
    return [reduce_lr, early_stopping]

def train_model(model, x_train, y_train, x_test, y_test, datagen, callbacks):
    """
    Train the model
    """
    print("\nTraining model...")
    
    batch_size = 64
    epochs = 50
    
    # Fit the model
    history = model.fit_generator(
        datagen.flow(x_train, y_train, batch_size=batch_size),
        epochs=epochs,
        validation_data=(x_test, y_test),
        verbose=1,
        steps_per_epoch=x_train.shape[0] // batch_size,
        callbacks=callbacks
    )
    
    return history

def evaluate_model(model, x_test, y_test, history):
    """
    Evaluate the model and plot training history
    """
    print("\nEvaluating model...")
    
    # Evaluate on test set
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    
    # Plot training history
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

def make_predictions(model, df_test):
    """
    Make predictions on test data and save to CSV
    """
    print("\nMaking predictions...")
    
    # Make predictions
    pred_digits_test = np.argmax(model.predict(df_test), axis=1)
    
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

def main():
    """
    Main function to run the complete MNIST CNN pipeline
    """
    print("=" * 60)
    print("MNIST Digit Recognition using CNN")
    print("=" * 60)
    
    try:
        # Load data
        df, df_test = load_data()
        
        # Check for missing values
        if not check_missing_values(df, df_test):
            print("Warning: Missing values detected in the data!")
            return
        
        # Preprocess data
        x_train, x_test, y_train, y_test, df_test, datagen = preprocess_data(df, df_test)
        
        # Build model
        model = build_model()
        
        # Compile model
        model = compile_model(model)
        
        # Create callbacks
        callbacks = create_callbacks()
        
        # Train model
        history = train_model(model, x_train, y_train, x_test, y_test, datagen, callbacks)
        
        # Evaluate model
        test_accuracy = evaluate_model(model, x_test, y_test, history)
        
        # Make predictions
        submission = make_predictions(model, df_test)
        
        # Save model
        save_model(model)
        
        print("\n" + "=" * 60)
        print(f"Training completed! Final test accuracy: {test_accuracy:.4f}")
        print("=" * 60)
        
    except FileNotFoundError as e:
        print(f"Error: Could not find data files. Please update the file paths in the load_data() function.")
        print(f"Original error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main() 