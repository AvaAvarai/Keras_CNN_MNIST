# MNIST Digit Recognition CNN

This is a standalone Python script converted from a Jupyter notebook that implements a Convolutional Neural Network (CNN) for MNIST digit recognition, achieving 99.6% accuracy. The original notebook is available at [Medium](https://medium.com/@BrendanArtley/mnist-keras-simple-cnn-99-6-731b624aee7f) and [GitHub](https://github.com/brendanartley/Medium-Article-Code/blob/main/code/mnist-keras-cnn-99-6.ipynb).

## Features

- **CNN Architecture**: Deep convolutional neural network with batch normalization and dropout
- **Data Augmentation**: Image rotation, zoom, and shift for better generalization
- **Learning Rate Scheduling**: Adaptive learning rate decay for optimal training
- **Early Stopping**: Prevents overfitting by monitoring validation loss
- **Model Evaluation**: Comprehensive evaluation with training history plots
- **Prediction Generation**: Outputs predictions in CSV format for submission

## Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Data Setup

1. Download the MNIST dataset from Kaggle:
   - Go to [Digit Recognizer](https://www.kaggle.com/c/digit-recognizer)
   - Download `train.csv` and `test.csv`

2. Update the file paths in the `load_data()` function in `mnist_cnn_script.py`:

   ```python
   train = pd.read_csv('path/to/your/train.csv')
   test = pd.read_csv('path/to/your/test.csv')
   ```

## Usage

Run the script:

```bash
python mnist_cnn_script.py
```

## Output

The script will:

1. **Load and preprocess** the MNIST data
2. **Build and train** the CNN model
3. **Evaluate** the model performance
4. **Generate plots** showing training history
5. **Save predictions** to `mnist_predictions.csv`
6. **Save the trained model** as `mnist_cnn_model.h5`

## Model Architecture

The CNN consists of:
- **2 Convolutional blocks** with 32 and 64 filters respectively
- **Batch normalization** after each convolutional layer
- **Max pooling** for dimensionality reduction
- **Dropout layers** for regularization
- **Dense layers** (512 and 1024 units) for classification
- **Softmax output** for 10-digit classification

## Training Parameters

- **Batch size**: 64
- **Epochs**: 50
- **Optimizer**: Adam with learning rate 0.001
- **Loss function**: Categorical crossentropy
- **Data augmentation**: Rotation (±10°), zoom (±10%), shift (±10%)

## Expected Performance

- **Training accuracy**: ~99%
- **Validation accuracy**: ~99.6%
- **Test accuracy**: ~99.6%

## Files Generated

- `mnist_predictions.csv`: Predictions for test set
- `mnist_cnn_model.h5`: Trained model file
- Training plots will be displayed during execution

## Notes

- The script includes comprehensive error handling
- All random seeds are set for reproducibility
- The model uses early stopping to prevent overfitting
- Learning rate scheduling helps achieve better convergence

## Troubleshooting

If you encounter issues:

1. **File not found errors**: Update the data file paths in `load_data()`
2. **Memory errors**: Reduce batch size in `train_model()`
3. **Import errors**: Ensure all dependencies are installed correctly

## Credits

This implementation is based on techniques from various Kaggle notebooks and external resources, including:

- Chinmay Rane's CNN tutorial
- DATAI's CNN tutorial  
- Chris Deotte's image processing techniques

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for full details.
