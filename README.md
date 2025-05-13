# Handwritten-Mathematical-Symbols-Recognition-using-NeuralNets-from-Scratch

# NumPy MNIST Handwritten Digit Recognition (From Scratch)

This project is a **pure NumPy** implementation of a feedforward neural network for handwritten digit recognition using the [MNIST dataset](http://yann.lecun.com/exdb/mnist/).  
**No deep learning frameworks** (like TensorFlow, PyTorch, Keras, scikit-learn, or nnfs) are used-everything is built from scratch for educational purposes.

---

## Features

- Loads and preprocesses MNIST data directly from Yann LeCun's website
- Implements dense (fully connected) layers, ReLU and Softmax activations, and cross-entropy loss
- Uses the Adam optimizer (with bias correction) for training
- Includes batching, shuffling, and validation evaluation
- Achieves ~90% accuracy on MNIST test set

---

---

## Requirements

- Python 3.6+
- NumPy
- Matplotlib (optional, for plotting)

---

## How to Run

1. **Clone the repository:**
    ```
    git clone https://github.com/yourusername/mnist-numpy-from-scratch.git
    cd mnist-numpy-from-scratch
    ```

2. **Install dependencies:**
    ```
    pip install numpy matplotlib
    ```

3. **Run the script:**
    ```
    python mnist_numpy_from_scratch.py
    ```

    The script will automatically download the MNIST dataset and start training.  
    After each epoch, it prints the test loss and accuracy.

---


---

## How it Works

- **Data Loading:**  
  The script downloads and parses the MNIST dataset using only standard Python and NumPy.
- **Neural Network:**  
  - 3 fully connected (dense) layers: 784→128→64→10
  - ReLU activation for hidden layers, Softmax for output
  - Cross-entropy loss for classification
  - Adam optimizer for parameter updates
- **Training:**  
  - Mini-batch gradient descent (batch size 128)
  - Shuffling of training data each epoch
  - Validation on test set after each epoch

---

## Customization

- **Change network architecture:**  
  Modify the number of layers or neurons in `mnist_numpy_from_scratch.py`.
- **Adjust training parameters:**  
  Change `epochs`, `batch_size`, or learning rate as needed.

---

## Educational Value

This repository is intended for learning and demonstration purposes.  
For production or research, use optimized libraries such as TensorFlow or PyTorch.

---

## License

MIT License

---

## References

- [MNIST Database](http://yann.lecun.com/exdb/mnist/)
- [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)

