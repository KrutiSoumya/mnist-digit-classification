🔢 MNIST Digit Classification with Keras

This project showcases two deep learning architectures to classify handwritten digits from the MNIST dataset using TensorFlow's Keras API:
- ✅ A simple Sequential Model
- 🔀 A more flexible Functional API Model with branching and merging layers

📚 Libraries & Tools
- numpy — Numerical operations and array manipulation
- matplotlib — Visualizing sample digits
- tensorflow.keras — Building, training, and evaluating models

🚀 Project Workflow
- Load & Preprocess Data
- Normalize pixel values (0–255 → 0–1)
- One-hot encode target labels
- Visualize Dataset
- Display a sample digit and its class
- Build Models
- 🔹 Sequential Model — Simple feedforward network
- 🔸 Functional API Model — Branching architecture for richer feature extraction
- Train & Evaluate
- Monitor performance on validation and test sets
- Compare model accuracies
- Model Visualization
- Save the architecture diagram for the Functional Model

🧠 Architectures
✅ Sequential Model
Input → Flatten → Dense(5, activation='relu') → Dense(10, activation='softmax')

A minimalistic model for baseline evaluation.

🔀 Functional API Model
Input → Flatten
       ├─ Dense(128, activation='relu') → Dense(64, activation='relu')
       ├─ Dense(256, activation='relu')
       ↓
      Concatenate → Dense(10, activation='softmax')


Designed to extract multi-path features and encourage richer learning.

📊 Results
After training, each model is evaluated on the test dataset. Accuracy and loss metrics help compare performance between the two architectures.

📁 Extras
- 📎 Saves a visual representation of the functional model using plot_model
- 📌 Ready for further tuning or hyperparameter experimentation



