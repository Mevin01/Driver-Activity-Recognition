# Driver Activity Recognition using Machine Learning Algorithms

## Overview
This project focuses on **Driver Activity Recognition** using five different machine learning and deep learning algorithms:

1. **Deep Belief Network (DBN)**
2. **Gaussian Network Model (GNM)**
3. **Hidden Markov Model (HMM)**
4. **Long Short-Term Memory (LSTM)**
5. **Fuzzy Rule-Based Network**

The goal is to classify driver activities based on sensor data using these models and evaluate their performance.

---

## Dataset
The dataset contains sensor readings from a vehicle, including parameters such as speed, acceleration, steering angle, and more. The target variable represents different driver activities.

- **Training Data:** `driver_activity_train.csv`
- **Testing Data:** `driver_activity_test.csv`

Each row consists of:
- **10 Features**: Sensor data from the vehicle
- **1 Target Label**: Represents driver activity

---

## Algorithms Implemented

### 1️⃣ Deep Belief Network (DBN)
- **Feature Extraction:** Uses Restricted Boltzmann Machines (RBMs)
- **Process:** Trains multiple layers of RBMs and fine-tunes using backpropagation
- **Performance:** Moderate accuracy, requires more training data

### 2️⃣ Gaussian Network Model (GNM)
- **Feature Extraction:** Gaussian Mixture Model-based clustering
- **Process:** Models the probability distribution of driver activities using Gaussian distributions
- **Performance:** Low accuracy due to data limitations

### 3️⃣ Hidden Markov Model (HMM)
- **Feature Extraction:** Time-series state transition probabilities
- **Process:** Learns sequential dependencies in driver behavior
- **Performance:** Works well for sequential data but needs fine-tuning

### 4️⃣ Long Short-Term Memory (LSTM)
- **Feature Extraction:** Captures temporal dependencies
- **Process:** Uses memory cells to retain long-term dependencies in data
- **Performance:** Best among all models but affected by data scarcity

### 5️⃣ Fuzzy Rule-Based Network
- **Feature Extraction:** Rule-based inference
- **Process:** Uses predefined rules to classify driver behavior
- **Performance:** Less sensitive to data but needs expert-defined rules

---

## Performance Evaluation
Each model is evaluated using:
- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**

| Model | Train Accuracy | Test Accuracy | Precision | Recall | F1-Score |
|--------|---------------|--------------|-----------|--------|---------|
| DBN    | 0.35         | 0.30         | 0.32      | 0.30   | 0.31    |
| GNM    | 0.20         | 0.21         | 0.20      | 0.21   | 0.20    |
| HMM    | 0.28         | 0.25         | 0.27      | 0.25   | 0.26    |
| LSTM   | 0.44         | 0.19         | 0.20      | 0.19   | 0.19    |
| Fuzzy  | 0.31         | 0.28         | 0.30      | 0.28   | 0.29    |

🛑 **Note:** The model performance is limited due to a small dataset. More data would significantly improve accuracy.

---

## Installation & Usage

### Prerequisites
- Python 3.x
- TensorFlow
- Scikit-learn
- NumPy

### Steps to Run the Models
```sh
# Clone the repository
git clone https://github.com/YourUsername/DriverActivityRecognition.git
cd DriverActivityRecognition

# Install dependencies
pip install -r requirements.txt

# Run the model scripts
python dbn_model.py
python gnm_model.py
python hmm_model.py
python lstm_model.py
python fuzzy_model.py
```

---

## Future Work
- Collecting a **larger dataset** to improve accuracy
- **Hyperparameter tuning** for better performance
- Combining multiple models for an **ensemble approach**

---

## Contributing
If you have any ideas to improve the model, feel free to open an issue or a pull request. Let's collaborate!

🚀 Happy Coding! 🎯

