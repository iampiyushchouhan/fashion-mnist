<div align="center">

# 👔 Fashion MNIST CNN Classifier

### *Deep Learning for Fashion Image Recognition*

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)
![Accuracy](https://img.shields.io/badge/Accuracy-91--93%25-success.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen.svg)

**A powerful Convolutional Neural Network achieving 91-93% accuracy on Fashion MNIST classification**

[Features](#-features) • [Architecture](#-architecture) • [Results](#-results) • [Quick Start](#-quick-start) • [Dataset](#-dataset)

---

</div>

## 🌟 Overview

This project implements a **state-of-the-art CNN architecture** for classifying fashion items from the Fashion MNIST dataset. Using advanced regularization techniques including **Batch Normalization** and **Dropout**, the model achieves exceptional performance in distinguishing between 10 different clothing categories.

<div align="center">

### 🎯 Performance Highlights

| Metric | Score |
|:------:|:-----:|
| **Test Accuracy** | 🎯 **91-93%** |
| **Architecture** | 🏗️ 3 Conv Blocks + 2 Dense |
| **Parameters** | 📊 Optimized |
| **Training Time** | ⚡ Fast Convergence |

</div>

---

## ✨ Features

<table>
<tr>
<td width="50%">

### 🏗️ **Architecture**
- 3 Convolutional Blocks
- 2 Fully Connected Layers
- Batch Normalization
- Dropout Regularization
- ReLU Activations
- Softmax Output

</td>
<td width="50%">

### 📊 **Evaluation**
- Accuracy & Loss Plots
- Confusion Matrix Analysis
- Classification Reports
- Per-Class Metrics
- Training Visualizations
- Performance Analytics

</td>
</tr>
</table>

---


## 🏛️ Architecture


```
┌─────────────────────────────────────────────┐
│          INPUT (28×28 Grayscale)            │
└─────────────────┬───────────────────────────┘
                  │
        ┌─────────▼─────────┐
        │  Conv Block 1     │  ◄── Conv2D + BatchNorm + ReLU + MaxPool
        └─────────┬─────────┘
                  │
        ┌─────────▼─────────┐
        │  Conv Block 2     │  ◄── Conv2D + BatchNorm + ReLU + MaxPool
        └─────────┬─────────┘
                  │
        ┌─────────▼─────────┐
        │  Conv Block 3     │  ◄── Conv2D + BatchNorm + ReLU + MaxPool
        └─────────┬─────────┘
                  │
        ┌─────────▼─────────┐
        │     Flatten       │
        └─────────┬─────────┘
                  │
        ┌─────────▼─────────┐
        │  Dense Layer 1    │  ◄── Dense + Dropout
        └─────────┬─────────┘
                  │
        ┌─────────▼─────────┐
        │  Dense Layer 2    │  ◄── Dense + Softmax
        └─────────┬─────────┘
                  │
        ┌─────────▼─────────┐
        │  OUTPUT (10 Classes) │
        └─────────────────────┘
```

---

## 📁 Dataset

<div align="center">

### Fashion MNIST - 10 Clothing Categories

</div>

| Class | Label | Category | Emoji |
|:-----:|:-----:|:--------:|:-----:|
| 0 | T-shirt/top | Upper Body | 👕 |
| 1 | Trouser | Lower Body | 👖 |
| 2 | Pullover | Upper Body | 🧥 |
| 3 | Dress | Full Body | 👗 |
| 4 | Coat | Upper Body | 🧥 |
| 5 | Sandal | Footwear | 👡 |
| 6 | Shirt | Upper Body | 👔 |
| 7 | Sneaker | Footwear | 👟 |
| 8 | Bag | Accessory | 👜 |
| 9 | Ankle boot | Footwear | 👢 |

**Dataset Specifications:**
- 📦 **Training Images:** 60,000
- 🧪 **Test Images:** 10,000
- 📐 **Image Size:** 28×28 pixels
- 🎨 **Color:** Grayscale
- 📊 **Classes:** 10 balanced categories

---

## 📈 Results

### 🎯 Model Performance

```
╔════════════════════════════════════════╗
║     CNN Performance Metrics            ║
╠════════════════════════════════════════╣
║  Overall Accuracy:    91-93%           ║
║  Training Stability:  Excellent        ║
║  Generalization:      Strong           ║
║  Convergence Speed:   Fast             ║
╚════════════════════════════════════════╝
```

### 🔍 Key Insights

> **✅ Strengths:**
> - Excellent performance on accessories (bags, footwear)
> - Strong differentiation between distinct categories
> - Robust generalization through regularization

> **⚠️ Challenges:**
> - **Shirt vs T-shirt:** Most confused pair (visually similar)
> - Upper body garments show some classification overlap
> - Expected behavior due to inherent visual similarity

### 📊 Visualization Outputs

The notebook generates comprehensive evaluation metrics:

1. **📈 Accuracy Plot** - Training vs Validation accuracy progression
2. **📉 Loss Plot** - Model convergence and overfitting detection  
3. **🎯 Confusion Matrix** - Detailed classification heatmap
4. **📋 Classification Report** - Precision, Recall, F1-Score per class

---

## 🚀 Quick Start

### Prerequisites

```bash
# Install required packages
pip install tensorflow numpy matplotlib scikit-learn seaborn
```

### 🏃‍♂️ Running the Project

```bash
# 1. Clone the repository
git clone <repository-url>
cd fashion-mnist-cnn

# 2. Launch Jupyter Notebook
jupyter notebook

# 3. Open and run the notebook
# Execute all cells to train and evaluate the model
```

### 📝 Notebook Workflow

```
1. 📥 Load Fashion MNIST Dataset
         ↓
2. 🔧 Preprocess & Normalize Data
         ↓
3. 🏗️ Build CNN Architecture
         ↓
4. 🎓 Train Model with Regularization
         ↓
5. 📊 Generate Evaluation Metrics
         ↓
6. 🎨 Visualize Results
```

---

## 🔬 Technical Implementation

### Regularization Strategies

<table>
<tr>
<td width="50%">

#### 🎯 Batch Normalization
- Normalizes layer inputs
- Accelerates training
- Improves stability
- Reduces internal covariate shift
- Enables higher learning rates

</td>
<td width="50%">

#### 🎲 Dropout
- Prevents overfitting
- Random neuron deactivation
- Improves generalization
- Forces redundant learning
- Enhances model robustness

</td>
</tr>
</table>

### 🧠 Model Insights

```
┌──────────────────────────────────────────────────┐
│  CONFUSION MATRIX ANALYSIS                       │
├──────────────────────────────────────────────────┤
│  ✅ High Accuracy Classes:                       │
│     • Bags (Class 8)                             │
│     • Ankle Boots (Class 9)                      │
│     • Sneakers (Class 7)                         │
│                                                  │
│  ⚠️  Frequently Confused:                        │
│     • Shirt ↔ T-shirt (Similar appearance)      │
│     • Pullover ↔ Coat (Similar styles)          │
└──────────────────────────────────────────────────┘
```

---

## 🔮 Future Enhancements

- [ ] 🔄 **Data Augmentation** - Rotation, flipping, zooming for dataset diversity
- [ ] 🏗️ **Advanced Architectures** - ResNet, EfficientNet, Vision Transformers
- [ ] 🎯 **Hyperparameter Tuning** - Grid/Random search optimization
- [ ] 🎭 **Transfer Learning** - Leverage pre-trained models
- [ ] 🤝 **Ensemble Methods** - Combine multiple models for higher accuracy
- [ ] ⚡ **Model Optimization** - Pruning, quantization for deployment
- [ ] 📱 **Web Deployment** - Flask/Streamlit application interface

---

## 🤝 Contributing

Contributions make the open-source community an amazing place to learn and create! Any contributions you make are **greatly appreciated**.

```bash
# Fork the Project
# Create your Feature Branch
git checkout -b feature/AmazingFeature

# Commit your Changes
git commit -m 'Add some AmazingFeature'

# Push to the Branch
git push origin feature/AmazingFeature

# Open a Pull Request
```

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---

## 🙏 Acknowledgments

- **Fashion MNIST Dataset** - [Zalando Research](https://github.com/zalandoresearch/fashion-mnist)
- **TensorFlow/Keras** - Deep learning framework
- **Python Community** - Amazing tools and libraries

---

<div align="center">

### 📬 Contact & Support

<div align="center">

<h3>👤 Author</h3>

<a href="https://github.com/iampiyushchouhan">
  <img src="https://github.com/iampiyushchouhan.png" alt="Piyush's Profile" width="120" style="border-radius: 50%;"/>
</a>

<p><strong>Piyush Chouhan</strong></p>
<h3> Need Help?</h3>

<a href="https://github.com/iampiyushchouhan/fashion-mnist/issues">
  <img src="https://img.shields.io/badge/GitHub-Issues-red?style=for-the-badge&logo=github" alt="GitHub Issues"/>
</a>
<a href="https://www.linkedin.com/in/iampiyushchouhan/">
  <img src="https://img.shields.io/badge/LinkedIn-Profile-blue?style=for-the-badge&logo=linkedin" alt="LinkedIn Profile"/>
</a>

</div>

If you found this project helpful, please consider giving it a ⭐!

**Built with ❤️ for the Deep Learning Community**

[Report Bug](../../issues) • [Request Feature](../../issues) • [Documentation](../../wiki)

</div>
