<p align="center">
  <img src="https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" />
  <img src="https://img.shields.io/badge/Keras-API-D00000?style=for-the-badge&logo=keras&logoColor=white" />
  <img src="https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" />
  <img src="https://img.shields.io/badge/Kivy-UI-40B5A4?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" />
</p>

<h1 align="center">🔐 Face Verification System</h1>
<h3 align="center">Real-Time One-Shot Face Verification using Siamese Neural Networks</h3>

<p align="center">
  A deep learning–powered face verification system that uses a <strong>Siamese Neural Network</strong> with a custom <strong>L1 Distance Layer</strong> to determine whether two face images belon[...]
</p>

---

## 📑 Table of Contents

- [✨ Features](#-features)
- [🏗️ System Architecture](#️-system-architecture)
- [🧠 How Siamese Networks Work](#-how-siamese-networks-work)
- [📂 Project Structure](#-project-structure)
- [⚙️ Installation & Setup](#️-installation--setup)
- [🗂️ Data Pipeline](#️-data-pipeline)
- [🧬 Model Architecture](#-model-architecture)
- [🏋️ Training Pipeline](#️-training-pipeline)
- [📊 Evaluation & Metrics](#-evaluation--metrics)
- [🖥️ Real-Time Application](#️-real-time-application)
- [🔄 Verification Algorithm](#-verification-algorithm)
- [🛠️ Technology Stack](#️-technology-stack)
- [🚀 Usage Guide](#-usage-guide)
- [📈 Performance](#-performance)
- [🤝 Contributing](#-contributing)
- [📜 License](#-license)

---

## ✨ Features

| Feature | Description |
|---|---|
| 🧠 **Siamese Neural Network** | Twin-network architecture for one-shot face verification |
| 📐 **Custom L1 Distance Layer** | Computes element-wise absolute difference between face embeddings |
| 📷 **Real-Time Webcam** | Live camera feed at 33 FPS via OpenCV |
| 🖥️ **Kivy Desktop App** | Cross-platform GUI with one-click verification |
| ⚡ **Batch Inference** | Efficient batch prediction against multiple verification images |
| 🔒 **Threshold-Based Decisions** | Configurable detection & verification thresholds |
| 💾 **Checkpoint System** | TensorFlow checkpoint saving every 10 epochs |
| 🖼️ **Google Colab Training** | Full training pipeline compatible with Google Colab + GPU |

---

## 🏗️ System Architecture

```mermaid
graph TB
    subgraph "📷 Input Layer"
        A[Webcam Feed<br/>OpenCV VideoCapture] -->|Frame Capture<br/>250x250px| B[Image Preprocessor<br/>Resize to 100x100]
    end

    subgraph "🧠 Deep Learning Engine"
        B --> C[Siamese Neural Network]
        C --> D[Embedding Network<br/>CNN → 4096-D Vector]
        E[Verification Images<br/>Stored References] --> F[Embedding Network<br/>Shared Weights]
        D --> G[L1 Distance Layer<br/>|Anchor − Validation|]
        F --> G
        G --> H[Dense Classifier<br/>Sigmoid → 0 or 1]
    end

    subgraph "📊 Decision Engine"
        H --> I{Detection<br/>Threshold > 0.5?}
        I -->|Yes| J[Positive Detection]
        I -->|No| K[Negative Detection]
        J --> L{Verification<br/>Ratio > 0.5?}
        K --> L
        L -->|Yes| M[✅ VERIFIED]
        L -->|No| N[❌ UNVERIFIED]
    end

    subgraph "🖥️ Kivy Application"
        M --> O[Update UI Label<br/>'Verified']
        N --> O
        O --> P[Display Result to User]
    end

    style A fill:#4FC3F7,stroke:#0288D1,color:#000
    style C fill:#FFB74D,stroke:#F57C00,color:#000
    style G fill:#CE93D8,stroke:#8E24AA,color:#000
    style M fill:#81C784,stroke:#388E3C,color:#000
    style N fill:#E57373,stroke:#D32F2F,color:#000
```

---

## 🧠 How Siamese Networks Work

A **Siamese Network** is a twin-network architecture where both branches share identical weights. It learns a **similarity function** rather than classifying faces directly — making it ideal for[...]

<!-- Pull Shark PR 2 -->
