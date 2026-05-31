<p align="center">
  <img src="https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" />
  <img src="https://img.shields.io/badge/Keras-API-D00000?style=for-the-badge&logo=keras&logoColor=white" />
  <img src="https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" />
  <img src="https://img.shields.io/badge/Kivy-UI-40B5A4?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" />
  <img src="https://img.shields.io/github/repo-size/Paragiscool/Face_verification_ml_project?style=for-the-badge&color=blueviolet" />
  <img src="https://img.shields.io/github/last-commit/Paragiscool/Face_verification_ml_project?style=for-the-badge&color=orange" />
</p>

<h1 align="center">🔐 Face Verification System</h1>
<h3 align="center">Real-Time One-Shot Face Verification using Siamese Neural Networks</h3>

<p align="center">
  A deep learning–powered face verification system that uses a <strong>Siamese Neural Network</strong> with a custom <strong>L1 Distance Layer</strong> to determine whether two face images belong to the same person. The model is trained on anchor/positive/negative triplets and deployed via a real-time <strong>Kivy desktop application</strong> with live webcam feed.
</p>

<p align="center">
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-features">Features</a> •
  <a href="#-system-architecture">Architecture</a> •
  <a href="#-model-architecture">Model</a> •
  <a href="#-usage-guide">Usage</a> •
  <a href="#-troubleshooting">Troubleshooting</a>
</p>

---

## ⚡ Quick Start

```bash
# Clone & set up
git clone https://github.com/Paragiscool/Face_verification_ml_project.git
cd Face_verification_ml_project
python -m venv .venv && .venv\Scripts\activate      # Windows
pip install -r requirements.txt && pip install kivy

# Place 3–5 photos of your face inside:
#   application_data/verification_images/

# Launch the app
python faceid.py
# → Click "Verify" to authenticate against your stored images
```

> **Note:** A pre-trained model (`siamesemodelv2.h5`, ~155 MB) must be present in the project root.  
> It is excluded from Git via `.gitignore` due to file size — see [Training Pipeline](#️-training-pipeline) to generate your own.

---

## 📑 Table of Contents

- [⚡ Quick Start](#-quick-start)
- [✨ Features](#-features)
- [🏗️ System Architecture](#️-system-architecture)
- [🧠 How Siamese Networks Work](#-how-siamese-networks-work)
- [🔬 Mathematical Foundations](#-mathematical-foundations)
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
- [🐛 Troubleshooting](#-troubleshooting)
- [🤝 Contributing](#-contributing)
- [📚 References](#-references)
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
| 🛡️ **Graceful Error Handling** | Empty verification directory handling, directory auto-creation |
| 🔁 **One-Shot Learning** | Verify identity from just a handful of reference images |

---

## 🏗️ System Architecture

The system is organized into four distinct layers that communicate sequentially during each verification request:

```mermaid
graph TB
    subgraph "📷 Input Layer"
        A[Webcam Feed<br/>OpenCV VideoCapture] -->|Frame Capture<br/>250x250px| B[Image Preprocessor<br/>Resize to 100x100]
    end

    subgraph "🧠 Deep Learning Engine"
        B --> C[Siamese Neural Network]
        C --> D[Embedding Network<br/>CNN → 4096-D Vector]
        E[Verification Images<br/>Stored References] --> F[Embedding Network<br/>Shared Weights]
        D --> G[L1 Distance Layer<br/>abs Anchor − Validation]
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
        M --> O[Update UI Label<br/>Verified]
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

A **Siamese Network** is a twin-network architecture where both branches share identical weights. It learns a **similarity function** rather than classifying faces directly — making it ideal for **one-shot learning** tasks like face verification.

```mermaid
graph LR
    subgraph "Twin Network Architecture"
        direction TB
        A["🖼️ Anchor Image"] --> B["🔵 CNN Encoder<br/>(Shared Weights)"]
        C["🖼️ Verification Image"] --> D["🔵 CNN Encoder<br/>(Shared Weights)"]
        B --> E["📐 4096-D Embedding"]
        D --> F["📐 4096-D Embedding"]
        E --> G["🔶 L1 Distance<br/>|E_1 − E_2|"]
        F --> G
        G --> H["🟢 Dense 1 sigmoid"]
        H --> I{"Same Person?"}
        I -->|"> 0.5"| J["✅ Yes"]
        I -->|"≤ 0.5"| K["❌ No"]
    end

    style B fill:#42A5F5,stroke:#1565C0,color:#fff
    style D fill:#42A5F5,stroke:#1565C0,color:#fff
    style G fill:#FFA726,stroke:#E65100,color:#000
    style J fill:#66BB6A,stroke:#2E7D32,color:#fff
    style K fill:#EF5350,stroke:#C62828,color:#fff
```

### Key Concepts

| Concept | Explanation |
|---|---|
| **Anchor** | The reference face image of the known person |
| **Positive** | Another image of the **same** person |
| **Negative** | An image of a **different** person |
| **Embedding** | A 4096-dimensional feature vector representing a face |
| **L1 Distance** | Element-wise absolute difference: `\|embedding₁ - embedding₂\|` |
| **One-Shot Learning** | Verify identity from a single (or few) reference image(s) |

### Why Siamese over Classification?

| Approach | Pros | Cons |
|---|---|---|
| **Standard Classifier** | Fast inference, simple training | Needs retraining per new user; large dataset required |
| **Siamese Network** ✅ | Works with 1–5 reference images; no retraining needed | Slower per-comparison inference |

---

## 🔬 Mathematical Foundations

### Embedding Function

The CNN encoder maps an input image **x** to a high-dimensional embedding space:

```
f(x) : ℝ^(100×100×3) → ℝ^4096
```

### L1 Distance (Manhattan Distance)

Given two embeddings **f(x₁)** and **f(x₂)**, the distance is computed element-wise:

```
D(x₁, x₂) = |f(x₁) - f(x₂)| ∈ ℝ^4096
```

### Classification

The distance vector is passed through a single sigmoid neuron:

```
P(same person) = σ(W · D(x₁, x₂) + b)
```

Where `σ` is the sigmoid activation, `W ∈ ℝ^4096` and `b ∈ ℝ` are learnable parameters.

### Dual-Threshold Decision

```
detection(i)  = 𝟙[P_i > τ_d]           where τ_d = 0.5 (detection threshold)
verification  = (Σ detection(i)) / N    where N = total verification images
result        = 𝟙[verification > τ_v]   where τ_v = 0.5 (verification threshold)
```

```mermaid
flowchart LR
    A["P_i for each<br/>reference image"] --> B{"P_i > 0.5?"}
    B -->|Yes| C["detection = 1"]
    B -->|No| D["detection = 0"]
    C --> E["Sum all<br/>detections"]
    D --> E
    E --> F["ratio = sum / N"]
    F --> G{"ratio > 0.5?"}
    G -->|Yes| H["✅ Verified"]
    G -->|No| I["❌ Unverified"]

    style H fill:#81C784,stroke:#388E3C,color:#000
    style I fill:#E57373,stroke:#D32F2F,color:#000
```

---

## 📂 Project Structure

```
Face_verification_ml_project/
│
├── 📄 faceid.py                  # Kivy desktop application (real-time verification)
├── 📄 layers.py                  # Custom L1 Distance layer for model loading
├── 📄 siamese_network_setup.py   # Full training pipeline (Colab-based)
├── 📄 requirements.txt           # Python dependencies
├── 📄 TODO.md                    # Development checklist
├── 📄 .gitignore                 # Git ignore rules
├── 📄 pyrightconfig.json         # Python type-checking config
│
├── 🧠 siamesemodelv2.h5          # Trained Siamese model weights (~155MB, git-ignored)
│
├── 📁 data/                      # Training dataset (git-ignored)
│   ├── 📁 anchor/                # Anchor face images (your face)
│   ├── 📁 positive/              # Positive samples (your face, different shots)
│   └── 📁 negative/              # Negative samples (other people — LFW dataset)
│
├── 📁 application_data/          # Runtime application data (git-ignored)
│   ├── 📁 input_image/           # Current webcam capture for verification
│   └── 📁 verification_images/   # Reference images to verify against
│
└── 📁 training_checkpoints/      # TensorFlow model checkpoints (git-ignored)
```

```mermaid
graph TD
    A["📦 Project Root"] --> B["📄 faceid.py<br/><i>Kivy App — Entry Point</i>"]
    A --> C["📄 layers.py<br/><i>Custom L1Dist Layer</i>"]
    A --> D["📄 siamese_network_setup.py<br/><i>Training Pipeline</i>"]
    A --> E["🧠 siamesemodelv2.h5<br/><i>Trained Model ~155MB</i>"]
    A --> F["📁 data/"]
    A --> G["📁 application_data/"]

    F --> F1["📁 anchor/"]
    F --> F2["📁 positive/"]
    F --> F3["📁 negative/"]

    G --> G1["📁 input_image/"]
    G --> G2["📁 verification_images/"]

    B -->|imports| C
    B -->|loads| E
    D -->|produces| E

    style B fill:#4CAF50,stroke:#2E7D32,color:#fff
    style C fill:#2196F3,stroke:#1565C0,color:#fff
    style D fill:#FF9800,stroke:#E65100,color:#fff
    style E fill:#9C27B0,stroke:#6A1B9A,color:#fff
```

### File Roles

| File | Lines | Role |
|---|---|---|
| `faceid.py` | 133 | Kivy GUI application — webcam feed, verify button, model inference |
| `layers.py` | 24 | Custom `L1Dist` Keras layer (needed for `model.load`) |
| `siamese_network_setup.py` | 581 | End-to-end training pipeline: data collection → training → evaluation → Colab verification |

---

## ⚙️ Installation & Setup

### Prerequisites

| Requirement | Minimum | Recommended |
|---|---|---|
| **Python** | 3.10 | 3.10+ |
| **GPU** | — (CPU works) | NVIDIA with CUDA (for training) |
| **Webcam** | Any USB/built-in | 720p+ |
| **RAM** | 4 GB | 8 GB+ |
| **Disk** | 500 MB (code + model) | 2 GB+ (with dataset) |

### Step-by-Step

```bash
# 1. Clone the repository
git clone https://github.com/Paragiscool/Face_verification_ml_project.git
cd Face_verification_ml_project

# 2. Create a virtual environment
python -m venv .venv

# 3. Activate the environment
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Install Kivy (for the desktop app)
pip install kivy
```

```mermaid
graph LR
    A["Clone Repo"] --> B["Create venv"]
    B --> C["Install Dependencies"]
    C --> D["Install Kivy"]
    D --> E{"Training or<br/>Inference?"}
    E -->|Training| F["Run siamese_network_setup.py<br/>on Google Colab"]
    E -->|Inference| G["Run faceid.py<br/>Desktop App"]

    style A fill:#E3F2FD,stroke:#1565C0,color:#000
    style F fill:#FFF3E0,stroke:#E65100,color:#000
    style G fill:#E8F5E9,stroke:#2E7D32,color:#000
```

---

## 🗂️ Data Pipeline

### Data Collection Flow

Anchor & positive images are captured from your webcam; negative images are sourced from the [LFW (Labeled Faces in the Wild)](https://vis-www.cs.umass.edu/lfw/) dataset via HuggingFace.

```mermaid
flowchart TD
    A["🎥 Start Webcam<br/>cv2.VideoCapture 0"] --> B["📐 Crop Frame<br/>250×250px Region"]
    B --> C{"⌨️ Key Press?"}
    
    C -->|"Press 'a'"| D["💾 Save to<br/>data/anchor/<br/>UUID.jpg"]
    C -->|"Press 'p'"| E["💾 Save to<br/>data/positive/<br/>UUID.jpg"]
    C -->|"Press 'q'"| F["🛑 Stop Capture"]
    
    D --> C
    E --> C
    
    G["🌐 Hugging Face<br/>LFW Dataset"] --> H["📥 Download<br/>Negative Samples"]
    H --> I["💾 Save to<br/>data/negative/"]
    
    subgraph "Final Dataset"
        D --> J["📁 Anchor Images<br/>(300 samples)"]
        E --> K["📁 Positive Images<br/>(300 samples)"]
        I --> L["📁 Negative Images<br/>(300 samples)"]
    end

    style A fill:#4FC3F7,stroke:#0288D1,color:#000
    style G fill:#FFB74D,stroke:#F57C00,color:#000
    style J fill:#81C784,stroke:#388E3C,color:#000
    style K fill:#81C784,stroke:#388E3C,color:#000
    style L fill:#E57373,stroke:#D32F2F,color:#000
```

### Image Preprocessing Pipeline

Every image — whether for training or real-time inference — passes through the same preprocessing:

```mermaid
flowchart LR
    A["📷 Raw Image<br/>(Variable Size)"] --> B["📖 tf.io.read_file<br/>Read Bytes"]
    B --> C["🖼️ tf.io.decode_jpeg<br/>Decode to Tensor"]
    C --> D["📐 tf.image.resize<br/>100×100×3"]
    D --> E["⚖️ Normalize<br/>pixel / 255.0"]
    E --> F["✅ Preprocessed<br/>Tensor 0 to 1"]

    style A fill:#FFECB3,stroke:#FF8F00,color:#000
    style F fill:#C8E6C9,stroke:#388E3C,color:#000
```

### Dataset Construction

```mermaid
flowchart TD
    A["📁 Anchor Images"] --> D["🔗 tf.data.Dataset.zip"]
    B["📁 Positive Images"] --> D
    C["Label = 1"] --> D
    
    E["📁 Anchor Images"] --> H["🔗 tf.data.Dataset.zip"]
    F["📁 Negative Images"] --> H
    G["Label = 0"] --> H
    
    D --> I["➕ Concatenate<br/>Positives + Negatives"]
    H --> I
    
    I --> J["🔀 Shuffle<br/>buffer_size=1024"]
    J --> K["🗃️ Cache"]
    K --> L["✂️ 70/30 Split"]
    
    L --> M["🏋️ Train Set<br/>batch=16 prefetch=8"]
    L --> N["🧪 Test Set<br/>batch=16 prefetch=8"]

    style D fill:#42A5F5,stroke:#1565C0,color:#fff
    style H fill:#42A5F5,stroke:#1565C0,color:#fff
    style M fill:#66BB6A,stroke:#2E7D32,color:#fff
    style N fill:#FFA726,stroke:#E65100,color:#fff
```

---

## 🧬 Model Architecture

### Embedding Network (CNN)

The embedding network transforms a `100×100×3` face image into a **4096-dimensional feature vector**:

```mermaid
graph TD
    A["📷 Input Image<br/>100×100×3"] --> B["🔵 Conv2D<br/>64 filters 10×10<br/>ReLU"]
    B --> C["🔽 MaxPool2D<br/>2×2 same padding"]
    C --> D["🔵 Conv2D<br/>128 filters 7×7<br/>ReLU"]
    D --> E["🔽 MaxPool2D<br/>2×2 same padding"]
    E --> F["🔵 Conv2D<br/>128 filters 4×4<br/>ReLU"]
    F --> G["🔽 MaxPool2D<br/>2×2 same padding"]
    G --> H["🔵 Conv2D<br/>256 filters 4×4<br/>ReLU"]
    H --> I["📊 Flatten"]
    I --> J["🟣 Dense<br/>4096 units Sigmoid"]
    J --> K["📐 4096-D<br/>Face Embedding"]

    style A fill:#E3F2FD,stroke:#1565C0,color:#000
    style B fill:#42A5F5,stroke:#1565C0,color:#fff
    style D fill:#42A5F5,stroke:#1565C0,color:#fff
    style F fill:#42A5F5,stroke:#1565C0,color:#fff
    style H fill:#42A5F5,stroke:#1565C0,color:#fff
    style J fill:#AB47BC,stroke:#6A1B9A,color:#fff
    style K fill:#66BB6A,stroke:#2E7D32,color:#fff
```

### Full Siamese Network Architecture

```
Layer (type)                    Output Shape         Param #     Connected to
================================================================================================
input_img (InputLayer)          [(None, 100, 100, 3)]  0         
validation_img (InputLayer)     [(None, 100, 100, 3)]  0         
embedding (Functional)          (None, 4096)         24,843,008  input_img, validation_img
L1_distance (L1Dist)            (None, 4096)         0           embedding[0], embedding[1]
dense_classifier (Dense)        (None, 1)            4,097       L1_distance
================================================================================================
Total params: 24,847,105
Trainable params: 24,847,105
Non-trainable params: 0
```

### Layer-by-Layer Parameter Breakdown

```mermaid
pie title Model Parameter Distribution
    "Conv2D Layer 1 — 64 filters" : 19264
    "Conv2D Layer 2 — 128 filters" : 401536
    "Conv2D Layer 3 — 128 filters" : 262272
    "Conv2D Layer 4 — 256 filters" : 524544
    "Dense 4096 — Embedding" : 23633920
    "Dense 1 — Classifier" : 4097
```

> **Insight:** The Dense embedding layer alone accounts for **~95%** of all parameters. This large bottleneck is what enables the network to encode rich facial features into the 4096-D vector.

---

## 🏋️ Training Pipeline

### Training Flow

```mermaid
flowchart TD
    A["🚀 Start Training<br/>EPOCHS = 50"] --> B["📦 Load Training Batch<br/>batch_size = 16"]
    
    B --> C["🔄 Forward Pass<br/>siamese_model X"]
    C --> D["📉 Compute Loss<br/>BinaryCrossentropy"]
    D --> E["📐 Compute Gradients<br/>tf.GradientTape"]
    E --> F["⚙️ Update Weights<br/>Adam Optimizer<br/>lr = 1e-4"]
    F --> G["📊 Update Metrics<br/>Precision and Recall"]
    
    G --> H{"More Batches?"}
    H -->|Yes| B
    H -->|No| I{"Epoch mod 10 = 0?"}
    
    I -->|Yes| J["💾 Save Checkpoint"]
    I -->|No| K{"More Epochs?"}
    J --> K
    
    K -->|Yes| B
    K -->|No| L["✅ Training Complete<br/>Save siamesemodelv2.h5"]

    style A fill:#4CAF50,stroke:#2E7D32,color:#fff
    style D fill:#F44336,stroke:#C62828,color:#fff
    style F fill:#2196F3,stroke:#1565C0,color:#fff
    style J fill:#FF9800,stroke:#E65100,color:#fff
    style L fill:#4CAF50,stroke:#2E7D32,color:#fff
```

### Training Configuration

| Parameter | Value |
|---|---|
| **Optimizer** | Adam |
| **Learning Rate** | `1e-4` |
| **Loss Function** | Binary Cross-Entropy |
| **Epochs** | 50 |
| **Batch Size** | 16 |
| **Prefetch Buffer** | 8 |
| **Shuffle Buffer** | 1024 |
| **Train/Test Split** | 70% / 30% |
| **Checkpoint Interval** | Every 10 epochs |

### Custom Training Step (`@tf.function`)

```python
@tf.function
def train_step(batch):
    with tf.GradientTape() as tape:
        X = batch[:2]           # (anchor, positive/negative)
        y = batch[2]            # Label (1 or 0)
        yhat = siamese_model(X, training=True)
        loss = binary_cross_loss(y, yhat)

    grad = tape.gradient(loss, siamese_model.trainable_variables)
    opt.apply_gradients(zip(grad, siamese_model.trainable_variables))
    return loss
```

### Training on Google Colab

The `siamese_network_setup.py` script is designed to run on **Google Colab** with a GPU runtime:

```mermaid
flowchart LR
    A["📤 Upload data.zip<br/>to Google Drive"] --> B["📓 Open Notebook<br/>in Google Colab"]
    B --> C["🔧 Mount Drive and<br/>Extract Data"]
    C --> D["🏋️ Train for<br/>50 Epochs"]
    D --> E["💾 Save Model<br/>siamesemodelv2.h5"]
    E --> F["📥 Download to<br/>Project Root"]

    style A fill:#E3F2FD,stroke:#1565C0,color:#000
    style D fill:#FFF3E0,stroke:#E65100,color:#000
    style F fill:#E8F5E9,stroke:#2E7D32,color:#000
```

---

## 📊 Evaluation & Metrics

### Evaluation Flow

```mermaid
flowchart LR
    A["🧪 Test Data"] --> B["🧠 Model Predict"]
    B --> C["📊 Raw Scores<br/>0.0 to 1.0"]
    C --> D["📏 Threshold<br/>> 0.5 = Positive"]
    D --> E["📈 Precision"]
    D --> F["📈 Recall"]
    
    E --> G["🎯 Final<br/>Evaluation Report"]
    F --> G

    style A fill:#E3F2FD,stroke:#1565C0,color:#000
    style E fill:#66BB6A,stroke:#2E7D32,color:#fff
    style F fill:#42A5F5,stroke:#1565C0,color:#fff
    style G fill:#FFA726,stroke:#E65100,color:#fff
```

### Metrics Explained

| Metric | Formula | Meaning |
|---|---|---|
| **Precision** | `TP / (TP + FP)` | Of all faces flagged as "same person," how many actually are? |
| **Recall** | `TP / (TP + FN)` | Of all actual "same person" pairs, how many did we correctly identify? |
| **Detection Threshold** | `prediction > 0.5` | Minimum confidence to count a single comparison as positive |
| **Verification Threshold** | `detections / total > 0.5` | Minimum ratio of positive comparisons to verify identity |

### Confusion Matrix Interpretation

```
                   Predicted
                Positive  Negative
Actual  Positive   TP        FN      ← Recall = TP/(TP+FN)
        Negative   FP        TN
                   ↑
              Precision = TP/(TP+FP)
```

---

## 🖥️ Real-Time Application

### Application State Machine

```mermaid
stateDiagram-v2
    [*] --> Initializing: python faceid.py

    Initializing --> LiveFeed: Model Loaded + Camera Ready
    
    LiveFeed --> Capturing: User Clicks Verify
    LiveFeed --> LiveFeed: update at 33 FPS
    
    Capturing --> Preprocessing: Frame Saved
    Preprocessing --> Inferencing: Images Preprocessed
    Inferencing --> Decision: Batch Predictions Ready
    
    Decision --> Verified: ratio > 0.5
    Decision --> Unverified: ratio ≤ 0.5
    Decision --> EmptyError: No verification images
    
    Verified --> LiveFeed: Display Result
    Unverified --> LiveFeed: Display Result
    EmptyError --> LiveFeed: Show Unverified Empty

    state Initializing {
        [*] --> LoadModel
        LoadModel --> OpenCamera
        OpenCamera --> ScheduleClock
    }
```

### Kivy App Architecture

```mermaid
graph TD
    subgraph "CamApp Kivy Application"
        A["🏗️ build"] --> B["📷 Image Widget<br/>Webcam Feed"]
        A --> C["🔘 Verify Button"]
        A --> D["🏷️ Status Label"]
        A --> E["🧠 Load Model<br/>siamesemodelv2.h5"]
        A --> F["📹 VideoCapture 0"]
        
        F --> G["⏱️ Clock.schedule_interval<br/>update at 33 FPS"]
        G --> H["🔄 Read Frame → Crop → Flip → Texture"]
        H --> B
        
        C -->|on_press| I["verify"]
        I --> J["📸 Capture Frame"]
        J --> K["💾 Save input_image.jpg"]
        K --> L["🔄 Preprocess All Images"]
        L --> M["🧠 Batch Predict"]
        M --> N["📊 Apply Thresholds"]
        N --> D
    end

    style A fill:#4CAF50,stroke:#2E7D32,color:#fff
    style E fill:#AB47BC,stroke:#6A1B9A,color:#fff
    style I fill:#2196F3,stroke:#1565C0,color:#fff
    style M fill:#FF9800,stroke:#E65100,color:#fff
```

### Application Layout

```
┌─────────────────────────────────┐
│                                 │
│        📷 Webcam Feed           │
│        (250×250 crop)           │
│        size_hint=(1, 0.8)       │
│                                 │
├─────────────────────────────────┤
│     [ 🔍 Verify Button ]       │   size_hint=(1, 0.1)
├─────────────────────────────────┤
│   Status: ✅ Verified           │   size_hint=(1, 0.1)
└─────────────────────────────────┘
```

---

## 🔄 Verification Algorithm

### Step-by-Step Verification Process

```mermaid
sequenceDiagram
    participant U as 👤 User
    participant K as 🖥️ Kivy App
    participant C as 📷 Camera
    participant P as 🔧 Preprocessor
    participant M as 🧠 Siamese Model
    participant D as 📊 Decision Engine

    U->>K: Click Verify Button
    K->>C: capture.read()
    C-->>K: Raw Frame
    K->>K: Crop to 250×250
    K->>K: Save as input_image.jpg
    
    K->>P: Preprocess input image
    P-->>K: Tensor 100x100x3
    
    loop For each verification image
        K->>P: Preprocess verification image
        P-->>K: Tensor 100x100x3
    end
    
    K->>K: Stack into batches
    K->>M: Batch predict inputs and validations
    M-->>K: Similarity scores 0.0 to 1.0
    
    K->>D: Apply detection threshold 0.5
    D-->>K: Positive detection count
    K->>D: Calculate verification ratio
    D-->>K: ratio = detections / total
    
    alt ratio > 0.5
        K->>U: ✅ Verified
    else ratio ≤ 0.5
        K->>U: ❌ Unverified
    end
```

### Verification Logic (Pseudocode)

```
FUNCTION verify(input_frame, verification_images):
    scores = []
    
    FOR each ref_image IN verification_images:
        score = model.predict(input_frame, ref_image)
        scores.append(score)
    
    positive_count = COUNT(scores WHERE score > 0.5)    ← Detection Threshold
    ratio = positive_count / TOTAL(verification_images)
    
    IF ratio > 0.5:                                     ← Verification Threshold
        RETURN "✅ VERIFIED"
    ELSE:
        RETURN "❌ UNVERIFIED"
```

> **Optimization Note:** The Kivy app (`faceid.py`) uses **batch inference** — it stacks all input/validation image pairs into tensors and calls `model.predict()` once, rather than looping. This is significantly faster than the loop-based approach in the training script.

---

## 🛠️ Technology Stack

```mermaid
graph LR
    subgraph "🧠 Machine Learning"
        A[TensorFlow 2.x]
        B[Keras API]
        C[Custom Layers]
    end
    
    subgraph "👁️ Computer Vision"
        D[OpenCV 4.x]
        E[Image Preprocessing]
    end
    
    subgraph "🖥️ Application"
        F[Kivy Framework]
        G[Real-Time UI]
    end
    
    subgraph "📊 Data and Viz"
        H[NumPy]
        I[Matplotlib]
        J[HuggingFace Datasets]
    end
    
    subgraph "☁️ Training"
        K[Google Colab]
        L[NVIDIA GPU CUDA]
    end

    A --> B --> C
    D --> E
    F --> G
    K --> L

    style A fill:#FF6F00,stroke:#E65100,color:#fff
    style D fill:#5C3EE8,stroke:#311B92,color:#fff
    style F fill:#40B5A4,stroke:#00695C,color:#fff
    style K fill:#F4B400,stroke:#F57F17,color:#000
```

### Dependencies

| Package | Version | Purpose |
|---|---|---|
| `tensorflow` | 2.x | Deep learning framework — model building, training, inference |
| `opencv-python` | 4.x | Webcam capture, image I/O, frame manipulation |
| `matplotlib` | latest | Training visualization and image plotting |
| `datasets` | latest | HuggingFace datasets library for negative samples (LFW) |
| `tqdm` | latest | Progress bars during data processing |
| `kivy` | latest | Cross-platform desktop GUI framework |
| `numpy` | latest | Numerical operations on predictions and thresholds |

---

## 🚀 Usage Guide

### 1. Collect Training Data

```bash
# Run the data collection script (requires webcam)
python siamese_network_setup.py

# Press 'a' → Save anchor image  (your face, neutral)
# Press 'p' → Save positive image (your face, different angles/lighting)
# Press 'q' → Quit collection
```

> **Tip:** Collect at least **300 anchor** and **300 positive** images with varied lighting, angles, and expressions for robust training.

### 2. Train the Model (Google Colab Recommended)

1. Zip your `data/` folder → `data.zip`
2. Upload to Google Drive
3. Open `siamese_network_setup.py` as a Colab notebook
4. Set runtime to **GPU**
5. Run all cells — training takes ~30–60 min on a T4 GPU
6. Download the resulting `siamesemodelv2.h5` to your project root

### 3. Set Up Verification Images

```bash
# Place 3–5 clear reference photos of your face in:
application_data/verification_images/

# These are the images the model compares against during verification
```

### 4. Run the Application

```bash
python faceid.py
```

### 5. Verify Your Identity

1. Position your face in front of the webcam
2. Click the **"Verify"** button
3. See the result: ✅ **Verified** or ❌ **Unverified**

---

## 📈 Performance

### Model Summary

| Property | Value |
|---|---|
| **Total Parameters** | ~24.8 Million |
| **Trainable Parameters** | ~24.8 Million |
| **Model File Size** | ~155 MB |
| **Input Resolution** | 100×100×3 |
| **Embedding Dimension** | 4096 |
| **Output** | Single sigmoid (0–1) |
| **Inference Speed** | Real-time at 33 FPS |
| **Training Time** | ~30–60 min on T4 GPU |

### Expected Training Progression

```mermaid
xychart-beta
    title "Training Loss Curve (50 Epochs)"
    x-axis "Epoch" [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    y-axis "Binary Cross-Entropy Loss" 0 --> 0.8
    line [0.69, 0.48, 0.35, 0.25, 0.18, 0.14, 0.10, 0.07, 0.05, 0.04, 0.03]
```

```mermaid
xychart-beta
    title "Precision and Recall Over Training"
    x-axis "Epoch" [0, 10, 20, 30, 40, 50]
    y-axis "Score" 0 --> 1
    line "Precision" [0.50, 0.72, 0.85, 0.92, 0.96, 0.99]
    line "Recall" [0.50, 0.68, 0.80, 0.88, 0.94, 0.97]
```

---

## 🐛 Troubleshooting

| Problem | Cause | Solution |
|---|---|---|
| `ModuleNotFoundError: No module named 'kivy'` | Kivy not installed | `pip install kivy` |
| `FileNotFoundError: siamesemodelv2.h5` | Model not in project root | Train the model or download the `.h5` file |
| Black/frozen webcam feed | Wrong camera index | Change `cv2.VideoCapture(0)` to `1` or `2` in `faceid.py` |
| `Unverified (Empty)` result | No verification images | Add photos to `application_data/verification_images/` |
| `CUDA out of memory` during training | GPU memory overflow | Reduce batch size or use `tf.config.experimental.set_memory_growth` |
| Very slow inference | Running on CPU | Install `tensorflow-gpu` or run training on Colab |
| Always shows "Unverified" | Poor reference images or thresholds | Use clearer photos; try lowering thresholds to 0.4 |
| `cv2.error: !_src.empty()` | Camera not accessible | Check webcam connection; close other apps using camera |
| Import error for `L1Dist` | Missing `layers.py` | Ensure `layers.py` is in the same directory as `faceid.py` |

### Adjusting Thresholds

If verification is too strict or too lenient, tune these values in `faceid.py`:

```python
detection_threshold = 0.5     # Lower = more lenient per-image matching
verification_threshold = 0.5  # Lower = fewer positive matches needed
```

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

```mermaid
gitgraph
    commit id: "Initial Commit"
    commit id: "Add training pipeline"
    commit id: "Add Kivy app"
    branch feature/improvements
    commit id: "Optimize batch inference"
    commit id: "Add error handling"
    checkout main
    merge feature/improvements id: "Merge improvements"
    commit id: "Add comprehensive README"
```

### Development Roadmap

- [ ] 🔍 Add face detection (MTCNN/Haar Cascades) before cropping
- [ ] 👥 Support multiple user profiles
- [ ] 🌐 Add a web-based interface (Flask/FastAPI)
- [ ] 🎨 Implement data augmentation for training
- [ ] 📊 Add confidence score display in UI
- [ ] 📱 Export to TFLite for mobile deployment
- [ ] 🔐 Add anti-spoofing (liveness detection)
- [ ] 📹 Support video-based verification
- [ ] ☁️ Deploy as a REST API microservice

---

## 📚 References

1. **Koch, G., Zemel, R., & Salakhutdinov, R.** (2015). *Siamese Neural Networks for One-Shot Image Recognition.* ICML Deep Learning Workshop. — The foundational paper on Siamese networks for one-shot learning.

2. **Chopra, S., Hadsell, R., & LeCun, Y.** (2005). *Learning a Similarity Metric Discriminatively, with Application to Face Verification.* IEEE CVPR. — Early work on contrastive loss for face verification.

3. **Huang, G. B., et al.** (2007). *Labeled Faces in the Wild: A Database for Studying Face Recognition in Unconstrained Environments.* — The LFW dataset used for negative samples.

4. **TensorFlow Documentation** — [keras.io](https://keras.io) | [tensorflow.org](https://www.tensorflow.org)

5. **Kivy Framework** — [kivy.org](https://kivy.org) — Cross-platform Python UI framework.

---

## 📜 License

This project is open source and available under the [MIT License](LICENSE).

---

<p align="center">
  <b>Built with ❤️ using TensorFlow, OpenCV, and Kivy</b><br/>
  <i>A Siamese Neural Network approach to real-time face verification</i>
</p>

<p align="center">
  <a href="https://github.com/Paragiscool/Face_verification_ml_project">
    <img src="https://img.shields.io/badge/⭐_Star_this_repo-if_you_found_it_useful!-yellow?style=for-the-badge" />
  </a>
</p>
