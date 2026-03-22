# Deepfake Detection & AI Analysis

An end-to-end deepfake detection system using **CNN + CNN-LSTM + DeepSeek LLM**.

## How It Works

Supports both **image** and **video** analysis.

| Model | Purpose | Data |
|-------|---------|------|
| **CNN (ResNet18)** | Classifies face images as Real or Fake | 140K real images |
| **CNN-LSTM** | Scans face regions as spatial sequence, finds suspicious areas | Same real images |
| **DeepSeek LLM** | Explains results in plain English, answers questions | API |

### Video Mode
- Extracts frames from uploaded video
- Runs CNN + CNN-LSTM on each frame
- Displays frame-by-frame timeline of fake probability
- Highlights most suspicious and most authentic frames
- Interactive frame browser with attention maps

### Why Two Models?

- **CNN alone**: Looks at the whole image and gives a verdict
- **CNN-LSTM**: Breaks the face into 49 patches (7x7 grid), reads them as a sequence using LSTM, and uses **attention** to highlight which face regions are most suspicious

The LSTM catches things the CNN might miss - like when the left eye looks fine but the right jawline has artifacts. The LSTM connects these distant regions through its sequential memory.

## Project Structure

```
deepfake_detection_project/
├── data/
│   ├── data_loader.py              # Dataset loading + preprocessing
│   └── Dataset/                    # Downloaded from Kaggle (not in git)
│       ├── Train/Real/ & Train/Fake/
│       ├── Validation/Real/ & Validation/Fake/
│       └── Test/Real/ & Test/Fake/
├── models/
│   ├── cnn_model.py                # ResNet18 binary classifier
│   ├── cnn_lstm_model.py           # CNN feature extractor + LSTM spatial analyzer
│   └── llm_assistant.py            # DeepSeek API integration
├── checkpoints/                    # Saved model weights
├── train_cnn.py                    # Train CNN classifier
├── train_lstm.py                   # Train CNN-LSTM model
├── app.py                          # Streamlit web interface
├── requirements.txt
└── .gitignore
```

## Setup

### 1. Create virtual environment

```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate
```

### 2. Install PyTorch

Visit https://pytorch.org/get-started/locally/

```bash
# NVIDIA GPU
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# CPU only
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# macOS Apple Silicon
pip install torch torchvision
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Dataset

### Download from Kaggle

```bash
pip install kaggle
kaggle datasets download -d manjilkarki/deepfake-and-real-images -p data --unzip
```

This gives you ~140K training images (70K real + 70K fake), 40K validation, and 11K test.

## Train Models

```bash
# CNN - quick test with subset (recommended first)
python train_cnn.py --data-dir data/Dataset --epochs 5 --max-per-class 5000

# CNN - full dataset
python train_cnn.py --data-dir data/Dataset --epochs 10

# CNN-LSTM - quick test
python train_lstm.py --data-dir data/Dataset --epochs 5 --max-per-class 5000

# CNN-LSTM - full dataset
python train_lstm.py --data-dir data/Dataset --epochs 10
```

## Run the App

```bash
streamlit run app.py
```

### DeepSeek API (for LLM features)

```bash
# Windows
set DEEPSEEK_API_KEY=your_key_here

# Linux/macOS
export DEEPSEEK_API_KEY=your_key_here
```

## Disclaimer

No AI detector is 100% accurate. Deepfake technology is constantly evolving. This tool is for educational and research purposes only.
