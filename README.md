# Ai/Ml Noise Detector

This project performs audio classification using the UrbanSound8K dataset and a neural network built with TensorFlow. The interface is developed using Streamlit for interactive inference.

## 📦 Features

- Feature extraction using MFCC
- Model training using a CNN
- Interactive Streamlit UI for real-time audio recording and prediction
- Dataset not included due to size (8GB) — must be downloaded separately

## 🚀 Getting Started

### Install Dependencies

Make sure you have Python 3.8+ installed. Then install dependencies:

```bash
pip install -r requirements.txt
```

### 📥 Download Dataset

Download the UrbanSound8K dataset manually from:
[UrbanSound8K](https://urbansounddataset.weebly.com/urbansound8k.html)

Extract it and place it in the root directory of the project as follows:

```
urbansound8k-classifier/
├── UrbanSound8K/
│   ├── audio/
│   └── metadata/
├── preprocess_data.py
├── app.py
```

### 🧹 Preprocess Data

```bash
python preprocess_data.py
```

This step extracts MFCC features and saves them in `X.npy` and `y.npy`.

### 🤖 Train and Use the Model

```bash
streamlit run app.py
```

Use the interface to record audio and classify the sound.

## 🧪 Dependencies

See `requirements.txt`.