import os
import numpy as np
import librosa
import sounddevice as sd
import wavio
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, Flatten, Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# ---------------------- Load Preprocessed Data ----------------------
X = np.load("X.npy")
y = np.load("y.npy")
y = to_categorical(y, num_classes=10)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = X_train.reshape(X_train.shape[0], 40, 1)
X_test = X_test.reshape(X_test.shape[0], 40, 1)

# ---------------------- Model Training ----------------------
MODEL_PATH = "sound_classification_model.h5"

def build_model():
    model = Sequential([
        Input(shape=(40, 1)),
        Conv1D(32, kernel_size=3, activation='relu'),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

if not os.path.exists(MODEL_PATH):
    st.write("Training model...")
    model = build_model()
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
    model.save(MODEL_PATH)
    st.write("Model trained and saved!")

model = load_model(MODEL_PATH)  # Load model only once

# ---------------------- Audio Recording ----------------------
def record_audio(filename="input.wav", duration=3, fs=22050):  
    st.write("Recording...")
    audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    wavio.write(filename, audio_data, fs, sampwidth=2)
    st.write("Recording complete!")
    return filename

# ---------------------- Classify Audio ----------------------
def extract_features(file_path):
    y_audio, sr = librosa.load(file_path, sr=22050)
    mfccs = librosa.feature.mfcc(y=y_audio, sr=sr, n_mfcc=40, hop_length=512)
    return np.mean(mfccs, axis=1)

def classify_audio(file_path):
    features = extract_features(file_path)
    features = features.reshape(1, 40, 1)
    prediction = model.predict(features)

    categories = ["Air Conditioner", "Car Horn", "Children Playing", "Dog Barking", "Drilling", 
                  "Engine Idling", "Gun Shot", "Jackhammer", "Siren", "Street Music"]
    return categories[np.argmax(prediction)]

# ---------------------- Streamlit UI ----------------------
st.title("AI Noise Detector")
st.write("Record audio and classify the sound type!")

if st.button("Record Audio (3 sec)"):
    filename = record_audio()
    st.audio(filename, format='audio/wav')
    st.write("Processing...")
    result = classify_audio(filename)
    st.write(f"Predicted Sound: **{result}**")
