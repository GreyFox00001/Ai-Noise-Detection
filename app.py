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

st.set_page_config(
    page_title="Accessible Environmental Sound Detector",
    page_icon="🔊",
    layout="wide"
)

# ---------------------- Load Preprocessed Data ----------------------
X = np.load("X.npy")
y = np.load("y.npy")
y = to_categorical(y, num_classes=10)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
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
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

if not os.path.exists(MODEL_PATH):
    st.write("Training model...")
    model = build_model()
    model.fit(
        X_train,
        y_train,
        epochs=10,
        batch_size=32,
        validation_data=(X_test, y_test)
    )
    model.save(MODEL_PATH)
    st.write("Model trained and saved!")

model = load_model(MODEL_PATH)

# ---------------------- Audio Recording ----------------------
def record_audio(filename="input.wav", duration=3, fs=22050):
    st.write("Recording...")
    audio_data = sd.rec(
        int(duration * fs),
        samplerate=fs,
        channels=1,
        dtype='int16'
    )
    sd.wait()
    wavio.write(filename, audio_data, fs, sampwidth=2)
    st.write("Recording complete!")
    return filename

# ---------------------- Classify Audio ----------------------
def extract_features(file_path):
    y_audio, sr = librosa.load(file_path, sr=22050)
    mfccs = librosa.feature.mfcc(
        y=y_audio,
        sr=sr,
        n_mfcc=40,
        hop_length=512
    )
    return np.mean(mfccs, axis=1)

def classify_audio(file_path):
    features = extract_features(file_path)
    features = features.reshape(1, 40, 1)
    prediction = model.predict(features)

    categories = [
        "Air Conditioner",
        "Car Horn",
        "Children Playing",
        "Dog Barking",
        "Drilling",
        "Engine Idling",
        "Gun Shot",
        "Jackhammer",
        "Siren",
        "Street Music"
    ]

    return categories[np.argmax(prediction)]

# ---------------------- Streamlit UI ----------------------
CLASSES = [
    "Air Conditioner",
    "Car Horn",
    "Children Playing",
    "Dog Barking",
    "Drilling",
    "Engine Idling",
    "Gun Shot",
    "Jackhammer",
    "Siren",
    "Street Music"
]

ICON_MAP = {
    "Air Conditioner": "❄️",
    "Car Horn": "🚗",
    "Children Playing": "🧒",
    "Dog Barking": "🐕",
    "Drilling": "🛠️",
    "Engine Idling": "🚙",
    "Gun Shot": "🔫",
    "Jackhammer": "🚧",
    "Siren": "🚨",
    "Street Music": "🎵"
}

DANGER_CLASSES = {"Siren", "Gun Shot", "Car Horn"}
CAUTION_CLASSES = {"Dog Barking", "Drilling", "Jackhammer", "Engine Idling"}
SAFE_CLASSES = {"Children Playing", "Street Music", "Air Conditioner"}

def apply_custom_styles():
    st.markdown(
        """
        <style>
            .main-title {
                font-size: 2.8rem;
                font-weight: 800;
                margin-bottom: 0.2rem;
            }

            .subtitle {
                font-size: 1.2rem;
                color: #555;
                margin-bottom: 1.5rem;
            }

            .listening-status {
                font-size: 2rem;
                font-weight: 900;
                color: white;
                background-color: #b00020;
                padding: 1rem;
                border-radius: 0.5rem;
                text-align: center;
                animation: pulse 1s infinite;
            }

            .prediction-text {
                font-size: 2.4rem;
                font-weight: 900;
                text-align: center;
                margin: 0.5rem 0;
            }

            .helper-text {
                font-size: 1rem;
                color: #444;
                text-align: center;
            }

            @keyframes pulse {
                0% { opacity: 1; }
                50% { opacity: 0.65; }
                100% { opacity: 1; }
            }
        </style>
        """,
        unsafe_allow_html=True
    )

def initialize_session_state():
    if "detection_history" not in st.session_state:
        st.session_state.detection_history = []

def display_prediction_alert(predicted_sound):
    sound_icon = ICON_MAP.get(predicted_sound, "🔊")
    prediction_display_text = f"{sound_icon} {predicted_sound}"

    if predicted_sound in DANGER_CLASSES:
        st.error(f"### {prediction_display_text}")
        st.markdown(
            f'<div class="prediction-text">{prediction_display_text}</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            '<div class="helper-text">Emergency or high-priority sound detected nearby.</div>',
            unsafe_allow_html=True
        )

    elif predicted_sound in CAUTION_CLASSES:
        st.warning(f"### {prediction_display_text}")
        st.markdown(
            f'<div class="prediction-text">{prediction_display_text}</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            '<div class="helper-text">Caution sound detected. Please check your surroundings.</div>',
            unsafe_allow_html=True
        )

    elif predicted_sound in SAFE_CLASSES:
        st.success(f"### {prediction_display_text}")
        st.markdown(
            f'<div class="prediction-text">{prediction_display_text}</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            '<div class="helper-text">Neutral environmental sound detected.</div>',
            unsafe_allow_html=True
        )

    else:
        st.info(f"### {prediction_display_text}")

def display_detection_history():
    history_col, class_col = st.columns([2, 1])

    with history_col:
        st.subheader("Recent Detected Sounds")

        if st.session_state.detection_history:
            for history_index, detected_item in enumerate(
                st.session_state.detection_history,
                start=1
            ):
                st.markdown(f"**{history_index}.** {detected_item}")
        else:
            st.info("No sounds detected yet.")

    with class_col:
        st.subheader("Alert Categories")
        st.error("🚨 Emergency: Siren, Gun Shot, Car Horn")
        st.warning("⚠️ Caution: Dog Barking, Drilling, Jackhammer, Engine Idling")
        st.success("✅ Neutral: Children Playing, Street Music, Air Conditioner")

def display_home_page():
    st.markdown(
        '<div class="main-title">Accessible Environmental Sound Detector</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="subtitle">AI-powered visual sound alerts for deaf and hard-of-hearing users.</div>',
        unsafe_allow_html=True
    )

    st.subheader("Welcome")
    st.write(
        "This application detects common environmental sounds and converts them "
        "into clear visual alerts. It is designed to support accessibility for "
        "hearing-impaired users by making nearby sounds easier to notice on screen."
    )

    st.subheader("How to Use")
    st.markdown(
        """
        1. Click the **Record Audio** button.
        2. Wait while the app listens for **3 seconds**.
        3. View the large visual alert showing the detected sound.
        4. Check the recent detection history for sounds you may have missed.
        """
    )

    st.divider()

    left_col, center_col, right_col = st.columns([1, 2, 1])

    with center_col:
        st.markdown("### Main Detection Tool")
        st.markdown("Tap the button below to listen for nearby environmental sound.")

        record_button_pressed = st.button(
            "🔴 Record Audio",
            use_container_width=True,
            type="primary"
        )

        status_placeholder = st.empty()

        if record_button_pressed:
            status_placeholder.markdown(
                '<div class="listening-status">🔴 LISTENING NOW... (3 seconds)</div>',
                unsafe_allow_html=True
            )

            user_audio_file = record_audio()

            status_placeholder.markdown(
                '<div class="listening-status">🟡 PROCESSING AUDIO...</div>',
                unsafe_allow_html=True
            )

            predicted_sound = classify_audio(user_audio_file)
            predicted_sound_icon = ICON_MAP.get(predicted_sound, "🔊")

            history_item = f"{predicted_sound_icon} {predicted_sound}"
            st.session_state.detection_history.insert(0, history_item)
            st.session_state.detection_history = st.session_state.detection_history[:5]

            status_placeholder.empty()

            st.audio(user_audio_file, format="audio/wav")
            display_prediction_alert(predicted_sound)

    st.divider()
    display_detection_history()

def display_about_page():
    st.title("About Us")
    st.write(
        "Our mission is to bridge the accessibility gap for the deaf and "
        "hard-of-hearing community using machine learning."
    )
    st.write(
        "This project uses environmental sound classification to provide clear, "
        "visual feedback about nearby sounds such as sirens, car horns, drilling, "
        "dog barking, and other common urban noises."
    )

def display_contact_page():
    st.title("Contact Us")
    st.write(
        "Have feedback, questions, or suggestions? Send us a message using the form below."
    )

    contact_name = st.text_input("Name")
    contact_email = st.text_input("Email")
    contact_message = st.text_area("Message")

    submit_button_pressed = st.button("Submit")

    if submit_button_pressed:
        if contact_name and contact_email and contact_message:
            st.success("Thank you for your message. We will get back to you soon.")
        else:
            st.warning("Please fill in your name, email, and message before submitting.")

def main():
    apply_custom_styles()
    initialize_session_state()

    st.sidebar.title("Navigation")

    selected_page = st.sidebar.radio(
        "Go to",
        ["Home", "About Us", "Contact Us"]
    )

    if selected_page == "Home":
        display_home_page()

    elif selected_page == "About Us":
        display_about_page()

    elif selected_page == "Contact Us":
        display_contact_page()

if __name__ == "__main__":
    main()
