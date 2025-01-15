import streamlit as st
import joblib
import numpy as np
import librosa
import matplotlib.pyplot as plt
import pandas as pd

MODEL_PATH = "models/rf_model.joblib"
LE_CLASSES_PATH = "Data/le_classes.npy"
max_len = 173

model = joblib.load(MODEL_PATH)
le_classes = np.load(LE_CLASSES_PATH, allow_pickle=True)


def plot_waveform(audio_data, sr):
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(audio_data, sr=sr, alpha=0.8)
    plt.title("Waveform")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    return plt

st.title("Speech Emotion Recognition")


uploaded_file = st.file_uploader("Choose an Audio file")
st.write(' ')
st.subheader('OR')
st.write(' ')
uploaded_file_recorded = st.audio_input("Record your Audio")

if uploaded_file:
    st.audio(uploaded_file)
    signal, sr = librosa.load(uploaded_file,sr=22050)
    signal, _ = librosa.effects.trim(signal, top_db=20)
    fig = plot_waveform(signal, sr)
    if len(signal) > max_len * sr:
        signal = signal[:max_len * sr]
    else:
        pad_width = max_len * sr - len(signal)
        signal = np.pad(signal, (0, pad_width), 'constant')
    mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=40)
    mfccs_mean = np.mean(mfccs.T, axis=0)

    mfccs_mean = mfccs_mean.reshape(1, -1)
    prediction = model.predict(mfccs_mean)
    emotion = le_classes[prediction[0]]

    st.subheader(f"Emotion : {emotion}")

    
    st.pyplot(fig)

elif uploaded_file_recorded:
    signal, sr = librosa.load(uploaded_file_recorded,sr=22050)
    signal, _ = librosa.effects.trim(signal, top_db=20)
    fig = plot_waveform(signal, sr)
    if len(signal) > max_len * sr:
        signal = signal[:max_len * sr]
    else:
        pad_width = max_len * sr - len(signal)
        signal = np.pad(signal, (0, pad_width), 'constant')
    mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=40)
    mfccs_mean = np.mean(mfccs.T, axis=0)

    mfccs_mean = mfccs_mean.reshape(1, -1)
    prediction = model.predict(mfccs_mean)
    emotion = le_classes[prediction[0]]

    st.subheader(f"Emotion : {emotion}")

    st.pyplot(fig)
    
else:
    st.write('File not found!')

