import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import pytz  # For timezone conversion
import time  # for time.sleep to update timestamp

import warnings
warnings.filterwarnings('ignore')

# -----------------------------
# Streamlit App Configuration
# -----------------------------
st.set_page_config(page_title="Streamlit App", layout="wide")
st.markdown("""
    <style>
        .main { background-color: #0e1117; color: white; }
        .css-1d391kg { background-color: #0e1117; }
        .stSidebar { background-color: #1c1e26; }
    </style>
""", unsafe_allow_html=True)

# Sidebar info
st.sidebar.title("🎓 Student Info")
st.sidebar.info("Submitted by: Anant Gupta | Roll No: 2311401280")

# Use pytz to ensure IST time zone (Asia/Kolkata)
ist = pytz.timezone('Asia/Kolkata')  # Set the time zone to IST
timestamp_placeholder = st.sidebar.empty()  # Create a placeholder for the timestamp

# Sidebar Controls
st.sidebar.markdown("## 🔧 Settings")
n_fft = st.sidebar.selectbox("FFT Size", [512, 1024, 2048], index=2)
n_mels = st.sidebar.slider("Number of Mel Filters", min_value=10, max_value=40, value=13)
sr_options = [8000, 16000, 22050]
custom_sr = st.sidebar.selectbox("Sampling Rate", sr_options, index=2)
frame_size = st.sidebar.slider("Frame Length (ms)", 20, 100, 25)
overlap_perc = st.sidebar.slider("Frame Overlap (%)", 0, 90, 50)

# Main Title
st.title("🔊Design of Automatic Speaker Recognition using MFCC Feature Extraction in Python")

# Upload audio file
uploaded_file = st.file_uploader("📁 Upload a .wav file", type=["wav"])

if uploaded_file is not None:
    y, _ = librosa.load(uploaded_file, sr=custom_sr)
    hop_length = int((frame_size * (100 - overlap_perc) / 100) * custom_sr / 1000)
    frame_length = int(frame_size * custom_sr / 1000)

    tab1, tab2, tab3, tab4 = st.tabs([
        "1️⃣ Spectrogram vs MFCC",
        "2️⃣ Frame Size & Overlap",
        "3️⃣ Filter Bank Customization",
        "4️⃣ Sampling Rate Impact"
    ])

    with tab1:
        st.header("1️⃣ Spectrogram vs MFCC Interpretation")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Mel Spectrogram")
            S = librosa.feature.melspectrogram(y=y, sr=custom_sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
            S_dB = librosa.power_to_db(S, ref=np.max)
            fig1, ax1 = plt.subplots(figsize=(10, 3))
            img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=custom_sr, ax=ax1, cmap='magma')
            fig1.colorbar(img, ax=ax1)
            st.pyplot(fig1)

        with col2:
            st.subheader("MFCC Heatmap")
            mfccs = librosa.feature.mfcc(S=S_dB, n_mfcc=13)
            fig2, ax2 = plt.subplots(figsize=(10, 3))
            sns.heatmap(mfccs, ax=ax2, cmap='coolwarm', cbar_kws={'label': 'MFCC Value'})
            ax2.set_title("MFCC Coefficients")
            st.pyplot(fig2)

    with tab2:
        st.header("2️⃣ Frame Size and Overlap Analysis")
        st.write(f"Current Frame Size: {frame_size}ms | Overlap: {overlap_perc}%")
        mfccs_var = librosa.feature.mfcc(y=y, sr=custom_sr, n_mfcc=13, n_fft=n_fft, hop_length=hop_length)
        fig3, ax3 = plt.subplots(figsize=(10, 3))
        sns.heatmap(mfccs_var, ax=ax3, cmap='coolwarm')
        ax3.set_title("MFCC with Dynamic Frame Settings")
        st.pyplot(fig3)

    with tab3:
        st.header("3️⃣ Filter Bank Customization")
        mel_filters = librosa.filters.mel(sr=custom_sr, n_fft=n_fft, n_mels=n_mels)
        fig4, ax4 = plt.subplots(figsize=(10, 3))
        for i in range(n_mels):
            ax4.plot(mel_filters[i])
        ax4.set_title(f"Mel Filter Bank - {n_mels} Filters")
        ax4.set_xlabel("FFT Bins")
        ax4.set_ylabel("Amplitude")
        st.pyplot(fig4)

    with tab4:
        st.header("4️⃣ Sampling Rate Impact on Features")
        fig5, ax5 = plt.subplots(figsize=(10, 3))
        librosa.display.waveshow(y, sr=custom_sr, ax=ax5, color='cyan')
        ax5.set_title(f"Waveform at {custom_sr} Hz")
        ax5.set_xlabel("Time (s)")
        ax5.set_ylabel("Amplitude")
        st.pyplot(fig5)

        fig6, ax6 = plt.subplots(figsize=(10, 3))
        spectrum = np.abs(np.fft.fft(y))**2
        freqs = np.fft.fftfreq(len(spectrum), 1/custom_sr)
        ax6.plot(freqs[:len(freqs)//2], spectrum[:len(freqs)//2], color='orange')
        ax6.set_title(f"Power Spectrum at {custom_sr} Hz")
        ax6.set_xlabel("Frequency (Hz)")
        ax6.set_ylabel("Power")
        st.pyplot(fig6)

        mfccs_sr = librosa.feature.mfcc(y=y, sr=custom_sr, n_mfcc=13)
        fig7, ax7 = plt.subplots(figsize=(10, 3))
        sns.heatmap(mfccs_sr, ax=ax7, cmap='coolwarm')
        ax7.set_title(f"MFCCs at {custom_sr} Hz")
        st.pyplot(fig7)

# Update the timestamp every second
while True:
    india_time = datetime.datetime.now(ist)  # Get time in IST
    timestamp = india_time.strftime("%Y-%m-%d %H:%M:%S")
    timestamp_placeholder.write(f"🕒 App Run At (IST): {timestamp}")
    time.sleep(1)  # Update every second
