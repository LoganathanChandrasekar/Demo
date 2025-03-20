import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from scipy.io import wavfile
from scipy.signal import spectrogram, butter, filtfilt, find_peaks
import os
import librosa

# Streamlit app title
st.title("Cracked Voice Analyzer")

# File uploader to allow user to upload a WAV file
uploaded_file = st.file_uploader("Upload a WAV audio file", type=["wav"])

# Function to analyze cracked voice
def analyze_audio(audio_path):
    # Read the WAV file
    try:
        sr, y = wavfile.read(audio_path)
        # Convert to float for processing
        if y.dtype == np.int16:
            y = y.astype(np.float32) / 32768.0
        elif y.dtype == np.int32:
            y = y.astype(np.float32) / 2147483648.0
    except:
        # Use librosa as fallback
        y, sr = librosa.load(audio_path, sr=None)
    
    # Convert stereo to mono if needed
    if len(y.shape) == 2:
        y_mono = np.mean(y, axis=1)
    else:
        y_mono = y
    
    # Plot waveform
    plt.figure(figsize=(12, 4))
    plt.plot(np.linspace(0, len(y_mono)/sr, len(y_mono)), y_mono)
    plt.title(f"Waveform for {os.path.basename(audio_path)}")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    st.pyplot(plt)
    plt.close()
    
    # Generate spectrogram with better resolution
    n_fft = 2048  # Increase for better frequency resolution
    hop_length = 512  # Decrease for better time resolution
    
    # Calculate spectrogram using librosa for better results
    S = librosa.stft(y_mono, n_fft=n_fft, hop_length=hop_length)
    D = librosa.amplitude_to_db(np.abs(S), ref=np.max)
    
    # Get time and frequency axes
    times = librosa.times_like(D, sr=sr, hop_length=hop_length)
    frequencies = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    
    # Extract pitch (fundamental frequency) using librosa
    pitches, magnitudes = librosa.piptrack(S=S, sr=sr, fmin=75, fmax=1600)
    
    # Get the most prominent pitch at each time frame
    pitch_track = []
    for t in range(magnitudes.shape[1]):
        index = magnitudes[:, t].argmax()
        pitch_track.append(pitches[index, t])
    
    # Convert to numpy array
    pitch_track = np.array(pitch_track)
    
    # Calculate pitch change rate (derivative)
    pitch_changes = np.diff(pitch_track)
    
    # Set thresholds for crack detection
    # These thresholds need to be tuned based on your specific data
    jump_threshold = 80  # Hz per frame - adjust based on your data
    magnitude_threshold = 0.1  # Minimum magnitude for consideration
    
    # Find peaks in the pitch change rate (both positive and negative)
    # These represent sudden jumps in pitch that might indicate a crack
    pos_peaks, _ = find_peaks(pitch_changes, height=jump_threshold, distance=5)
    neg_peaks, _ = find_peaks(-pitch_changes, height=jump_threshold, distance=5)
    
    # Combine all detected peaks
    all_peaks = np.sort(np.concatenate([pos_peaks, neg_peaks]))
    
    # Filter out peaks with low magnitude (likely noise)
    filtered_peaks = []
    for peak in all_peaks:
        if peak < len(magnitudes[0]) and np.max(magnitudes[:, peak]) > magnitude_threshold:
            filtered_peaks.append(peak)
    
    # Convert to numpy array
    filtered_peaks = np.array(filtered_peaks)
    
    # Identify cracked voice sections
    cracked_sections = []
    min_duration = 0.05  # Minimum crack duration in seconds
    max_duration = 0.5   # Maximum crack duration in seconds
    time_per_frame = hop_length / sr
    
    for i in range(len(filtered_peaks)):
        start_idx = filtered_peaks[i]
        if i < len(filtered_peaks) - 1 and filtered_peaks[i+1] - start_idx < max_duration / time_per_frame:
            end_idx = filtered_peaks[i+1]
        else:
            end_idx = start_idx + int(min_duration / time_per_frame)
        
        # Ensure end_idx is within bounds
        end_idx = min(end_idx, len(times) - 1)
        
        # Add to cracked sections
        if end_idx > start_idx:
            cracked_sections.append((start_idx, end_idx))
    
    # Plot the spectrogram and pitch track
    plt.figure(figsize=(12, 8))
    
    # Plot spectrogram
    plt.subplot(2, 1, 1)
    librosa.display.specshow(D, sr=sr, hop_length=hop_length, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    
    # Plot pitch track
    plt.subplot(2, 1, 2)
    plt.plot(times, pitch_track, label='Pitch (Hz)')
    
    # Highlight cracked voice sections
    for start, end in cracked_sections:
        plt.axvspan(times[start], times[end], color="red", alpha=0.3)
    
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title('Pitch Track with Detected Voice Cracks')
    plt.grid(True)
    plt.tight_layout()
    
    st.pyplot(plt)
    plt.close()
    
    # Create a combined visualization
    plt.figure(figsize=(12, 8))
    librosa.display.specshow(D, sr=sr, hop_length=hop_length, x_axis='time', y_axis='log')
    plt.plot(times, pitch_track, color='white', linewidth=1, alpha=0.8, label='Pitch Track')
    
    # Highlight cracked voice sections
    for i, (start, end) in enumerate(cracked_sections):
        plt.axvspan(times[start], times[end], color="red", alpha=0.3, 
                   label="Cracked Voice" if i == 0 else "")
    
    plt.colorbar(format='%+2.0f dB')
    plt.title('Cracked Voice Detection')
    plt.legend()
    plt.tight_layout()
    
    # Save the final plot
    output_image_path = "cracked_voice_result.png"
    plt.savefig(output_image_path)
    
    # Show the result
    st.image(output_image_path, caption="Cracked Voice Detection Result", use_container_width=True)
    
    # Report statistics
    st.subheader("Analysis Results")
    st.write(f"Number of detected voice cracks: {len(cracked_sections)}")
    
    # List all detected cracks with timestamps
    if len(cracked_sections) > 0:
        st.write("Detected voice cracks at:")
        for start, end in cracked_sections:
            st.write(f"- {times[start]:.2f}s to {times[end]:.2f}s (duration: {times[end]-times[start]:.3f}s)")
    else:
        st.write("No voice cracks detected. You may need to adjust detection parameters.")

# Process the uploaded file
if uploaded_file is not None:
    # Save the uploaded file temporarily
    audio_path = "temp_audio.wav"
    with open(audio_path, "wb") as f:
        f.write(uploaded_file.read())
    
    # Run analysis
    st.subheader("Analyzing audio for cracked voice...")
    analyze_audio(audio_path)
    
    # Clean up temporary file
    os.remove(audio_path)
else:
    st.info("Please upload a WAV file to analyze for cracked voice.")

# Add parameter tuning section
st.sidebar.header("Advanced Settings")
st.sidebar.write("Adjust detection parameters if needed:")

# Disclaimer
st.sidebar.markdown("""
**Note:** This tool detects sudden changes in pitch that may indicate a cracked voice. 
It works best with clean vocal recordings and may need parameter adjustments for different voices and recording conditions.
""")