import streamlit as st
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import io
import os
import subprocess
import sys
import zipfile
import speech_recognition as sr  # Import SpeechRecognition
import soundfile as sf  # Import soundfile for wav conversion
from io import BytesIO  # Import BytesIO
import csv

# PDF Report Generation Libraries
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.platypus import Image, Paragraph, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch


# --- Audio Processing Functions ---
def detect_cracked_voice(audio_data, sample_rate, threshold=0.8, frame_length=2048, hop_length=512):
    """Detects cracked voice based on sudden pitch changes."""
    pitches, magnitudes = librosa.piptrack(y=audio_data, sr=sample_rate, hop_length=hop_length, n_fft=frame_length)
    pitch_changes = np.diff(np.argmax(pitches, axis=0))  # Find indices of max pitches and get the difference.
    change_points = np.where(np.abs(pitch_changes) > threshold)[0]  # Absolute value used because direction of change matters.

    # Convert frame indices to time segments
    segments = []
    for frame in change_points:
        time = librosa.frames_to_time(frame, sr=sample_rate, hop_length=hop_length)
        end_time = librosa.frames_to_time(frame + 1, sr=sample_rate, hop_length=hop_length)  # small segment length
        segments.append((time, end_time))
    return segments

def detect_frame_dropping(audio_data, sample_rate, threshold=0.02, frame_length=1024, hop_length=512, cutoff_freq=22000, higherthan_threshold_num=4):
    """Detects frame drops by analyzing high-frequency energy."""
    segments = []
    audio_length = len(audio_data)
    if audio_length < frame_length:
        return segments  # Audio too short to analyze

    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio_data, n_fft=frame_length, hop_length=hop_length)), ref=np.max)
    frequencies = librosa.fft_frequencies(sr=sample_rate, n_fft=frame_length)
    high_freq_mask = frequencies >= cutoff_freq
    D_high_freq = D[high_freq_mask, :]
    high_freq_energy = np.sum(librosa.db_to_amplitude(D_high_freq), axis=0)

    def detect_anomalies(energy, threshold):
        anomalies = []
        drop_energy_info = []
        i = 0
        while i < len(energy):
            if energy[i] > threshold:
                start = i
                while i < len(energy) and energy[i] > threshold:
                    i += 1
                length = i - start
                if length == 1 or length == 2:
                    anomalies.append(('drop', start, min(i, len(energy) - 1)))  # 修复：确保索引不超出范围
                    drop_energy_info.append(f"{np.max(energy[start:i]):.2f}")
                elif length >= higherthan_threshold_num:
                    anomalies.append(('noise', start, min(i, len(energy) - 1)))  # 修复：确保索引不超出范围
            else:
                i += 1
        return anomalies, drop_energy_info

    anomalies, _ = detect_anomalies(high_freq_energy, threshold)

    time_axis = np.linspace(0, audio_length / sample_rate, len(high_freq_energy))
    for anomaly_type, start, end in anomalies:
        start_time = time_axis[start] if start < len(time_axis) else time_axis[-1]
        end_time = time_axis[end] if end < len(time_axis) else time_axis[-1]
        if anomaly_type == 'drop':
            segments.append((start_time, end_time))
    return segments

def detect_audio_popping(audio_data, sample_rate, threshold=2.0, frame_length=2048, hop_length=512):
    """Detects audio popping by identifying sudden spikes in the waveform."""
    # Calculate the short-time Fourier transform (STFT)
    stft = librosa.stft(audio_data, n_fft=frame_length, hop_length=hop_length)
    amplitude = np.abs(stft)

    # Calculate the mean amplitude
    mean_amplitude = np.mean(amplitude)

    # Find spikes significantly above the mean
    spike_points = np.where(amplitude > mean_amplitude * threshold)

    # Convert frame indices to time segments
    segments = []
    for frame in np.unique(spike_points[1]):  # Use unique frame indices
        time = librosa.frames_to_time(frame, sr=sample_rate, hop_length=hop_length)
        end_time = librosa.frames_to_time(frame + 1, sr=sample_rate, hop_length=hop_length)
        segments.append((time, end_time))
    return segments


def calculate_percentage_clean(issue_segments, total_duration):
    """Calculates the percentage of audio *without* issues."""
    total_issue_duration = 0
    for issue_type, segments in issue_segments.items():
        for start, end in segments:
            total_issue_duration += (end - start)
    clean_duration = total_duration - total_issue_duration
    percentage_clean = (clean_duration / total_duration) * 100 if total_duration > 0 else 100.0  # Avoid division by zero
    return max(0.0, min(100.0, percentage_clean))  # Ensure within 0-100 range.


# --- PDF Generation Function ---
def generate_pdf_report(file_name, analysis_report, waveform_image_buffer):
    pdf_buffer = BytesIO()
    c = canvas.Canvas(pdf_buffer, pagesize=letter)
    c.setTitle(f"Audio Analysis Report - {file_name}")

    # --- Styles ---
    styles = getSampleStyleSheet()
    title_style = styles['h1']
    title_style.alignment = 1  # TA_CENTER
    label_style = styles['Normal']
    label_style.fontName = 'Helvetica-Bold'
    value_style = styles['Normal']
    paragraph_style = styles['Normal']
    paragraph_style.wordWrap = True
    paragraph_style.leading = 14

    # --- Layout ---
    margin = inch
    text_width = 6 * inch
    current_y = 10.5 * inch
    c.leading = 14 #Set default leading for canvas

    def draw_text(text, style, y_position):
        p = Paragraph(text, style)
        p.wrapOn(c, text_width, 1 * inch)  # Adjust wrapOn's height for multi-line text
        p.drawOn(c, margin, y_position - p.height)
        return y_position - p.height

    # Title
    current_y = draw_text(f"Audio Analysis Report: {file_name}", title_style, current_y)
    current_y -= 0.25 * inch
    c.line(margin, current_y, margin + text_width, current_y)

    # Waveform Image
    waveform_image = Image(waveform_image_buffer, width=5 * inch, height=1.5 * inch)
    waveform_image.drawOn(c, margin, current_y - 1.5 * inch)
    current_y -= (1.5 * inch + 0.25 * inch)  # Account for image height and spacing
    c.line(margin, current_y, margin + text_width, current_y)

    current_y = draw_text("Analysis Results:", label_style, current_y)
    c.line(margin, current_y, margin + text_width, current_y)
    current_y -= 0.25 * inch

    # Analysis Report Details
    for key, value in analysis_report.items():
        label_text = f"{key}:"
        current_y = draw_text(label_text, label_style, current_y)
        c.line(margin, current_y, margin + text_width, current_y)

        if key == 'Speech-to-Text':
            value_text = value
            current_y = draw_text(value_text, paragraph_style, current_y) #Use paragraph here
        elif "Quality" in key or "Integrity" in key or "Clarity" in key:
            value_text = value
            current_y = draw_text(value_text, value_style, current_y)

        else:
            value_text = value
            current_y = draw_text(value_text, value_style, current_y)

        current_y -= 0.25 * inch  # Spacing after each key-value pair
    c.line(margin, current_y, margin + text_width, current_y)

    c.save()
    pdf_buffer.seek(0)
    return pdf_buffer

#Combined PDF
def generate_combined_pdf_report(analysis_results_list):
    pdf_buffer = io.BytesIO()
    c = canvas.Canvas(pdf_buffer, pagesize=letter)
    c.setTitle("Combined Audio Analysis Report")

    # --- Styles --- (Define styles here to ensure they are available)
    styles = getSampleStyleSheet()
    title_style = styles['h1']
    title_style.alignment = 1  # TA_CENTER
    label_style = styles['Normal']
    label_style.fontName = 'Helvetica-Bold'
    value_style = styles['Normal']
    paragraph_style = styles['Normal']
    paragraph_style.wordWrap = True
    paragraph_style.leading = 14  # Line Spacing

    # --- Layout ---
    margin = inch
    text_width = 6 * inch
    c.leading = 14

    def draw_text(text, style, y_position):
        p = Paragraph(text, style)
        p.wrapOn(c, text_width, 1 * inch)  # Adjust wrapOn's height for multi-line text
        p.drawOn(c, margin, y_position - p.height)
        return y_position - p.height

    for file_data in analysis_results_list:
        file_name = file_data["file_name"]
        analysis_report = file_data["report"]
        waveform_image_buffer = file_data["waveform_image_buffer"]

        c.showPage() # Every File start with showpage,
        current_y = 10.5 * inch

        # Title
        current_y = draw_text(f"Audio Analysis Report: {file_name}", title_style, current_y)
        current_y -= 0.25 * inch
        c.line(margin, current_y, margin + text_width, current_y) #Horizontal Line


        # Waveform Image
        waveform_image = Image(waveform_image_buffer, width=5 * inch, height=1.5 * inch)
        waveform_image.drawOn(c, margin, current_y - 1.5 * inch)
        current_y -= (1.5 * inch + 0.25 * inch)  # Account for image height and spacing
        c.line(margin, current_y, margin + text_width, current_y)


        current_y = draw_text("Analysis Results:", label_style, current_y)
        c.line(margin, current_y, margin + text_width, current_y)
        current_y -= 0.25 * inch #Give room

        # Analysis Report Details
        for key, value in analysis_report.items():
            label_text = f"{key}:"
            current_y = draw_text(label_text, label_style, current_y)
            c.line(margin, current_y, margin + text_width, current_y)

            if key == 'Speech-to-Text':
                value_text = value # For multiline text, use paragraph
                current_y = draw_text(value_text, paragraph_style, current_y)
            elif "Quality" in key or "Integrity" in key or "Clarity" in key:
                value_text = value #For Percentage
                current_y = draw_text(value_text, value_style, current_y) #Percentage format

            else:
                value_text = value
                current_y = draw_text(value_text, value_style, current_y) #Normal formatting
            current_y -= 0.25 * inch  # Space down
        c.line(margin, current_y, margin + text_width, current_y)

    c.save()
    pdf_buffer.seek(0)
    return pdf_buffer

def generate_csv_report(file_name, analysis_report):
    csv_buffer = io.StringIO()
    writer = csv.writer(csv_buffer)

    # Write header
    writer.writerow(["Metric", "Value"])

    # Write data
    for key, value in analysis_report.items():
        writer.writerow([key, value])

    csv_buffer.seek(0)
    return csv_buffer.getvalue()

def generate_combined_csv_report(analysis_results_list):
    csv_buffer = io.StringIO()
    writer = csv.writer(csv_buffer)

    # Write header
    header = ["File Name", "Metric", "Value"]
    writer.writerow(header)

    # Write data
    for file_data in analysis_results_list:
        file_name = file_data["file_name"]
        analysis_report = file_data["report"]
        for key, value in analysis_report.items():
            writer.writerow([file_name, key, value])

    csv_buffer.seek(0)
    return csv_buffer.getvalue()


# Function to install packages if not already installed
def install_packages():
    packages_to_install = [
        "librosa",
        "numpy",
        "matplotlib",
        "streamlit",
        "scipy",
        "SpeechRecognition",
        "soundfile",
        "reportlab",
    ]
    installed_packages = [
        pkg.split("==")[0] for pkg in sys.modules.keys() if pkg in packages_to_install
    ]
    for package in packages_to_install:
        if package not in installed_packages:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"Installed {package}")


# Run package installation at the beginning
install_packages()

st.title("Audio Analytic Tool")

uploaded_files = st.file_uploader(
    "Drag and Drop Audio Files or Folder", type=["wav", "mp3"], accept_multiple_files=True
)

# Initialize session state variables
if "analysis_completed" not in st.session_state:
    st.session_state["analysis_completed"] = False
if "uploaded_file_names" not in st.session_state:
    st.session_state["uploaded_file_names"] = []

if uploaded_files:
    new_file_names = [file.name for file in uploaded_files if file.name not in st.session_state["uploaded_file_names"]]
    if new_file_names:
        # Reset analysis-related session state only when new files are uploaded
        st.session_state["analysis_completed"] = False
        st.session_state["analysis_results"] = {}
        st.session_state["audio_data"] = {}
        st.session_state["sample_rates"] = {}
        st.session_state["fixed_audio_data"] = {}
        st.session_state["fix_reports"] = {}
        st.session_state["uploaded_file_names"] = [file.name for file in uploaded_files]

    for uploaded_file in uploaded_files:
        file_name = uploaded_file.name
        if file_name not in st.session_state["analysis_results"]:
            st.session_state["analysis_results"][file_name] = {}
        if file_name not in st.session_state["fix_reports"]:
            st.session_state["fix_reports"][file_name] = {}

        try:
            audio_bytes = uploaded_file.read()
            audio_data, sample_rate = librosa.load(
                io.BytesIO(audio_bytes), sr=None
            )  # Load with original sample rate
            st.session_state["audio_data"][file_name] = audio_data
            st.session_state["sample_rates"][file_name] = sample_rate

            st.audio(audio_bytes, format="audio/wav")  # Streamlit audio player

            fig, ax = plt.subplots(figsize=(10, 2))
            librosa.display.waveshow(audio_data, sr=sample_rate, ax=ax)
            ax.set_title(f"Waveform for {file_name}")
            st.pyplot(fig)
            plt.close(fig)  # Close the plot to free memory

        except Exception as e:
            st.error(f"Error loading {file_name}: {e}")
            st.session_state["analysis_results"][file_name]["error"] = str(e)

    if st.button("Analyze Audio and Highlight Issues"):  # Changed button text
        st.session_state[
            "analysis_completed"
        ] = True  # Set analysis completion status to True when analysis starts
        analysis_results_list = (
            []
        )  # Clear the list before processing files for combined PDF

        for uploaded_file in uploaded_files:
            file_name = uploaded_file.name
            if "error" in st.session_state["analysis_results"][file_name]:
                continue  # Skip files with loading errors

            audio_data = st.session_state["audio_data"][file_name]
            sample_rate = st.session_state["sample_rates"][file_name]
            analysis_report = {}  # Initialize analysis_report here for each file
            issue_segments = {}

            st.subheader(f"Analysis Report for {file_name}")

            # --- Issue Detection (Using new functions) ---

            try:
                issue_segments["Cracked Voice"] = detect_cracked_voice(
                    audio_data, sample_rate
                )
                issue_segments["Frame Dropping"] = detect_frame_dropping(
                    audio_data, sample_rate
                )
                issue_segments["Audio Popping"] = detect_audio_popping(
                    audio_data, sample_rate
                )

                analysis_report["Frame Dropping"] = (
                    "Detected" if issue_segments["Frame Dropping"] else "Not Detected"
                )
                analysis_report["Audio Popping"] = (
                    "Detected" if issue_segments["Audio Popping"] else "Not Detected"
                )
                analysis_report["Cracked Voice"] = (
                    "Detected" if issue_segments["Cracked Voice"] else "Not Detected"
                )  # From example output, now detected because of segments

                if issue_segments["Frame Dropping"]:
                    st.warning(
                        f"Frame Dropping: Detected at {', '.join([f'{start:.2f}-{end:.2f}s' for start, end in issue_segments['Frame Dropping']])}"
                    )  # Adjusted to show start times
                if issue_segments["Audio Popping"]:
                    st.warning(
                        f"Audio Popping: Detected at {', '.join([f'{start:.2f}-{end:.2f}s' for start, end in issue_segments['Audio Popping']])}"
                    )  # Adjusted to show start times
                if issue_segments["Cracked Voice"]:
                    st.warning(
                        f"Cracked Voice: Detected at {', '.join([f'{start:.2f}-{end:.2f}s' for start, end in issue_segments['Cracked Voice']])}"
                    )
                else:
                    st.success("Cracked Voice: Not Detected")

            except Exception as issue_detection_err:
                st.error(f"Error during issue detection: {issue_detection_err}")
                analysis_report["Issue Detection Error"] = str(issue_detection_err)
                issue_segments = {}  # Ensure issue_segments is empty on error


            # --- Score Calculation  ---

            try:
                total_duration = librosa.get_duration(
                    y=audio_data, sr=sample_rate
                )  # Get the total duration

                cracked_voice_clean_percentage = calculate_percentage_clean(
                    {"Cracked Voice": issue_segments.get("Cracked Voice", [])},
                    total_duration,
                )
                frame_dropping_clean_percentage = calculate_percentage_clean(
                    {"Frame Dropping": issue_segments.get("Frame Dropping", [])},
                    total_duration,
                )
                audio_popping_clean_percentage = calculate_percentage_clean(
                    {"Audio Popping": issue_segments.get("Audio Popping", [])},
                    total_duration,
                )

                cracked_voice_issue_percentage = 100 - cracked_voice_clean_percentage
                frame_dropping_issue_percentage = 100 - frame_dropping_clean_percentage
                audio_popping_issue_percentage = 100 - audio_popping_clean_percentage

                st.metric(
                    "Cracked Voice Quality",
                    f"{cracked_voice_clean_percentage:.2f}% Clean, {cracked_voice_issue_percentage:.2f}% Issues",
                )
                analysis_report[
                    "Cracked Voice Quality"
                ] = f"{cracked_voice_clean_percentage:.2f}% Clean, {cracked_voice_issue_percentage:.2f}% Issues"
                st.metric(
                    "Frame Integrity",
                    f"{frame_dropping_clean_percentage:.2f}% Clean, {frame_dropping_issue_percentage:.2f}% Issues",
                )
                analysis_report[
                    "Frame Integrity"
                ] = f"{frame_dropping_clean_percentage:.2f}% Clean, {frame_dropping_issue_percentage:.2f}% Issues"
                st.metric(
                    "Audio Clarity",
                    f"{audio_popping_clean_percentage:.2f}% Clean, {audio_popping_issue_percentage:.2f}% Issues",
                )
                analysis_report[
                    "Audio Clarity"
                ] = f"{audio_popping_clean_percentage:.2f}% Clean, {audio_popping_issue_percentage:.2f}% Issues"
            except Exception as score_err:
                st.error(f"Error during score calculation: {score_err}")
                analysis_report["Score Calculation Error"] = str(score_err)


            # 4. Speech-to-Text (using SpeechRecognition Library)
            try:
                r = sr.Recognizer()
                # Convert audio_bytes to wav format using soundfile in memory
                wav_buffer = io.BytesIO()
                sf.write(wav_buffer, audio_data, sample_rate, format='WAV')
                wav_bytes = wav_buffer.getvalue() # Get the wav bytes
                wav_buffer.seek(0) # Reset buffer to read from beginning for SpeechRecognition

                with sr.AudioFile(wav_buffer) as source: # Use wav_buffer instead of io.BytesIO(audio_bytes)
                    audio_text = r.record(source)  # read the entire audio file

                try:
                    text = r.recognize_google(
                        audio_text
                    )  # Using google speech recognition. For different accents, you might need to specify the language
                    analysis_report["Speech-to-Text"] = text
                    st.subheader("Speech-to-Text Transcription")
                    st.write(text)
                except sr.UnknownValueError:
                    analysis_report["Speech-to-Text"] = (
                        "Speech Recognition could not understand audio"
                    )
                    st.subheader("Speech-to-Text Transcription")
                    st.write("Speech Recognition could not understand audio")
                except sr.RequestError as e:
                    analysis_report["Speech-to-Text"] = (
                        f"Could not request results from Speech Recognition service; {e}"
                    )
                    st.subheader("Speech-to-Text Transcription")
                    st.write(
                        f"Could not request results from Speech Recognition service; {e}"
                    )
                except sr.WaitTimeoutError as e:
                    analysis_report[
                        "Speech-to-Text"
                    ] = f"Speech Recognition request timed out; {e}"
                    st.subheader("Speech-to-Text Transcription")
                    st.write(f"Speech Recognition request timed out; {e}")

            except Exception as e:
                analysis_report["Speech-to-Text"] = f"Error during Speech-to-Text: {e}"
                st.subheader("Speech-to-Text Transcription")
                st.write(f"Error during Speech-to-Text: {e}")

            st.session_state["analysis_results"][file_name]["report"] = analysis_report

            # Visualization with issue highlighting
            try:
                fig, ax = plt.subplots(figsize=(10, 2))
                librosa.display.waveshow(audio_data, sr=sample_rate, ax=ax)
                ax.set_title(f"Waveform with Issue Highlighting for {file_name}")

                colors = {
                    "Cracked Voice": "red",
                    "Frame Dropping": "yellow",
                    "Audio Popping": "green",
                }  # Define colors for issues

                for issue_type, segments in issue_segments.items():
                    for start_time, end_time in segments:
                        ax.axvspan(
                            start_time,
                            end_time,
                            color=colors.get(issue_type, "gray"),
                            alpha=0.3,
                            label=issue_type
                            if (start_time, end_time) == segments[0]
                            else None,
                        )  # Label only once per type

                # Create legend only if there are issues to display
                if issue_segments and any(
                    issue_segments.values()
                ):  # Check if issue_segments is not empty and has values
                    handles, labels = ax.get_legend_handles_labels()
                    unique_labels = []
                    unique_handles = []
                    for handle, label in zip(handles, labels):
                        if label not in unique_labels:
                            unique_labels.append(label)
                            unique_handles.append(handle)
                    ax.legend(
                        unique_handles, unique_labels
                    )  # Display legend for issue types

                st.pyplot(fig)
                image_buffer = io.BytesIO()  # Capture plot to buffer for PDF
                fig.savefig(image_buffer, format="png")
                plt.close(fig)  # Close plot after saving to buffer

            except Exception as plot_err:
                st.error(f"Error during visualization: {plot_err}")
                analysis_report["Visualization Error"] = str(plot_err)
                image_buffer = io.BytesIO()  # Create an empty image buffer


            st.subheader(f"Analysis Details for {file_name}")
            for key, value in analysis_report.items():
                st.write(f"- **{key}**: {value}")

            file_data = {
                "file_name": file_name,
                "report": analysis_report,
                "waveform_image_buffer": image_buffer,
            }
            analysis_results_list.append(file_data)


            # PDF Report Generation
            try:
                pdf_report_bytes = generate_pdf_report(
                    file_name, analysis_report, image_buffer
                )
                st.session_state["analysis_results"][file_name][
                    "pdf_report"
                ] = pdf_report_bytes  # Store pdf report in session state

            except Exception as pdf_err:
                st.error(f"Error generating PDF report for {file_name}: {pdf_err}")
                analysis_report["PDF Generation Error"] = str(pdf_err)
                st.session_state["analysis_results"][file_name][
                    "pdf_error"
                ] = str(pdf_err)

            # CSV Report Generation
            try:
                csv_report_string = generate_csv_report(file_name, analysis_report)
                st.session_state["analysis_results"][file_name][
                    "csv_report"
                ] = csv_report_string  # Store csv report in session state
            except Exception as csv_err:
                st.error(f"Error generating CSV report for {file_name}: {csv_err}")
                analysis_report["CSV Generation Error"] = str(csv_err)
                st.session_state["analysis_results"][file_name][
                    "csv_error"
                ] = str(csv_err)

        # Combined PDF Generation
        if len(uploaded_files) > 1:  # Generate combined PDF only if multiple files are uploaded
            try:
                combined_pdf_report_bytes = generate_combined_pdf_report(analysis_results_list)
                st.session_state["combined_pdf_report"] = combined_pdf_report_bytes  # Store combined pdf report in session state
            except Exception as combined_pdf_err:
                st.error(f"Error generating combined PDF report: {combined_pdf_err}")
                st.session_state["combined_pdf_error"] = str(combined_pdf_err)

            try:
                combined_csv_report_string = generate_combined_csv_report(analysis_results_list)
                st.session_state["combined_csv_report"] = combined_csv_report_string  # Store combined csv report in session state
            except Exception as combined_csv_err:
                st.error(f"Error generating combined CSV report: {combined_csv_err}")
                st.session_state["combined_csv_error"] = str(combined_csv_err)
    else:
        analysis_results_list = [] # Ensure this is empty if the button wasn't pressed

# Show Download Options Conditionally
if st.session_state["analysis_completed"] and uploaded_files:
    if "analysis_results" in st.session_state and st.session_state["analysis_results"]:
        st.subheader("Download Options")

        # Download buttons
        if len(uploaded_files) == 1:
            file_name = uploaded_files[0].name  # Get the single filename
            if "pdf_report" in st.session_state["analysis_results"][file_name]:
                pdf_report_bytes = st.session_state["analysis_results"][file_name]["pdf_report"]
                st.download_button(
                    label=f"Download PDF Report: {file_name}",
                    data=pdf_report_bytes,
                    file_name=f"analysis_report_{file_name}.pdf",
                    mime="application/pdf",
                    key=f"pdf_download_{file_name}"
                )
            if "csv_report" in st.session_state["analysis_results"][file_name]:
                csv_report_string = st.session_state["analysis_results"][file_name]["csv_report"]
                st.download_button(
                    label=f"Download CSV Report: {file_name}",
                    data=csv_report_string,
                    file_name=f"analysis_report_{file_name}.csv",
                    mime="text/csv",
                    key=f"csv_download_{file_name}"
                )
        elif len(uploaded_files) > 1: # Modified to elif for clarity
            if "combined_pdf_report" in st.session_state:
                combined_pdf_report_bytes = st.session_state["combined_pdf_report"]
                st.download_button(
                    label="Download Combined PDF Report (All Files)",
                    data=combined_pdf_report_bytes,
                    file_name="combined_analysis_report.pdf",
                    mime="application/pdf",
                    key="combined_pdf_download",
                )
            if "combined_csv_report" in st.session_state:
                combined_csv_report_string = st.session_state["combined_csv_report"]
                st.download_button(
                    label="Download Combined CSV Report (All Files)",
                    data=combined_csv_report_string,
                    file_name="combined_analysis_report.csv",
                    mime="text/csv",
                    key="combined_csv_download",
                )

        # ZIP creation happens *outside* the single/multi file conditional
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
            for file_name in st.session_state["analysis_results"]:
                if "error" in st.session_state["analysis_results"][file_name]:
                    continue

                if "pdf_report" in st.session_state["analysis_results"][file_name]:
                    pdf_report_bytes = st.session_state["analysis_results"][file_name]["pdf_report"]
                    zipf.writestr(f"analysis_report_{file_name}.pdf", pdf_report_bytes.getvalue())

                if "csv_report" in st.session_state["analysis_results"][file_name]:
                    csv_report_string = st.session_state["analysis_results"][file_name]["csv_report"]
                    zipf.writestr(f"analysis_report_{file_name}.csv", csv_report_string)

                if file_name in st.session_state["sample_rates"]:
                    audio_data = st.session_state["audio_data"][file_name]
                    sample_rate = st.session_state["sample_rates"][file_name]
                    audio_bytes_zip = io.BytesIO()
                    sf.write(audio_bytes_zip, audio_data, sample_rate, format="WAV")
                    zipf.writestr(file_name, audio_bytes_zip.getvalue())

        zip_buffer.seek(0)
        st.download_button(
            label="Download All as ZIP",
            data=zip_buffer,
            file_name="audio_analysis_reports.zip",
            mime="application/zip",
            key="zip_download_all",  # Unique key for the zip download button
        )