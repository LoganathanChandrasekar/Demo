import streamlit as st
import boto3
import os
import pandas as pd
import io
import tempfile
import uuid
import time
import requests
import string  # Import the string module for punctuation removal
from botocore.exceptions import ClientError, NoCredentialsError, CredentialRetrievalError

class AudioTranscriptionTool:
    def __init__(self):
        """Initializes the AudioTranscriptionTool, setting up AWS clients and session state."""
        try:
            # --- AWS Credentials Handling ---
            aws_access_key_id = "access key ID"  # Replace with your actual access key ID
            aws_secret_access_key = "secret access key" # Replace with your actual secret access key
            aws_region = "us-east-1"  # Or your desired AWS region
            s3_bucket_name = "audio-transcripe-stt" # Replace with your actual S3 bucket name - 

            if not s3_bucket_name:
                st.error("S3_BUCKET_NAME is missing from Streamlit secrets. Please configure it.")
                raise ValueError("S3_BUCKET_NAME not configured")

            # Create Transcribe and S3 clients with explicit credentials if available
            if aws_access_key_id and aws_secret_access_key:
                self.transcribe_client = boto3.client(
                    'transcribe',
                    aws_access_key_id=aws_access_key_id,
                    aws_secret_access_key=aws_secret_access_key,
                    region_name=aws_region
                )
                self.s3_client = boto3.client(
                    's3',
                    aws_access_key_id=aws_access_key_id,
                    aws_secret_access_key=aws_secret_access_key,
                    region_name=aws_region
                )
                st.success("AWS credentials loaded from Streamlit secrets.")
            else:
                st.warning("Using default AWS credentials provider chain. Ensure AWS CLI is configured or running in an environment with IAM role.")
                self.transcribe_client = boto3.client('transcribe', region_name=aws_region)
                self.s3_client = boto3.client('s3', region_name=aws_region)

            self.s3_bucket_name = s3_bucket_name # Store bucket name
            self.aws_region = aws_region # Store region

        except (NoCredentialsError, CredentialRetrievalError) as e:
            st.error(f"AWS Credentials Error: {e}")
            st.error("Please configure AWS credentials:")
            st.error("1. Set AWS_ACCESS_KEY_ID in Streamlit secrets")
            st.error("2. Set AWS_SECRET_ACCESS_KEY in Streamlit secrets")
            st.error("3. Set AWS_REGION in Streamlit secrets (e.g., us-east-1)")
            st.error("4. Set S3_BUCKET_NAME in Streamlit secrets (your S3 bucket name)") # Important: S3_BUCKET_NAME
            st.error("5. Ensure AWS CLI is configured if not using secrets.")
            st.stop()

        except Exception as e:
            st.error(f"Initialization Error: {e}")
            st.error(f"Detailed error: {e}")
            st.stop()

        # --- Session State Initialization ---
        if 'uploaded_files' not in st.session_state:
            st.session_state.uploaded_files = []
        if 'transcription_results' not in st.session_state:
            st.session_state.transcription_results = []

        # --- Word Mapping ---
        self.word_mapping = {
            "S0001":"Hey Anka",
            "S0002":"Hey Anka",
            "S0003":"Hey Anka",
            "S0004":"Hey Anka",
            "S0005":"Hey Anka",
"S0006":"Hey Anka",
"S0007":"Hi Anka",
"S0008":"Hi Anka",
"S0009":"Hi Anka",
"S0010":"Hi Anka",
"S0011":"Hi Anka",
"S0012":"Hi Anka",
"S0013":"Previous Track",
"S0014":"Previous Track",
"S0015":"Previous Track",
"S0016":"Previous Track",
"S0017":"Previous Track",
"S0018":"Previous Track",
"S0019":"Next Track",
"S0020":"Next Track",
"S0021":"Next Track",
"S0022":"Next Track",
"S0023":"Next Track",
"S0024":"Next Track",
"S0025":"Increase Volume",
"S0026":"Increase Volume",
"S0027":"Increase Volume",
"S0028":"Increase Volume",
"S0029":"Increase Volume",
"S0030":"Increase Volume",
"S0031":"Decrease Volume",
"S0032":"Decrease Volume",
"S0033":"Decrease Volume",
"S0034":"Decrease Volume",
"S0035":"Decrease Volume",
"S0036":"Decrease Volume",
"S0037":"Start Playing",
"S0038":"Start Playing",
"S0039":"Start Playing",
"S0040":"Start Playing",
"S0041":"Start Playing",
"S0042":"Start Playing",
"S0043":"Resume Playing",
"S0044":"Resume Playing",
"S0045":"Resume Playing",
"S0046":"Resume Playing",
"S0047":"Resume Playing",
"S0048":"Resume Playing",
"S0049":"Pause Audio",
"S0050":"Pause Audio",
"S0051":"Pause Audio",
"S0052":"Pause Audio",
"S0053":"Pause Audio",
"S0054":"Pause Audio",
"S0055":"ANC Mode",
"S0056":"ANC Mode",
"S0057":"ANC Mode",
"S0058":"ANC Mode",
"S0059":"ANC Mode",
"S0060":"ANC Mode",
"S0061":"Transparent Mode",
"S0062":"Transparent Mode",
"S0063":"Transparent Mode",
"S0064":"Transparent Mode",
"S0065":"Transparent Mode",
"S0066":"Transparent Mode",
"S0067":"Adaptive Mode",
"S0068":"Adaptive Mode",
"S0069":"Adaptive Mode",
"S0070":"Adaptive Mode",
"S0071":"Adaptive Mode",
"S0072":"Adaptive Mode",
"S0073":"Answer Call",
"S0074":"Answer Call",
"S0075":"Answer Call",
"S0076":"Answer Call",
"S0077":"Answer Call",
"S0078":"Answer Call",
"S0079":"Reject Call",
"S0080":"Reject Call",
"S0081":"Reject Call",
"S0082":"Reject Call",
"S0083":"Reject Call",
"S0084":"Reject Call",
"S0085":"Start Translation",
"S0086":"Start Translation",
"S0087":"Start Translation",
"S0088":"Start Translation",
"S0089":"Start Translation",
"S0090":"Start Translation",
"S0091":"Hey Siri",
"S0092":"Hey Siri",
"S0093":"Hey Siri",
"S0094":"Hey Siri",
"S0095":"Hey Siri",
"S0096":"Hey Siri",
"S0097":"Hey Anka",
"S0098":"Hey Anka",
"S0099":"Hey Anka",
"S0100":"Hey Anka",
"S0101":"Hey Anka",
"S0102":"Hey Anka",
"S0103":"Hi Anka",
"S0104":"Hi Anka",
"S0105":"Hi Anka",
"S0106":"Hi Anka",
"S0107":"Hi Anka",
"S0108":"Hi Anka",
"S0109":"Previous Track",
"S0110":"Previous Track",
"S0111":"Previous Track",
"S0112":"Previous Track",
"S0113":"Previous Track",
"S0114":"Previous Track",
"S0115":"Next Track",
"S0116":"Next Track",
"S0117":"Next Track",
"S0118":"Next Track",
"S0119":"Next Track",
"S0120":"Next Track",
"S0121":"Increase Volume",
"S0122":"Increase Volume",
"S0123":"Increase Volume",
"S0124":"Increase Volume",
"S0125":"Increase Volume",
"S0126":"Increase Volume",
"S0127":"Decrease Volume",
"S0128":"Decrease Volume",
"S0129":"Decrease Volume",
"S0130":"Decrease Volume",
"S0131":"Decrease Volume",
"S0132":"Decrease Volume",
"S0133":"Start Playing",
"S0134":"Start Playing",
"S0135":"Start Playing",
"S0136":"Start Playing",
"S0137":"Start Playing",
"S0138":"Start Playing",
"S0139":"Resume Playing",
"S0140":"Resume Playing",
"S0141":"Resume Playing",
"S0142":"Resume Playing",
"S0143":"Resume Playing",
"S0144":"Resume Playing",
"S0145":"Pause Audio",
"S0146":"Pause Audio",
"S0147":"Pause Audio",
"S0148":"Pause Audio",
"S0149":"Pause Audio",
"S0150":"Pause Audio",
"S0151":"ANC Mode",
"S0152":"ANC Mode",
"S0153":"ANC Mode",
"S0154":"ANC Mode",
"S0155":"ANC Mode",
"S0156":"ANC Mode",
"S0157":"Transparent Mode",
"S0158":"Transparent Mode",
"S0159":"Transparent Mode",
"S0160":"Transparent Mode",
"S0161":"Transparent Mode",
"S0162":"Transparent Mode",
"S0163":"Adaptive Mode",
"S0164":"Adaptive Mode",
"S0165":"Adaptive Mode",
"S0166":"Adaptive Mode",
"S0167":"Adaptive Mode",
"S0168":"Adaptive Mode",
"S0169":"Answer Call",
"S0170":"Answer Call",
"S0171":"Answer Call",
"S0172":"Answer Call",
"S0173":"Answer Call",
"S0174":"Answer Call",
"S0175":"Reject Call",
"S0176":"Reject Call",
"S0177":"Reject Call",
"S0178":"Reject Call",
"S0179":"Reject Call",
"S0180":"Reject Call",
"S0181":"Start Translation",
"S0182":"Start Translation",
"S0183":"Start Translation",
"S0184":"Start Translation",
"S0185":"Start Translation",
"S0186":"Start Translation",
"S0187":"Hey Siri",
"S0188":"Hey Siri",
"S0189":"Hey Siri",
"S0190":"Hey Siri",
"S0191":"Hey Siri",
"S0192":"Hey Siri"
            # Add more mappings here
        }

    def extract_code_from_filename(self, filename: str) -> str:
        """Extracts S-code from filename based on underscore delimiter."""
        parts = filename.split('_')
        for part in parts:
            if part.startswith('S') and part[1:].isdigit():
                return part
        return ""

    def transcribe_file(self, file_path: str, filename: str) -> str:
        """Transcribes audio file using AWS Transcribe, uploading to S3 temporarily."""
        job_name = f'transcription-{uuid.uuid4()}'
        s3_key = f'audio_uploads/{filename}' # Key for S3 object

        try:
            # --- Upload to S3 ---
            try:
                self.s3_client.upload_file(file_path, self.s3_bucket_name, s3_key) # Upload temp file to S3
                st.info(f"File '{filename}' temporarily uploaded to S3 bucket '{self.s3_bucket_name}' in region '{self.aws_region}' for transcription.")
            except Exception as s3_upload_error:
                st.error(f"S3 Upload Error for '{filename}': {s3_upload_error}")
                st.error(f"Please check if: \n- S3_BUCKET_NAME is correct in secrets.\n- AWS credentials have 's3:PutObject' permission for the bucket.\n- The bucket '{self.s3_bucket_name}' exists in region '{self.aws_region}'.")
                return ""

            # --- Start Transcription Job (using S3 URI) ---
            try:
                self.transcribe_client.start_transcription_job(
                    TranscriptionJobName=job_name,
                    Media={'MediaFileUri': f's3://{self.s3_bucket_name}/{s3_key}'}, # Use S3 URI
                    MediaFormat='wav',
                    LanguageCode='en-US'
                )
                st.info(f"Transcription job started for '{filename}' with job name '{job_name}' using S3 URI.")
            except ClientError as job_error:
                st.error(f"Transcription Job Start Error for '{filename}': {job_error}")
                st.error(f"Detailed Job Error: {job_error}")
                st.error(f"Please check if: \n- AWS credentials have 'transcribe:StartTranscriptionJob' and **'s3:GetObject'** permissions.\n- The AWS region is correctly configured and supported by Transcribe.\n- The S3 URI 's3://{self.s3_bucket_name}/{s3_key}' is valid and accessible by Transcribe.") # **Added s3:GetObject to error message**
                return ""

            # --- Polling for Job Completion ---
            max_tries = 60  # 30 minutes total (30 seconds * 60)
            for _ in range(max_tries):
                try:
                    result = self.transcribe_client.get_transcription_job(TranscriptionJobName=job_name)
                    job_status = result['TranscriptionJob']['TranscriptionJobStatus']

                    if job_status == 'COMPLETED':
                        st.info(f"Transcription job for '{filename}' COMPLETED successfully.")
                        # --- Retrieve Transcription ---
                        try:
                            transcript_uri = result['TranscriptionJob']['Transcript']['TranscriptFileUri']
                            transcript_response = requests.get(transcript_uri)
                            transcript_data = transcript_response.json()
                            transcript_text = transcript_data['results']['transcripts'][0]['transcript'].lower().strip()
                            return transcript_text
                        except Exception as transcript_error:
                            st.error(f"Transcript Retrieval Error for '{filename}': {transcript_error}")
                            st.error(f"Could not retrieve transcript from: {transcript_uri}")
                            return ""

                    elif job_status == 'FAILED':
                        failure_reason = result['TranscriptionJob'].get('FailureReason', 'No reason provided')
                        st.error(f"Transcription job failed for '{filename}'.")
                        st.error(f"Failure Reason: {failure_reason}")
                        return ""

                    time.sleep(30)

                except Exception as poll_error:
                    st.error(f"Error during transcription job polling for '{filename}': {poll_error}")
                    return ""

            st.error(f"Transcription job timed out for '{filename}' after 30 minutes.")
            return ""

        except Exception as e:
            st.error(f"General transcription error for '{filename}': {e}")
            st.error(f"Full Exception Details: {e}")
            return ""

    def compare_transcript(self, original_word: str, transcript_word: str) -> tuple[float, str]: # Return score and string "True"/"False"
        """Compares original and transcribed words, ignoring punctuation, and returns match score and "True"/"False" string."""
        original_word = original_word.lower().strip()
        transcript_word = transcript_word.lower().strip()

        # Remove punctuation from both strings before comparison
        original_word_no_punct = original_word.translate(str.maketrans('', '', string.punctuation))
        transcript_word_no_punct = transcript_word.translate(str.maketrans('', '', string.punctuation))

        if original_word_no_punct == transcript_word_no_punct:
            return 100.0, "True"  # 100% match and "True" string
        else:
            return 0.0, "False"   # 0% match and "False" string


    def process_batch_transcription(self, uploaded_files):
        """Processes batch transcription for all uploaded audio files."""
        results = []

        for uploaded_file in uploaded_files:
            # --- Save uploaded file temporarily ---
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
                temp_file.write(uploaded_file.getvalue())
                temp_file_path = temp_file.name

            try:
                filename = uploaded_file.name
                s_code = self.extract_code_from_filename(filename)
                original_word = self.word_mapping.get(s_code, "")

                # --- Transcribe file ---
                transcript = self.transcribe_file(temp_file_path, filename)

                # --- Compare transcript and calculate score and boolean match ---
                match_score, match_text = self.compare_transcript(original_word, transcript) # Get score and "True"/"False" string

                results.append({
                    'Filename': filename,
                    'S-Code': s_code,
                    'Original Word': original_word,
                    'Transcribed Text': transcript,
                    'Match Score (%)': match_score,
                    'Match': match_text  # Use the "True"/"False" string for 'Match' column
                })

            except Exception as e:
                st.error(f"Error processing file '{uploaded_file.name}': {e}")
                results.append({
                    'Filename': uploaded_file.name,
                    'S-Code': '',
                    'Original Word': '',
                    'Transcribed Text': 'Error',
                    'Match Score (%)': 0.0,
                    'Match': "False" # Default to "False" string in case of error
                })

            finally:
                # --- Clean up temporary file ---
                try:
                    os.unlink(temp_file_path)
                except Exception as cleanup_error:
                    st.warning(f"Warning: Could not delete temporary file '{temp_file_path}'. {cleanup_error}")
                # --- Optionally, you might want to delete the file from S3 after transcription
                # --- However, for simplicity and to avoid potential issues, I'm skipping S3 deletion in this code.
                # --- If you want to implement S3 deletion, add code here using self.s3_client.delete_object(...)

        return results


def main():
    """Main function to run the Streamlit app."""
    st.title("Audio Transcription & Comparison Tool (S3 Temporary Upload)") # Updated title
    st.write("Upload WAV audio files to transcribe and compare against expected words (Uses S3 temporarily for transcription).") # Updated description

    try:
        # --- Initialize the tool ---
        tool = AudioTranscriptionTool()

    except Exception as init_error:
        st.error(f"App Initialization Failed: {init_error}")
        st.error("Please check the error messages above and your AWS configuration.")
        return

    # --- File uploader ---
    uploaded_files = st.file_uploader(
        "Upload Audio Files (.wav)",
        type=['wav'],
        accept_multiple_files=True
    )

    if uploaded_files:
        st.session_state.uploaded_files = uploaded_files
        st.success(f"{len(uploaded_files)} file(s) uploaded and ready for transcription.")

    # --- Transcribe Button ---
    if st.button("Transcribe Audio Files"):
        if st.session_state.uploaded_files:
            if not st.secrets.get('S3_BUCKET_NAME'): # Check if S3 bucket is configured
                st.error("S3_BUCKET_NAME is not configured in Streamlit secrets. Transcription cannot proceed.")
                st.stop()

            with st.spinner('Processing transcriptions... This may take a few minutes.'):
                try:
                    results = tool.process_batch_transcription(st.session_state.uploaded_files)
                    st.session_state.transcription_results = results

                    # --- Display Results in DataFrame ---
                    if results:
                        df = pd.DataFrame(results)
                        st.dataframe(df)

                        # --- Dynamic CSV Filename ---
                        # Get the first uploaded filename to derive CSV name
                        first_filename = st.session_state.uploaded_files[0].name
                        base_filename = first_filename.split('_')[0]  # Get part before first underscore
                        csv_filename = f"{base_filename}.csv"

                        # --- Download Results Button ---
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="Download Transcription Results (CSV)",
                            data=csv,
                            file_name=csv_filename,  # Use the dynamic filename here
                            mime="text/csv"
                        )
                    else:
                        st.warning("No transcription results to display.")

                except Exception as process_error:
                    st.error(f"Transcription Process Error: {process_error}")
        else:
            st.warning("Please upload audio files first!")

if __name__ == "__main__":
    main()
