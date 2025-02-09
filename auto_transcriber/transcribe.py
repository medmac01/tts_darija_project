import pandas as pd
import os
from pydub import AudioSegment
from pydub.silence import split_on_silence
import google.generativeai as genai
import time
from tqdm import tqdm
from datasets import Dataset, Audio, Features, Value
from datasets import DatasetDict

# import gspread
# from oauth2client.service_account import ServiceAccountCredentials

# def google_sheet_to_csv(sheet_url, csv_path):
#     scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
#     creds = ServiceAccountCredentials.from_json_keyfile_name('path/to/credentials.json', scope)
#     client = gspread.authorize(creds)
    
#     sheet = client.open_by_url(sheet_url).sheet1
#     data = sheet.get_all_records()
    
#     df = pd.DataFrame(data)
#     df.to_csv(csv_path, index=False)

def get_pending_videos(csv_path):
    df = pd.read_csv(csv_path)
    pending_videos = df[df['isdone'] == False]['video'].tolist()
    return pending_videos

def mark_as_done(csv_path, video_id):
    df = pd.read_csv(csv_path)
    df.loc[df['video'] == video_id, 'isdone'] = True
    
    # Get the foldername from the title of the video
    foldername = df.loc[df['video'] == video_id, 'title'].values[0]
    
    # Count the number of files in the subfolder
    subfolder_path = os.path.join(foldername, video_id)
    n_chunks = len([f for f in os.listdir(subfolder_path) if os.path.isfile(os.path.join(subfolder_path, f))])
    
    df.loc[df['video'] == video_id, 'n_chunks'] = n_chunks
    df.to_csv(csv_path, index=False)

def split_wav_on_silence(file_path, silence_thresh=-40, min_silence_len=500, keep_silence=500):
    """
    Splits a WAV file into chunks based on silence.

    :param file_path: Path to the WAV file
    :param silence_thresh: Silence threshold in dB
    :param min_silence_len: Minimum length of silence to be used for splitting (in ms)
    :param keep_silence: Amount of silence to leave at the beginning and end of each chunk (in ms)
    :return: List of AudioSegment chunks
    """
    audio = AudioSegment.from_wav(file_path)
    chunks = split_on_silence(audio, 
                              min_silence_len=min_silence_len, 
                              silence_thresh=silence_thresh, 
                              keep_silence=keep_silence)
    return chunks

def remove_short_chunks(chunks, min_duration=1700, max_duration=31000):
    """
    Removes chunks with duration less than or equal to the specified minimum duration.

    :param chunks: List of AudioSegment chunks
    :param min_duration: Minimum duration in milliseconds
    :return: List of filtered AudioSegment chunks
    """
    return [chunk for chunk in chunks if len(chunk) > min_duration and len(chunk) < max_duration]


def upload_to_gemini(path, mime_type=None):
    """Uploads the given file to Gemini.

    See https://ai.google.dev/gemini-api/docs/prompting_with_media
    """
    file = genai.upload_file(path, mime_type=mime_type)
    # print(f"Uploaded file '{file.display_name}' as: {file.uri}")
    return file

class APIKeyManager:
    def __init__(self, api_keys):
        self.api_keys = api_keys
        self.index = 0

    def get_next_key(self):
        key = self.api_keys[self.index]
        self.index = (self.index + 1) % len(self.api_keys)
        return key

def configure_gemini(api_key):
    """Configures the API key for Gemini."""
    genai.configure(api_key=api_key) # Put your API key here, don't use mine :)


    # Create the model
    generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
    }

    model = genai.GenerativeModel(
    model_name="gemini-2.0-flash-exp",
    generation_config=generation_config,
    )

    return model

def transcribe_chunks(chunks_path, model):
    """ 
    Transcribes the audio chunks using the specified model.

    :param chunks_path: Path to the folder containing the audio chunks
    :param model: GenerativeModel instance
    :return: DataFrame containing the transcriptions with filenames
    """

    df = pd.DataFrame(columns=["filename", "transcription"])
    chunk_files = [f for f in os.listdir(chunks_path) if f.endswith('.wav')]
    for i in tqdm(range(len(chunk_files))):
        if i > 0 and i % 9 == 0:
            time.sleep(60)  # Sleep for 1 minute every 9 iterations
            # print("Sleeping for 1 minute...")

        file = upload_to_gemini(os.path.join(chunks_path, f"chunk{i}.wav"), mime_type="audio/mpeg")

        chat_session = model.start_chat(
            history=[
                {
                "role": "user",
                "parts": [
                    file,
                    "transcribe the Moroccan Darija audio i sent return the transcription",
                ],
                }
            ]
        )

        try:
            response = chat_session.send_message("transcribe the Moroccan Darija audio i sent return the transcription")
        except Exception as e:
            print(f"Error transcribing chunk{i}.wav: {e}")
            time.sleep(60)  # Sleep for 1 minute before retrying
            response = chat_session.send_message("transcribe the Moroccan Darija audio i sent return the transcription")

        df.loc[i] = [f"chunk{i}.wav", response.text]

    return df

def update_transcriptions(df, chunks_path, main_csv_path):
    """Stores the transcriptions in a CSV file."""
    df.to_csv(os.path.join(chunks_path, "transcriptions.csv"), index=False)

    # Update the global csv file with the transcriptions
    df_main = pd.read_csv(main_csv_path)
    for i, row in df.iterrows():
        video_id = os.path.basename(chunks_path)
        chunk_filename = row['filename']
        transcription = row['transcription']
        new_row = {'video': video_id, 'filename': chunk_filename, 'transcription': transcription}
        df_main.loc[len(df_main)] = new_row

    df_main.to_csv(main_csv_path, index=False)


def create_and_push_hf_dataset(main_df, dataset_name):
    """
    Creates a Hugging Face dataset with audio, filename, and transcription columns, and pushes it to the Hugging Face Hub.

    :param chunks_dir: Directory containing the .wav chunk files
    :param df: DataFrame containing filenames and transcriptions
    :param dataset_name: Name of the dataset to be created on Hugging Face Hub
    """
    # Define the features of the dataset
    features = Features({
        'audio': Audio(sampling_rate=22050),
        'filename': Value('string'),
        'transcription': Value('string')
    })

    # Create a list of dictionaries for the dataset
    data = []
    for _, row in main_df.iterrows():
        file_path = os.path.join("wavs", "chunks", row["video"], row['filename'])
        data.append({
            'audio': file_path,
            'filename': row['filename'],
            'transcription': row['transcription']
        })

    # Create the dataset
    dataset = Dataset.from_list(data, features=features)

    # Create a DatasetDict
    dataset_dict = DatasetDict({'train': dataset})

    # Push the dataset to the Hugging Face Hub
    dataset_dict.push_to_hub(dataset_name)


def init_csv():
    # Create an empty videos.csv with the necessary columns
    if not os.path.exists("videos.csv"):
        df = pd.DataFrame(columns=["video", "title", "isdone", "n_chunks", "transcription"])
        df.to_csv("videos.csv", index=False)


def main():
    # pending = get_pending_videos("videos.csv")
    api_keys_list = [
                    "AIzaSyDK0RmDHf6OojBPV58ViZbGxQq5hmqY2Co",
                    "AIzaSyDaDjD8mCZDyA9LQuhbfhONv8mkwJG35L0", # nouamane :)
                    "AIzaSyCvg6mmeWFZi6Dd84-skVje2QmJC8kzs5o",
                    "AIzaSyDjDQ0toJFzxC3qKBzr4gRcKx_RCnVG6mw",
                    ]
    api_key_manager = APIKeyManager(api_keys_list)
    wav_dir = "wav/raw_4"
    output_dir = "wav/chunks_4"

    init_csv()

    # Loop over all .wav files in the directory
    for wav_file in os.listdir(wav_dir):
        # Check if the folder with the title in pending does not exist
        if wav_file.endswith(".wav"):
            file_path = os.path.join(wav_dir, wav_file)
            chunks = split_wav_on_silence(file_path, silence_thresh=-40, min_silence_len=150, keep_silence=500)
            print(f"Number of chunks before filtering for {wav_file}: {len(chunks)}")
            filtered_chunks = remove_short_chunks(chunks)
            print(f"Number of chunks after filtering for {wav_file}: {len(filtered_chunks)}")

            # Create a subfolder for each wav file
            subfolder = os.path.join(output_dir, os.path.splitext(wav_file)[0])
            os.makedirs(subfolder, exist_ok=True)

            # Export chunks to the subfolder
            for i, chunk in enumerate(filtered_chunks):
                chunk.export(os.path.join(subfolder, f"chunk{i}.wav"), format="wav")

    chunked_dir = "wav/chunks_4"
    for folder in os.listdir(chunked_dir):
        folder_path = os.path.join(chunked_dir, folder)
        if os.path.isdir(folder_path):
            print(f"Processing: {folder}")
            model = configure_gemini(api_key_manager.get_next_key())
            df_transcriptions = transcribe_chunks(folder_path, model)
            update_transcriptions(df_transcriptions, folder_path, "videos.csv")
                # mark_as_done("videos.csv", folder)

        
        # print(f"Processing video: {video_id}")
        # # mark_as_done("videos.csv", video_id)

        # # Split the audio file into chunks
        # audio_path = os.path.join(video_id, f"wav{video_id}.wav")
        # chunks = split_wav_on_silence(audio_path)
        # chunks = remove_short_chunks(chunks)

        # # Save the chunks to a subfolder
        # subfolder_path = os.path.join("wavs", "chunks", video_id)
        # os.makedirs(subfolder_path, exist_ok=True)
        # for i, chunk in enumerate(chunks):
        #     chunk.export(os.path.join(subfolder_path, f"chunk{i}.wav"), format="wav")

        # # Transcribe the chunks
        # model = configure_gemini(api_key_manager.get_next_key())

if __name__ == "__main__":
    main()