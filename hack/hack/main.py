from pathlib import Path
from utils.whisper_service import get_text_stats
from utils.wav_chunks import extract_wav_chunk
from utils.emotional_extractor import get_emotional_pipeline
from utils.yolo_service import filtered_object

# whisper_data = get_text_stats(str(Path.cwd() / "data" / "external" / "input_car.mp4"))


# for i in range(len(whisper_data)):
#     print(whisper_data["start"][i])
#     print(whisper_data["end"][i])
#     print(whisper_data["text"][i])
#     if whisper_data["start"][i] == whisper_data["end"][i]:
#         continue
#     chunk_bytes = extract_wav_chunk(
#         str(Path.cwd() / "data" / "external" / "input_car.wav"), 
#         start_sec=whisper_data["start"][i],
#         end_sec=whisper_data["end"][i],
#         output_file=str(Path.cwd() / "data" / "external" / f"input_car{i}.wav"),
#         )
    


#     emotionals = get_emotional_pipeline(str(Path.cwd() / "data" / "external" / f"input_car{i}.wav"))

#     filtered_frames= filtered_object(
#         str(Path.cwd() / "data" / "external" / "input_car.mp4"),
#         frame_rate=1,
#         threshold=0.8,
#         start_time=whisper_data["start"][i],
#         end_time=whisper_data["end"][i],
#         )
    
#     print(filtered_frames)
#     print(emotionals)
import streamlit as st
from zipfile import ZipFile
import os
import cv2
import tempfile

def main():
    st.title("Video Processing App")

    # File upload section
    st.header("Upload Video or Zip Archive")
    uploaded_file = st.file_uploader("Choose a file", type=["mp4", "zip"])

    if uploaded_file is not None:
        # Check if the uploaded file is a video or a zip archive
        if uploaded_file.type == "video/mp4":
            process_video(uploaded_file)
        elif uploaded_file.type == "application/zip":
            process_zip_archive(uploaded_file)

def process_video(uploaded_file):
    st.header("Processing Video")

    # Save the uploaded video to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        temp_file_path = temp_file.name

    whisper_data = get_text_stats(temp_file_path)
    whisper_data["emotionals"] = ""
    whisper_data["objects"] = [[] for _ in range(len(whisper_data))]
    for i in range(len(whisper_data)):
        print(whisper_data["start"][i])
        print(whisper_data["end"][i])
        print(whisper_data["text"][i])
        if whisper_data["start"][i] == whisper_data["end"][i]:
            continue
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav_file:
            output_file = temp_wav_file.name
            extract_wav_chunk(
                temp_file_path[:-3] + "wav", 
                start_sec=whisper_data["start"][i],
                end_sec=whisper_data["end"][i],
                output_file=output_file,
                )
        
            emotionals = get_emotional_pipeline(output_file)
        
        whisper_data.loc[i, "emotionals"] = emotionals

        filtered_frames= filtered_object(
            temp_file_path,
            frame_rate=1,
            threshold=0.8,
            start_time=whisper_data["start"][i],
            end_time=whisper_data["end"][i],
            )
        
        whisper_data["objects"][i] = filtered_frames

    st.write(whisper_data[["text", "emotionals", "objects"]])

    os.remove(temp_wav_file)
    os.remove(temp_file_path)

def process_zip_archive(uploaded_file):
    st.header("Processing Zip Archive")

    # Save the uploaded zip file to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        temp_file_path = temp_file.name

    # Extract the zip file
    with ZipFile(temp_file_path, "r") as zip_ref:
        zip_ref.extractall(tempfile.gettempdir())

    # Perform video processing on each video in the archive
    for filename in os.listdir(tempfile.gettempdir()):
        if filename.endswith(".mp4"):
            st.subheader(f"Обработка {filename}")
            video_path = os.path.join(tempfile.gettempdir(), filename)
            
            whisper_data = get_text_stats(video_path)
            whisper_data["emotionals"] = ""
            whisper_data["objects"] = [[] for _ in range(len(whisper_data))]
            for i in range(len(whisper_data)):
                print(whisper_data["start"][i])
                print(whisper_data["end"][i])
                print(whisper_data["text"][i])
                if whisper_data["start"][i] == whisper_data["end"][i]:
                    continue
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav_file:
                    output_file = temp_wav_file.name
                    extract_wav_chunk(
                        temp_file_path[:-3] + "wav", 
                        start_sec=whisper_data["start"][i],
                        end_sec=whisper_data["end"][i],
                        output_file=output_file,
                        )
                
                    emotionals = get_emotional_pipeline(output_file)
                
                whisper_data.loc[i, "emotionals"] = emotionals

                filtered_frames= filtered_object(
                    video_path,
                    frame_rate=1,
                    threshold=0.8,
                    start_time=whisper_data["start"][i],
                    end_time=whisper_data["end"][i],
                    )
                
                whisper_data["objects"][i] = filtered_frames

            st.write(whisper_data[["text", "emotionals", "objects"]])
            os.remove(temp_wav_file)
            os.remove(temp_file_path)

            os.remove(video_path)

    # Clean up
    os.remove(temp_file_path)

if __name__ == "__main__":
    main()
