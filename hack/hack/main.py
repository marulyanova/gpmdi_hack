import os
import cv2
import tempfile
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

import pickle

import streamlit as st

from utils.whisper_service import get_text_stats
from utils.wav_chunks import extract_wav_chunk
from utils.emotional_extractor import get_emotional_pipeline
from utils.yolo_service import filtered_object, get_frames, get_object_from_video
from utils.clips import get_embedd, get_clips
from utils.text_tone import get_inappropriate
from utils.morph import get_morph
from utils.photo_search import Embedding, Similar, create_db






def main():
    st.title("Разметка видеоконтента")


    # File upload section
    st.header("Загрузите видео")
    uploaded_file = st.file_uploader("Выберите файл", type=["mp4", "zip", "mov", "avi"])

    if uploaded_file is not None:
        # Check if the uploaded file is a video or a zip archive
        if uploaded_file.type in ["video/mp4", "video/x-msvideo"]:
            process_video(uploaded_file)
        elif uploaded_file.type == "application/zip":
            process_zip_archive(uploaded_file)
        else:
            st.error(
                "Неподдерживаемый формат файла. Пожалуйста, загрузите видео в формате mp4, mov или avi."
            )


def process_video(uploaded_file):
    suffix = ".avi" if uploaded_file.type == "video/x-msvideo" else ".mp4"
    vifeo_format = (
        "video/x-msvideo" if uploaded_file.type == "video/x-msvideo" else "video/mp4"
    )

    save_directory = str(Path.cwd() / "data")
    os.makedirs(save_directory, exist_ok=True)  # Создайте директорию, если она не существует

    # Сохранение видеофайла
    if uploaded_file is not None:
        file_path = os.path.join(save_directory, "uploaded_video.mp4")
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

    # with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
    #     temp_file.write(uploaded_file.getbuffer())
    #     temp_file_path = temp_file.name

    if "start_time" not in st.session_state:
        st.session_state.start_time = 0

    whisper_data = get_text_stats(file_path)
    whisper_data["emotionals"] = ""
    whisper_data["objects"] = [[] for _ in range(len(whisper_data))]
    whisper_data["anomaly"] = ""
    whisper_data["objects_boxes"] = ""
    whisper_data["anomaly_boxes"] = ""

    frames = get_frames(
        file_path,
        step_seconds=1,
    )

    emb = Embedding(name_db="db")

    # db_path = f"dbs/{file_path}"

    step_seconds = 1
    emb.proccessing(file_path, step_seconds, "1")


    for i in range(len(whisper_data)):
        if whisper_data["start"][i] == whisper_data["end"][i]:
            continue
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav_file:
            output_file = temp_wav_file.name
            extract_wav_chunk(
                file_path[:-3] + "wav",
                start_sec=whisper_data["start"][i],
                end_sec=whisper_data["end"][i],
                output_file=output_file,
            )

            emotionals = get_emotional_pipeline(output_file)

        whisper_data.loc[i, "emotionals"] = emotionals

        start_frame = int(whisper_data["start"][i])
        end_frame = int(whisper_data["end"][i])
        object_arr, boxplot_info, anomaly_object_arr, anomaly_boxplot_info = (
            get_object_from_video(frames[start_frame:end_frame])
        )

        whisper_data["objects"][i] = object_arr
        whisper_data["anomaly"][i] = anomaly_object_arr

        whisper_data["objects_boxes"][i] = boxplot_info
        whisper_data["anomaly_boxes"][i] = anomaly_boxplot_info

    whisper_data["text_tone"] = get_inappropriate(whisper_data["text"].to_list())
    whisper_data["text_morph"] = get_morph(list(whisper_data["text"]))

    whisper_data.to_pickle("whisper.pkl")

    embeds = get_embedd(frames)

    clips = get_clips(embeds, frames)

    whisper_data["anomaly"] = whisper_data["anomaly"].astype(str)
    whisper_data["text_tone"] = whisper_data["text_tone"].astype(str)
    whisper_data["text_morph"] = whisper_data["text_morph"].astype(str)
    st.write(whisper_data[["text", "emotionals", "objects", "anomaly", "text_tone", "text_morph"]])# "text", "emotionals", "objects", "anomaly", "text_tone", "text_morph"

    st.video(
        data=file_path, start_time=st.session_state.start_time, format=vifeo_format
    )

    if st.button("Переместить начало видео на 10 секунд"):
        st.session_state.start_time += 10
        st.rerun()

    os.remove(output_file)


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
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".wav"
                ) as temp_wav_file:
                    output_file = temp_wav_file.name
                    extract_wav_chunk(
                        temp_file_path[:-3] + "wav",
                        start_sec=whisper_data["start"][i],
                        end_sec=whisper_data["end"][i],
                        output_file=output_file,
                    )

                    emotionals = get_emotional_pipeline(output_file)

                whisper_data.loc[i, "emotionals"] = emotionals

                filtered_frames = filtered_object(
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
