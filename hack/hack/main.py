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
from utils.clips import get_embedd, get_clips, time_to_seconds
from utils.text_tone import get_inappropriate
from utils.morph import get_morph
from utils.photo_search import Embedding
from utils.lemm import get_scene_info


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
    os.makedirs(
        save_directory, exist_ok=True
    )  # Создайте директорию, если она не существует

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

    if "whisper_data" not in st.session_state:
        st.session_state.whisper_data = None
        st.session_state.frames = None
        st.session_state.vifeos_array = []

    st.session_state.whisper_data = get_text_stats(file_path)
    st.session_state.whisper_data["emotionals"] = ""
    st.session_state.whisper_data["objects"] = [[] for _ in range(len(st.session_state.whisper_data))]
    st.session_state.whisper_data["anomaly"] = ""
    st.session_state.whisper_data["objects_boxes"] = ""
    st.session_state.whisper_data["anomaly_boxes"] = ""

    st.session_state.frames = get_frames(
        file_path,
        step_seconds=1,
    )

    emb = Embedding(name_db="db")

    # db_path = f"dbs/{file_path}"

    step_seconds = 1
    emb.proccessing(file_path, step_seconds, "1")

    for i in range(len(st.session_state.whisper_data)):
        if st.session_state.whisper_data["start"][i] == st.session_state.whisper_data["end"][i]:
            continue
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav_file:
            output_file = temp_wav_file.name
            extract_wav_chunk(
                file_path[:-3] + "wav",
                start_sec=st.session_state.whisper_data["start"][i],
                end_sec=st.session_state.whisper_data["end"][i],
                output_file=output_file,
            )

            emotionals = get_emotional_pipeline(output_file)

        st.session_state.whisper_data.loc[i, "emotionals"] = emotionals

        start_frame = int(st.session_state.whisper_data["start"][i])
        end_frame = int(st.session_state.whisper_data["end"][i])
        object_arr, boxplot_info, anomaly_object_arr, anomaly_boxplot_info = (
            get_object_from_video(st.session_state.frames[start_frame:end_frame])
        )

        st.session_state.whisper_data["objects"][i] = object_arr
        st.session_state.whisper_data["anomaly"][i] = anomaly_object_arr

        st.session_state.whisper_data["objects_boxes"][i] = boxplot_info
        st.session_state.whisper_data["anomaly_boxes"][i] = anomaly_boxplot_info

    st.session_state.whisper_data["text_tone"] = get_inappropriate(st.session_state.whisper_data["text"].to_list())
    st.session_state.whisper_data["text_morph"] = get_morph(list(st.session_state.whisper_data["text"]))

    st.session_state.whisper_data.to_pickle("whisper.pkl")

    embeds = get_embedd(st.session_state.frames)

    clips = get_clips(embeds, st.session_state.frames)

    st.session_state.whisper_data["anomaly"] = st.session_state.whisper_data["anomaly"].astype(str)
    st.session_state.whisper_data["text_tone"] = st.session_state.whisper_data["text_tone"].astype(str)
    st.session_state.whisper_data["text_morph"] = st.session_state.whisper_data["text_morph"].astype(str)
    # st.write(
    #     st.session_state.whisper_data[
    #         ["text", "emotionals", "objects", "anomaly", "text_tone", "text_morph"]
    #     ]
    # )  # "text", "emotionals", "objects", "anomaly", "text_tone", "text_morph"

    st.video(
        data=file_path, start_time=st.session_state.start_time,
    )

    new_df = get_scene_info(st.session_state.whisper_data, clips)

    st.header("Сцены видео:")
    # Создаем две колонки: одна для изображений, другая для кнопок

    for i, clip in enumerate(clips):
        st.header(f"Сцена {i + 1}")
        col1, col2, col3, col4 = st.columns([2, 1, 1, 1]) 
        # Преобразуем таймкод в формат "минуты:секунды"
        with col1:
            st.image(st.session_state.frames[time_to_seconds(clip)][1], width=200)

        with col2:
            st.markdown(f'Эмоциональный окрас {new_df["emotion"][i]}')

        with col3:
             st.markdown(f'Ключевые объекты: {new_df["important_objects"][i]}')

        with col4:
            # Создаем кнопку для перехода на этот момент
             st.markdown(f"Сцена  {i + 1} ({clip})")

    # Фильтрация строк по выбранной категории
    st.header("Музыка.")
    music_lines = [line for line in st.session_state.whisper_data["text"] if "МУЗЫКА" in line]
    if music_lines:
        st.write("Строки, содержащие музыку:")
        for line in music_lines:
            st.write(line)
    else:
        st.write("Нет строк, содержащих музыку.")

    # st.session_state.whisper_data["anomaly"] = st.session_state.whisper_data["anomaly"].astype(str)
    # st.session_state.whisper_data["text_tone"] = st.session_state.whisper_data["text_tone"].astype(str)
    # st.session_state.whisper_data["text_morph"] = st.session_state.whisper_data["text_morph"].astype(str)
    st.write(
        st.session_state.whisper_data[
            ["start", "text", "emotionals", "objects", "anomaly", "text_tone", "text_morph"]
        ]
    )


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
    st.set_page_config(
        layout='wide'
    )
    main()
