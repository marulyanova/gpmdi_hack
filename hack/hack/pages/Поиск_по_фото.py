import streamlit as st
import tempfile
import os
import cv2

import time
import lib
from PIL import Image
import numpy as np


from pathlib import Path

from utils.photo_search import Similar
from utils.person_clustering import clustering
def list_video_files(directory):
    # Расширения видеофайлов
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv')
    
    # Проверяем, существует ли указанная директория
    if not os.path.exists(directory):
        print("Указанная директория не существует.")
        return
    ans = []
    # Проходим по всем файлам в директории
    for filename in os.listdir(directory):
        # Проверяем, является ли файл видео
        if filename.lower().endswith(video_extensions):
            ans.append(filename)
    
    return ans

def capture_frame(video_path, frame_number, output_image_path):
    # Открываем видеофайл
    cap = cv2.VideoCapture(video_path)

    # Проверяем, удалось ли открыть видео
    if not cap.isOpened():
        print("Ошибка: Не удалось открыть видео.")
        return

    # Устанавливаем номер кадра, который хотим извлечь
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    # Считываем кадр
    ret, frame = cap.read()

    # Проверяем, удалось ли считать кадр
    if ret:
        # Сохраняем кадр как изображение
        cv2.imwrite(output_image_path, frame)
        print(f"Кадр {frame_number} сохранен как {output_image_path}.")
    else:
        print("Ошибка: Не удалось считать кадр.")

    # Освобождаем объект VideoCapture
    cap.release()


def main():

    # Заголовок приложения
    # st.set_page_config(
    #     layout='wide'
    # )
    st.title("Поиск по фото")

    name_db = 'aaa'

    s = lib.Similar(name_db=name_db)

    uploaded_file = st.file_uploader("Выберите изображение", type=["jpg", "jpeg", "png"])

    video_dict = {
        'Silicon Valley.mp4': '0',
        'Silicon Valley S06E01.2019.KvK.WEB-DLRip.avi': '1'
    }

    col1, col2 = st.columns(2)
    # Проверка, было ли загружено изображение
    if uploaded_file is not None:
        image = np.array(Image.open(uploaded_file))
        with col1:
            st.image(image, width=400)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imwrite('data/temp.jpg', image)

        with col2:
            video_name = st.selectbox("Выберите видео", list(video_dict.keys()))
        video_name_for_db = video_dict[video_name]
        video_path = f'project/data/{video_name}'
        with col2:
            button = st.button('Найти')
        if button:
            frames_idx, timings = s.get_timings('data/temp.jpg', video_path, video_name_for_db, 60)
            st.title("Результат")
            if not timings:
                st.write("Ничего не найдено")
            else:
                dict_col = {i: col for i, col in enumerate(st.columns(3))}
                for i, (frame, time) in enumerate(zip(frames_idx, timings)):
                    frame = s.get_frame(video_path, frame)
                    with dict_col[i % 3]:
                        st.image(frame, width=400)
                        st.write(f"{int(time // 60):02}:{int(time % 60):02}")


    # st.title("Поиск похожих фото по видео")

    # uploaded_images = st.file_uploader("Загрузите изображения", type=["jpg", "png"])
    
    # # suffix = ".avi" if uploaded_file.type == "video/x-msvideo" else ".mp4"
    # if uploaded_images:
    #     with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
    #         temp_file.write(uploaded_images.read())
    #         temp_file_path = temp_file.name


    #     s = Similar(name_db="db")


    #     df_dist = s.find_similar(temp_file_path, "1")

    #     option = st.selectbox(label= "Выбери", options=list_video_files(Path.cwd() / "data"))

    #     frames_idx, timings = s.get_timings(temp_file_path, str(Path.cwd() / "data" / option), "1", 60)
     
    #     for i in len(frames_idx):
    #         st.image(s.get_frame(str(Path.cwd() / "data" / option), frames_idx[i]), caption=timings[i])
        

    # st.header("Ключевые персонажи")
    # if st.button("Рассчитать!"):
    #     output_paths = clustering(
    #         whisper_df_=st.session_state.whisper_data, time_frames=st.session_state.frames, path="/project/data/images"
    #     )
    #     st.image(output_paths)


    

if __name__ == "__main__":
    main()
