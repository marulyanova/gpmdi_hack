import streamlit as st
import tempfile
import os
import cv2

from pathlib import Path

from utils.photo_search import Similar


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
    st.title("Поиск похожих фото по видео")

    uploaded_images = st.file_uploader("Загрузите изображения", type=["jpg", "png"])
    
    # suffix = ".avi" if uploaded_file.type == "video/x-msvideo" else ".mp4"
    if uploaded_images:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file.write(uploaded_images.read())
            temp_file_path = temp_file.name


        s = Similar(name_db="db")


        df_dist = s.find_similar(temp_file_path, "1")

        option = st.selectbox(label= "Выбери", options=list_video_files(Path.cwd() / "data"))

        frames_idx, timings = s.get_timings(temp_file_path, str(Path.cwd() / "data" / option), "1", 60)

        st.image([s.get_frame(str(Path.cwd() / "data" / option), frame_id) for frame_id in frames_idx])

if __name__ == "__main__":
    main()
