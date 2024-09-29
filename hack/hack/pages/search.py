import streamlit as st
import time
import lib
import cv2
from PIL import Image
import numpy as np

# Заголовок приложения
st.set_page_config(
    layout='wide'
)
st.title("Поиск по фото")

name_db = 'aaa'

s = lib.Similar(name_db=name_db)

uploaded_file = st.file_uploader("Выберите изображение", type=["jpg", "jpeg", "png"])

video_dict = {
    'Silicon Valley S01E01.2014.KvK.BDRip.avi': '0',
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
    video_path = f'data/{video_name}'
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

