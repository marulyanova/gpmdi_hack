import cv2
from tqdm.notebook import tqdm
import numpy as np

import streamlit as st

def make_poi_video(video_path: str, step_seconds: float = 0.5):    
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)  # Частота кадров
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))  # Ширина кадров
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Высота кадров
    step_frames = int(fps * step_seconds)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Создаем объект VideoWriter для записи нового видео
    output_video_path = "data/output.mp4"  # Измените на .mp4
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Кодек для MP4
    out = cv2.VideoWriter(output_video_path, fourcc, fps // step_frames, (width, height))

    success, first_frame = video.read()
    if not success:
        print("Не удалось прочитать первый кадр.")
        return

    feature_params = dict(maxCorners = 300, qualityLevel = 0.2, minDistance = 2, blockSize = 7)
    lk_params = dict(winSize = (15,15), maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    prev = cv2.goodFeaturesToTrack(prev_gray, mask = None, **feature_params)
    mask = np.zeros_like(first_frame)
    color = (0, 255, 0)

    for frame_idx in tqdm(range(1, total_frames, step_frames)):
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        success, frame = video.read()
        if not success:
            print(f"Не удалось прочитать кадр {frame_idx}.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prev_gray = gray.copy()
        prev = cv2.goodFeaturesToTrack(prev_gray, mask = None, **feature_params)
        next, status, error = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev, None, **lk_params)
        # Selects good feature points for previous position
        good_old = prev[status == 1].astype(int)
        # Selects good feature points for next position
        good_new = next[status == 1].astype(int)
        # Draws the optical flow tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            # Returns a contiguous flattened array as (x, y) coordinates for new point
            a, b = new.ravel()
            # Returns a contiguous flattened array as (x, y) coordinates for old point
            c, d = old.ravel()
            # Draws line between new and old position with green color and 2 thickness
            # mask = cv2.line(mask, (a, b), (c, d), color, 2)
            # Draws filled circle (thickness of -1) at new position with green color and radius of 3
            frame = cv2.circle(frame, (a, b), 7, color, -1)
        # Overlays the optical flow tracks on the original frame
        output = cv2.add(frame, mask)
        # Updates previous frame
        # prev_gray = gray.copy()
        # Updates previous good feature points
        # prev = good_new.reshape(-1, 1, 2)
        # Opens a new window and displays the output frame
        out.write(output)

    video.release()
    out.release()
    cv2.destroyAllWindows()
    print("Обработка завершена. Результат записан в:", output_video_path)

# Пример вызова функции
make_poi_video(step_seconds=0.5)


def main():
    st.title("Видео по точкаМ))))")

    


if __name__ == "__main__":
    main()
