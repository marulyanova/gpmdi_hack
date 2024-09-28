import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
from sentence_transformers import SentenceTransformer
from sklearn.metrics import pairwise_distances
import pandas as pd
from numba import njit
from datetime import timedelta
from scipy.stats import mode


model_name = "clip-ViT-B-16"
st_model = SentenceTransformer(model_name)


def frames_to_time(frame_idx, fps):
    """Преобразует индекс кадра в строку времени в формате hh:mm:ss"""
    total_seconds = frame_idx / fps
    time_str = str(timedelta(seconds=int(total_seconds)))
    return time_str


# refactor
def extract_frames_with_time(video_path, step_seconds, time_list_in_seconds):
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    step_frames = int(fps * step_seconds)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Преобразуем список времени (в секундах) в кадры
    time_frames = [int(time * fps) for time in time_list_in_seconds]

    # Уникальные кадры с шагом 2 + кадры из списка времени
    # frames_to_capture = sorted(set(time_frames))
    frames_to_capture = sorted(
        set(range(0, total_frames, step_frames)) | set(time_frames)
    )
    frames_with_time = []

    for frame_idx in tqdm(frames_to_capture):
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        success, frame = video.read()
        if success:
            frame_time = frames_to_time(frame_idx, fps)  # Преобразуем кадр в время
            frames_with_time.append((frame, frame_time))

    video.release()
    return frames_with_time


def vectorize_img(image_array, model=st_model):
    if isinstance(image_array, np.ndarray):
        img = Image.fromarray(image_array)
    else:
        raise ValueError("Input must be a NumPy array.")
    return model.encode(img)


def get_embedd(frames_time):
    embedding = []
    frames = [i[1] for i in frames_time]
    for emb in tqdm(frames):
        embed_frame = vectorize_img(emb)
        embedding.append(embed_frame)
    return embedding


def find_closest_t(cosine_distance_matrix, t_start=0.05, t_end=0.95, t_step=0.05):
    t_values = np.arange(t_start, t_end + t_step, t_step)
    ones_counts = []

    for t in t_values:
        cosine_distance_matrix2 = (cosine_distance_matrix > t).astype(int)
        ones_count = np.sum(cosine_distance_matrix2)  # Подсчет единиц
        ones_counts.append(ones_count)

    min_value = np.min(ones_counts)
    max_value = np.max(ones_counts)
    scaled_ones_counts = (ones_counts - min_value) / (max_value - min_value)

    closest_index = (np.abs(scaled_ones_counts - 0.5)).argmin()
    corresponding_t = t_values[closest_index]

    return corresponding_t


@njit
def count_different_neighbors(matrix, i, j):
    different_neighbors = 0
    rows, cols = matrix.shape
    for x in range(max(0, i - 1), min(rows, i + 2)):
        for y in range(max(0, j - 1), min(cols, j + 2)):
            if (x != i or y != j) and matrix[x, y] != matrix[i, j]:
                different_neighbors += 1
    return different_neighbors


# Шаг заражения
@njit
def infection_step(matrix):
    new_matrix = matrix.copy()  # Используем NumPy для копирования массива
    rows, cols = matrix.shape
    for i in range(rows):
        for j in range(cols):
            different_neighbors = count_different_neighbors(matrix, i, j)
            if different_neighbors >= 4:
                new_matrix[i, j] = (
                    1 - matrix[i, j]
                )  # Меняем значение ячейки на противоположное
    return new_matrix


# Функция для подсчета нулей и единиц (без Numba, так как она работает быстро)
def count_zeros_ones(matrix):
    zeros = np.sum(matrix == 0)
    ones = np.sum(matrix == 1)
    return zeros, ones


# Основная функция для многократного заражения
def multi_infection(matrix, N):
    current_matrix = matrix
    for step in range(N):
        # print(f"\nШаг {step + 1}:")
        # Подсчет количества нулей и единиц до заражения
        zeros_before, ones_before = count_zeros_ones(current_matrix)

        # Заражение
        new_matrix = infection_step(current_matrix)

        # Подсчет количества нулей и единиц после заражения
        zeros_after, ones_after = count_zeros_ones(new_matrix)

        # Процент изменений
        zero_change_percent = (
            ((zeros_after - zeros_before) / zeros_before * 100)
            if zeros_before > 0
            else 0
        )
        one_change_percent = (
            ((ones_after - ones_before) / ones_before * 100) if ones_before > 0 else 0
        )

        # Вывод результатов
        # print(f"До заражения: 0 - {zeros_before}, 1 - {ones_before}")
        # print(f"После заражения: 0 - {zeros_after}, 1 - {ones_after}")
        # print(f"Изменение: 0 -> {zero_change_percent:.2f}%, 1 -> {one_change_percent:.2f}%")

        # Если изменения незначительные, останавливаем процесс
        if (zero_change_percent < 0.01) and (one_change_percent < 0.01):
            break

        # Обновляем текущую матрицу для следующего шага
        current_matrix = new_matrix

    return current_matrix


# Функция для скользящей моды
def sliding_mode(arr, window_size):
    result = []
    for i in range(len(arr)):
        start = max(0, i - window_size // 2)
        end = min(len(arr), i + window_size // 2 + 1)
        window_mode = mode(arr[start:end], keepdims=True).mode[
            0
        ]  # Используем keepdims=True
        result.append(window_mode)
    return np.array(result)


def seconds_to_time(seconds):
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02}:{m:02}:{s:02}"


# Функция для преобразования времени в секунды
def time_to_seconds(time_str):
    h, m, s = map(int, time_str.split(":"))
    return h * 3600 + m * 60 + s


def adjust_values_based_on_eps(original_values, eps):
    # Копируем оригинальный массив для создания нового с заменами
    adjusted_values = np.copy(original_values)

    # Проходим по массиву
    for i in range(len(original_values) - 1):
        # Сравниваем соседние элементы на оригинальном массиве
        if abs(original_values[i + 1] - original_values[i]) <= eps:
            # Если разница меньше или равна eps, заменяем следующий элемент
            adjusted_values[i + 1] = adjusted_values[i]

    return adjusted_values


def get_clips(embedding, frames_time, eps: int = 9):
    print([i[0] for i in frames_time])

    cosine_distance_matrix = pairwise_distances(embedding, metric="cosine")

    corresponding_t = find_closest_t(cosine_distance_matrix)

    cosine_distance_matrix2 = (cosine_distance_matrix > corresponding_t).astype(int)

    smooth2 = multi_infection(cosine_distance_matrix2, 50)

    n = smooth2.shape[0]

    # Список для хранения средних значений по строкам и столбцам
    average_values = []

    # Пройтись по всем элементам диагонали
    for i in range(n):
        # Извлекаем соответствующие строку и столбец, исключая элемент на диагонали
        row_sum = np.sum(smooth2[i, :])
        col_sum = np.sum(smooth2[:, i])

        # Среднее значение по строке и столбцу
        mean_value = (row_sum + col_sum) / (
            2 * (n - 1)
        )  # Двухкратное исключение диагонального элемента
        average_values.append(mean_value)

    smoothed_values = sliding_mode(np.array(average_values), 10)

    shift = 5
    kek_times = [i[0] for i in frames_time]
    # Массив для хранения временных значений
    time_values = []

    # Собираем значения времени со смещением
    for idx in np.where(smoothed_values > 0.5)[0]:
        shifted_idx = max(
            0, idx - shift
        )  # Учитываем смещение, не выходя за пределы массива
        time_values.append(kek_times[shifted_idx])

    # Преобразуем временные значения в секунды
    # time_values_in_seconds = [t for t in time_values]
    window_size = 10
    time_values_in_seconds = sliding_mode(np.array(time_values), window_size)
    # Построение графика

    # Пример использования

    adjusted_time_values = adjust_values_based_on_eps(time_values_in_seconds, eps)
    # print("Оригинальный массив:", time_values_in_seconds)

    time_final = np.unique([seconds_to_time(t) for t in adjusted_time_values])
    print(kek_times)

    return time_final
