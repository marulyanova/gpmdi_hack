import cv2
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.metrics import pairwise_distances
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import OPTICS
from typing import List, Tuple
from tqdm import tqdm
from PIL import Image
import os


model_name = "clip-ViT-B-16"
st_model = SentenceTransformer(model_name)


def get_crop_person_photos(
    whisper_df_: pd.DataFrame, time_frames: List[Tuple[int, np.ndarray]]
):
    """Получить кропнутые фотки людей и их секунда

    Parameters
    ----------
    whisper_df_ : pd.DataFrame
        wisper после добавления всех колонок
    time_frames :  List[Tuple[int, np.ndarray]]
        Фреймы и время

    Returns
    -------
    dict
        frames - массив обрезанных фото
        frame_numbers - время фото

    """
    whisper_df = whisper_df_.copy()

    if all(col in whisper_df.columns for col in ["objects", "objects_boxes"]):
        whisper_df["objects"] = whisper_df["objects"].map(
            lambda x: x[0] if len(x) > 0 else []
        )
        whisper_df["objects_boxes"] = whisper_df["objects_boxes"].map(
            lambda x: x[0] if len(x) > 0 else []
        )
    else:
        raise ValueError(
            "Required columns 'objects' and 'objects_boxes' are not present in whisper_df."
        )

    cropped_frames_dict = {
        "frames": [],  # массив для обрезанных фреймов
        "frame_numbers": [],  # массив для номеров фреймов
    }

    for index, row in whisper_df.iterrows():
        frame_number = int(row["start"])
        if frame_number >= len(time_frames):
            break
        frame = time_frames[frame_number][1]  # извлекаем текущий фрейм
        objects = row["objects"]
        boxes = row["objects_boxes"]
        if len(objects) == 0:
            continue

        for obj, box in zip(objects, boxes):
            if obj == "person":
                x_min, y_min, x_max, y_max = box
                cropped_frame = frame[int(y_min) : int(y_max), int(x_min) : int(x_max)]

                # Добавляем обрезанный фрейм и номер фрейма в словарь
                cropped_frames_dict["frames"].append(cropped_frame)
                cropped_frames_dict["frame_numbers"].append(index)
    return cropped_frames_dict


def vectorize_img(image_array, model=st_model):
    if isinstance(image_array, np.ndarray):
        img = Image.fromarray(image_array)
    else:
        raise ValueError("Input must be a NumPy array.")
    return model.encode(img)

def get_crop_embedd(frames):
    embedding = []
    for emb in tqdm(frames):
        embed_frame = vectorize_img(emb)
        embedding.append(embed_frame)
    return embedding


def scaler(emb: pd.Series):
    """
    StandartScaler для ембеддингов
    """
    embeddings = np.vstack(emb.values)
    scaler = StandardScaler()
    scale_emb = scaler.fit_transform(embeddings)
    return scale_emb


def optics_clustering(
    embeddings: np.ndarray,
    min_samples: int = 20,
    xi: float = 0.01,
    min_cluster_size: float = 0.01,
):
    """
    Выполняет кластеризацию с использованием алгоритма OPTICS

    Parameters
    ----------
    embeddings : np.ndarray
        Массив эмбеддингов, где каждая строка представляет собой вектор признаков для одного объекта.

    min_samples : int, по умолчанию 20
        Минимальное количество образцов в группе, необходимое для создания кластера.
        Используется для определения плотности.

    xi : float, по умолчанию 0.01
        Параметр, определяющий, как сильно отделяются кластеры друг от друга.
        Чем больше значение, тем менее агрессивная отделка.

    min_cluster_size : float, по умолчанию 0.01
        Минимальный размер кластера как доля от общего числа точек.
        Если размер кластера меньше этого значения, он не будет считаться кластером.

    Returns
    -------
    np.ndarray
        Массив меток кластеров, где каждая метка соответствует строке в массиве `embeddings`.
        Значение -1 обозначает шум (выбросы).
    """
    optics = OPTICS(
        min_samples=min_samples,
        xi=xi,
        min_cluster_size=min_cluster_size,
        metric="cosine",
    )
    clusters = optics.fit_predict(embeddings)
    return clusters


def get_best_example(stats_df: pd.DataFrame):
    """
    Находит ближайшие объекты до центра кластера для каждого кластера на основе их эмбеддингов.

    Parameters
    ----------
    stats_df : pd.DataFrame
        DataFrame, содержащий эмбеддинги и метки кластеров,
        полученные из функции кластеризации.

    Returns
    -------
    dict
        Словарь, где ключами являются метки кластеров,
        а значениями - индексы объектов, ближайших к центру кластера.
    """
    df_all_labels = np.copy(np.vstack(stats_df["embeddings"].values))
    cluster = np.array(stats_df["cluster"])
    unique_labels = set(list(stats_df["cluster"]))
    unique_labels.discard(-1)

    closest_objects = {}
    for label in unique_labels:
        cluster_indices = np.where(cluster == label)[0]
        cluster_objects = df_all_labels[cluster_indices]
        cluster_center = np.mean(cluster_objects, axis=0)
        distances = pairwise_distances(
            cluster_objects, cluster_center.reshape(1, -1), metric="cosine"
        ).flatten()
        sorted_indices = np.argsort(distances)
        n = 1
        closest_n_objects = cluster_indices[sorted_indices[:n]]
        closest_objects[label] = closest_n_objects
    return closest_objects


def save_main_photo(path: str, closest_objects: dict, cropped_frames_dict: dict):
    """
    Сохраняет фотографии в заданную директорию на основе индексов ближайших объектов.

    Parameters
    ----------
    path : str
        Путь к директории, в которую будут сохранены фотографии.

    closest_objects : dict
        Словарь, где ключами являются метки кластеров,
        а значениями - индексы объектов, которые нужно сохранить.

    cropped_frames_dict : dict
        Словарь, содержащий обрезанные фотографии,
        доступные по индексу.

    Returns
    -------
    List[str]
        Функция сохраняет фотографии и возвращает пути до них.
    """

    os.makedirs(path, exist_ok=True)
    paths = []
    for label, indices in closest_objects.items():
        for index in indices:
            # Извлечение обрезанного изображения по индексу
            # print(index, cropped_frames_dict["frame_numbers"])
            if (
                index in cropped_frames_dict["frame_numbers"]
            ):  # Убедимся, что индекс существует
                image = cropped_frames_dict["frames"][index]  # Получаем изображение
                if image.shape[0] > 300 and image.shape[1] > 270:
                    # Формируем имя файла
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    base_filename = f"cluster_{label}_object_{index}.jpg"
                    filename = os.path.join(path, base_filename)

                    # Проверка существования файла и добавление суффикса _{k} при необходимости
                    k = 0
                    while os.path.exists(filename):
                        k += 1
                        filename = os.path.join(
                            path, f"cluster_{label}_object_{index}_{k}.jpg"
                        )
                    cv2.imwrite(filename, image)  # Сохраняем изображение
                    paths.append(filename)
    return paths


def clustering(
    whisper_df_: pd.DataFrame, time_frames: List[Tuple[int, np.ndarray]], path: str
):
    """
    Выполняет кластеризацию обрезанных фотографий людей на основе их эмбеддингов. Сохраняет фотографии

    Parameters
    ----------
    whisper_df_ : pd.DataFrame
        wisper после добавления всех колонок
    time_frames :  List[Tuple[int, np.ndarray]]
        Фреймы и время

    Returns
    -------
    Функция сохраняет фотографии в указанной директории. И возвращает пути
    """
    cropped_frames_dict = get_crop_person_photos(whisper_df_, time_frames)
    embeddings = get_crop_embedd(cropped_frames_dict["frames"])
    # embeddings = get_crop_embedd()

    embeddings = np.array(embeddings)
    stats_df = pd.DataFrame({"embeddings": list(embeddings)})
    embeddings = scaler(stats_df["embeddings"])

    clusters = optics_clustering(embeddings)
    stats_df["cluster"] = list(clusters)
    closest_objects = get_best_example(stats_df)
    paths = save_main_photo(path, closest_objects, cropped_frames_dict)
    return paths
