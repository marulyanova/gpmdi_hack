from ultralytics import YOLO
import cv2
import numpy as np

from tqdm import tqdm

model = YOLO("yolov8s.pt")
anomaly_model = YOLO("yolov8s_15epochs_3dataset.pt")


def get_frames(video_path: str, step_seconds: int):
    """Получить фреймы и время фрейма

    Parameters
    ----------
    video_path : str
        путь до видео
    step_seconds : int
        Шаг в сек, с которым мы режем видео

    Returns
    -------
    np.ndarray
        массив, сореджащий фрейм и его время
    """
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    step_frames = int(fps * step_seconds)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    frames = []

    for frame_idx in range(0, total_frames, step_frames):
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        success, frame = video.read()
        if success:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = np.array(frame)
            frames.append((int(frame_idx / fps), frame))
    video.release()

    return frames


def filtered_object(detected_objects, probability):
    """Фильтрация объектов в соответствии с вероятностью и частотой

    Parameters
    ----------
    detected_objects : List
        каждый массив листа это объекты фрейма

    probability : List
        вероятности каждого объекта

    Returns
    -------
    List
        отфильтрованный список объектов
    """
    detected_objects_1d = [item for sublist in detected_objects for item in sublist]
    probability_1d = [item for sublist in probability for item in sublist]

    stats_dict = {}
    for obj, prob in zip(detected_objects_1d, probability_1d):
        if obj not in stats_dict:
            stats_dict[obj] = {"count": 0, "sum_prob": 0.0}
        stats_dict[obj]["count"] += 1
        stats_dict[obj]["sum_prob"] += prob

    if stats_dict == {}:
        return []

    max_count = max(stats_dict[obj]["count"] for obj in stats_dict)

    for obj in stats_dict:
        count = stats_dict[obj]["count"]
        avg_prob = stats_dict[obj]["sum_prob"] / count
        norm_count = count / max_count
        threshold_value = 1 - (0.5 * (norm_count**3) + 0.5 * avg_prob)
        stats_dict[obj].update(
            {
                "avg_prob": avg_prob,
                "norm_count": norm_count,
                "threshold": threshold_value,
            }
        )

    filtered_detected_objects = []
    filtered_probabilities = []

    for objects, probs in zip(detected_objects, probability):
        filtered_objects = []
        filtered_probs = []
        for obj, prob in zip(objects, probs):
            if prob >= stats_dict.get(obj, {}).get("threshold", 0):
                filtered_objects.append(obj)
                filtered_probs.append(prob)
        if filtered_objects:
            filtered_detected_objects.append(filtered_objects)
            filtered_probabilities.append(filtered_probs)
        else:
            filtered_detected_objects.append([])
            filtered_probabilities.append([])

    return filtered_detected_objects


def get_object_from_video(frames_):
    """Получить объекты объектов и их вероятности, объекты запрещенки, боксплоты всего

    Parameters
    ----------
    frames : np.ndarray
        массив, сореджащий фрейм и его время

    Returns
    -------
    typle
        2D лист объектов(объекты на каждом фрейме),
        вероятности этих объектов,
        боксы этих объектов,
        2D лист запрещенки
    """
    object_arr = []
    anomaly_object_arr = []
    probability_arr = []
    boxplot_info = []
    boxplot_info_anomaly = []
    frames = [i[1] for i in frames_]  # достаем картинки без времени
    for frame in tqdm(frames):
        # получаем инфу для обычных объектов
        results = model(frame, verbose=False)[0]
        classes_names = results.names
        classes = results.boxes.cls.cpu().numpy()
        confidences = results.boxes.conf.cpu().numpy()
        boxes = results.boxes.xyxy.cpu().numpy()  # добавил боксплоты, пригодятся

        class_name = [classes_names[int(class_id)] for class_id in classes]
        probability = list(confidences)
        probability_arr.append(probability)
        object_arr.append(class_name)
        boxplot_info.append(list(boxes))
        # инфа о запрещенке
        results_anomaly = anomaly_model(frame, verbose=False)[0]
        classes_names = results_anomaly.names

        classes = results_anomaly.boxes.cls.cpu().numpy()
        scores = results_anomaly.boxes.conf.cpu().numpy()
        boxes_anomaly = (
            results_anomaly.boxes.xyxy.cpu().numpy()
        )  # добавил боксплоты, пригодятся

        threshold = 0.6  # Задайте нужное значение порога
        filtered_classes_anomaly = [
            classes_names[int(class_id)]
            for class_id, score in zip(classes, scores)
            if score >= threshold
        ]
        anomaly_object_arr.append(filtered_classes_anomaly)
        boxplot_info_anomaly.append(boxes_anomaly)

    object_arr = filtered_object(object_arr, probability_arr)
    return object_arr, boxplot_info, anomaly_object_arr, boxplot_info_anomaly
