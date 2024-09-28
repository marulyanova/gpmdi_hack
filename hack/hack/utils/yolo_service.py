from ultralytics import YOLO
import cv2

model = YOLO("yolov8s.pt")

def get_video_params(capture):
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    return fps, total_frames

def get_object_from_video(input_video_path: str, frame_rate: float, start_time: float, end_time: float):
    frame_rate = int(frame_rate * 1000)
    capture = cv2.VideoCapture(input_video_path)
    if not capture.isOpened():
        raise ValueError(f"Не удалось открыть видеофайл: {input_video_path}")

    fps, total_frames = get_video_params(capture)

    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)

    object = []
    probability_arr = []

    for frame_idx in range(start_frame, end_frame, frame_rate):
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        success, frame = capture.read()
        if not success:
            break
        results = model(frame, verbose=False)[0]
        classes_names = results.names

        classes = results.boxes.cls.cpu().numpy()
        confidences = results.boxes.conf.cpu().numpy()

        class_name = [classes_names[int(class_id)] for class_id in classes]
        probability = list(confidences)

        probability_arr.append(probability)
        object.append(class_name)

    capture.release()
    return object, probability_arr


def filtered_object(input_video_path: str, frame_rate: float, threshold: float, start_time: float, end_time: float):
    detected_objects, probability = get_object_from_video(input_video_path, frame_rate, start_time, end_time)
    detected_objects_1d = [item for sublist in detected_objects for item in sublist]
    probability_1d = [item for sublist in probability for item in sublist]

    stats_dict = {}
    for obj, prob in zip(detected_objects_1d, probability_1d):
        if obj not in stats_dict:
            stats_dict[obj] = {'count': 0, 'sum_prob': 0.0}
        stats_dict[obj]['count'] += 1
        stats_dict[obj]['sum_prob'] += prob
    
    if stats_dict == {}:
        return []

    max_count = max(stats_dict[obj]['count'] for obj in stats_dict)

    for obj in stats_dict:
        count = stats_dict[obj]['count']
        avg_prob = stats_dict[obj]['sum_prob'] / count
        norm_count = count / max_count
        threshold_value = 1 - (0.5 * (norm_count ** 3) + 0.5 * avg_prob)
        stats_dict[obj].update({
            'avg_prob': avg_prob,
            'norm_count': norm_count,
            'threshold': threshold_value
        })

    filtered_detected_objects = []
    filtered_probabilities = []

    for objects, probs in zip(detected_objects, probability):
        filtered_objects = []
        filtered_probs = []
        for obj, prob in zip(objects, probs):
            if prob >= stats_dict.get(obj, {}).get('threshold', 0):
                filtered_objects.append(obj)
                filtered_probs.append(prob)
        if filtered_objects:
            filtered_detected_objects.append(filtered_objects)
            filtered_probabilities.append(filtered_probs)
        else:
            filtered_detected_objects.append([])
            filtered_probabilities.append([])
    return filtered_detected_objects

