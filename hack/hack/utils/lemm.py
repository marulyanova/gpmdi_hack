import numpy as np
import spacy

import pandas as pd
from collections import Counter
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer


# nlp = spacy.load('ru_core_news_sm')
# nlp.add_pipe("textrank")


def time_to_seconds(time_str):
    h, m, s = map(int, time_str.split(':'))
    return h * 3600 + m * 60 + s


def get_closest_indices(df: pd.DataFrame, scene_cutter_output: List[str]):
    """
    Возвращает индексы строк, где значения в df["start"] максимально близки и меньше значений в scene_cutter_output_sec.

    Parameters
    ----------
    df : pd.DataFrame
        Датафрейм, содержащий колонку "start".
    scene_cutter_output_sec : list
        Время сцен

    Returns
    -------
    list
        Индексы строк, где значения в df["start"] меньше и максимально близки к значениям в scene_cutter_output_sec.
    """
    scene_cutter_output_sec = [time_to_seconds(i) for i in scene_cutter_output]
    indices = [0]
    start_values = df["start"].values
    
    for sec in scene_cutter_output_sec:
        valid_indices = np.where(start_values <= sec)[0]    
        if len(valid_indices) > 0:
            # closest_idx = start_values[np.argmax(start_values[valid_indices])]
            closest_idx = np.argmax(start_values[valid_indices])
            indices.append(closest_idx)
        else:
            indices.append(None)
    if len(start_values) - 1 not in indices:
        indices.append(len(start_values) - 1)
    return list(np.unique(indices))

def combine_text_by_indices(df: pd.DataFrame, indices: list):
    """
    Объединяет строки из колонки df["text"] на основе диапазонов индексов.

    Parameters
    ----------
    df : pd.DataFrame
        Датафрейм, содержащий колонку "text".
    indices : list
        Список индексов, где каждая пара индексов [i, j] определяет диапазон строк для объединения.
        Верхняя граница не включается.

    Returns
    -------
    list
        Список объединенных строк текста между указанными индексами.
    """
    combined_texts = []
    
    for i in range(len(indices) - 1):
        start_idx = indices[i]
        end_idx = indices[i + 1]
        combined_text = " ".join(df.loc[start_idx:end_idx-1, "text"].tolist())
        combined_texts.append(combined_text)
    
    return combined_texts

# def lemmatize_text(text):
#     """
#     Лемматизирует текст, приводя все слова к начальной форме.
    
#     Parameters
#     ----------
#     text : str
#         Исходный текст для лемматизации.
    
#     Returns
#     -------
#     str
#         Лемматизированный текст.
#     """
#     doc = nlp(text)
#     lemmatized_text = " ".join([token.lemma_ for token in doc])
#     return lemmatized_text

# def extract_keywords(texts):
#     """
#     Лемматизирует и извлекает ключевые слова из массива строк с помощью spaCy и TextRank.

#     Parameters
#     ----------
#     texts : list
#         Список строк для обработки.

#     Returns
#     -------
#     list
#         Список ключевых слов для каждой строки.
#     """
#     keyword_list = []

#     for text in texts:
#         lemmatized_text = lemmatize_text(text)
#         doc = nlp(lemmatized_text)
#         words = [phrase.text for phrase in doc._.phrases]
#         keyword_list.append(words)

    # return keyword_list

def get_top_objects(df_, scene_indices, n=6):
    """
    Объединяет объекты из колонки df["objects"] по заданным индексам,
    создает колонку с классами, подсчитывает вхождения и возвращает
    три самых частых предмета для каждой сцены.

    Parameters
    ----------
    df_ : pd.DataFrame
        Датафрейм, содержащий колонку "objects" с массивами строк.
    
    scene_indices : list of list
        Список индексов строк для каждой сцены.
    
    n : int
        Количество главных предметов, которые нужно вернуть.

    Returns
    -------
    dict
        Словарь, где ключами являются номера классов, а значениями — массивы с
        главными n предметами для каждой сцены.
    """
    df = df_.copy()
    df["objects"] = df["objects"].map(lambda x: x if isinstance(x, list) else [])
    
    # Результаты
    top_objects_per_class = []

    for i in range(len(scene_indices) - 1):
        all_objects = []
        start_idx = scene_indices[i]
        end_idx = scene_indices[i + 1]
        objects_for_class = df.iloc[start_idx:end_idx]["objects"]
        for obj_list in objects_for_class:
            all_objects.extend(obj_list) 
        all_objects = [item for sublist in all_objects for item in sublist]
        object_counts = Counter(all_objects)

        most_common_objects = object_counts.most_common(n)
        top_objects = [obj for obj, count in most_common_objects]

        top_objects_per_class.append(top_objects)

    return top_objects_per_class


def apply_tfidf_to_objects(df: pd.DataFrame, column: str, top_n: int = 4) -> pd.DataFrame:
    """
    Применяет tf-idf для выявления ключевых объектов и создает новый столбец с важными объектами.

    Parameters
    ----------
    df : pd.DataFrame
        Датафрейм, содержащий колонку с массивами объектов.
    
    column : str
        Название колонки, содержащей массивы объектов.
    
    top_n : int
        Количество ключевых объектов, которые нужно вернуть.

    Returns
    -------
    pd.DataFrame
        Датафрейм с новым столбцом, содержащим важные объекты.
    """
    df['objects_strings'] = df[column].apply(lambda x: ' '.join(x))
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df['objects_strings'])
    feature_names = vectorizer.get_feature_names_out()
    important_objects = []
    for row in tfidf_matrix.toarray():
        top_indices = row.argsort()[-top_n:][::-1]  # Индексы топ-n объектов
        important_objects.append([feature_names[i] for i in top_indices])
    df['important_objects'] = important_objects
    df = df.drop(columns=["objects_strings", "top_objects"])
    return df


def get_main_emotions(df_: pd.DataFrame, scene_indices) -> pd.DataFrame:
    """
    Определяет главную эмоцию для каждого диапазона индексов в колонке 'emotionals',
    основываясь на частоте вхождения.

    Parameters
    ----------
    df_ : pd.DataFrame
        Датафрейм, содержащий колонку 'emotionals' с эмоциями.
    
    scene_indices : list of list
        Список индексов строк для каждой сцены.

    Returns
    -------
    Lisr[str]
        Главные эмоции для каждого диапазона.
    """
    df = df_.copy()
    main_emotions = []



    for i in range(len(scene_indices) - 1):
        all_emotions = []
        start_idx = scene_indices[i]
        end_idx = scene_indices[i + 1]
        emotions_for_class = df.iloc[start_idx:end_idx]["emotionals"]
        for obj_list in emotions_for_class:
            all_emotions.append(obj_list) 
        emotions_counts = Counter(all_emotions)

        if emotions_counts:
            main_emotion = emotions_counts.most_common(1)[0][0]  # Получаем наиболее частую эмоцию
        else:
            main_emotion = None  # Если эмоций нет, присваиваем None
        
        main_emotions.append(main_emotion)

    return main_emotions


def get_scene_info(wisper_df_: pd.DataFrame, scene_cutter_output: List[str]) -> pd.DataFrame:
    """
    Получает информацию о сценах из датафрейма и массива разметки.

    Parameters
    ----------
    wisper_df_ : pd.DataFrame
        Исходный датафрейм, содержащий информацию о видео, включая колонки с текстами и временными метками.
    
    scene_cutter_output : List[str]
        Список строк, представляющий временные метки для разбиения видео на сцены.

    Returns
    -------
    pd.DataFrame
        Датафрейм с двумя колонками:
        - 'keywords': массивы ключевых слов, извлеченных из текста для каждой сцены.
        - 'top_objects': массивы самых частых объектов, обнаруженных в каждой сцене.
    """
    wisper_df = wisper_df_.copy()
    wisper_df["start"] = wisper_df["start"].astype(int)
    
    # Получаем ближайшие индексы и текст для сцен
    closest_indices = get_closest_indices(wisper_df, scene_cutter_output)
    # text_from_scenes = combine_text_by_indices(wisper_df, closest_indices)
    # key_keywords = extract_keywords(text_from_scenes)
    top_objects = get_top_objects(wisper_df, closest_indices)
    main_emotions = get_main_emotions(wisper_df, closest_indices)
    # Создаем DataFrame из двух двумерных массивов
    scene_info_df = pd.DataFrame({
        # "keywords": key_keywords,
        "top_objects": top_objects,
        "emotion": main_emotions
    })
    scene_info_df = apply_tfidf_to_objects(scene_info_df, "top_objects")
    scene_info_df.index = [f"scene_{i}" for i in range(len(scene_info_df))]
    return scene_info_df
