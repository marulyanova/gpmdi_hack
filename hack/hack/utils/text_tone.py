from typing import List, Dict, Any
from transformers import pipeline

pipe_inappropriate = pipeline(
    "text-classification", model="apanc/russian-inappropriate-messages"
)


def reveal_inappropriate_lexicon(text: List[str]) -> List[Dict[str, Any]]:
    response = pipe_inappropriate(text)
    return response


label_words = {"LABEL_1": "Непристойный текст", "LABEL_0": "Нормальный текст"}


def label_to_words(response: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    for i, elem in enumerate(response):
        response[i]["label"] = label_words[response[i]["label"]]
        response[i]["score"] = round(response[i]["score"], 2)

    return response


def get_inappropriate(text: List[str]) -> List[float]:
    response = label_to_words(reveal_inappropriate_lexicon(text))
    result = []
    for i, elem in enumerate(text):
        result.append([response[i]["label"], response[i]["score"]])
    return result


if __name__ == "__main__":
    text = [
        "Ладно бы видного деятеля завалили а тут какого то ноунейм нигру преступника",
        "сука блять",
        "хуй",
        "я был в магазине",
        "хер ей а не моя квартира",
        "пизда тупая",
        "черным не место в нашей стране",
        "понаехали чурки, негде нормально вздохнуть",
        "Вам сюда. Налево, пожалуйста.",
        "Красивый галстук.",
        "Спасибо. Сам повязал.",
    ]

    get_inappropriate(text)
