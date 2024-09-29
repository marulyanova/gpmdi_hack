import pymorphy3
import re
import pandas as pd
from pathlib import Path

morph = pymorphy3.MorphAnalyzer()


def file_openner(filepath):
    keywords = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f.readlines():
            if line == "" or line == "\n":
                continue
            line = line.replace("\n", "")
            if line[-1] == " ":
                line = line[:-1]
            keywords += [line]
    return keywords


def lemmatize_sentence(sentence):
    lemmatize_sentence = []

    for word in sentence.split(" "):
        if word == "":
            continue
        word = word.lower()
        parsed = morph.parse(word)
        lemmatize_sentence += [parsed[0].normal_form]

    return " ".join(lemmatize_sentence)


def lemmatize_file(filepath):
    keywords = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f.readlines():
            keywords += [lemmatize_sentence(line.replace("\n", ""))]

    return keywords


BLACK = file_openner(Path.cwd() / "project" / "data" / "black_lemmatized.txt")
GREY = file_openner(Path.cwd() / "project" / "data" / "grey_lemmatized.txt")
PLUS18 = file_openner(Path.cwd() / "project" / "data" / "18plus_lemmatized.txt")


def get_morph(text):
    lemmatize_text = [lemmatize_sentence(i) for i in text]
    print(len(lemmatize_text))
    ans = []
    for i in lemmatize_text:
        dictionary = {"black": [], "grey": [], "18+": []}
        result = []
        for j in i.split(" "):
            if len(re.findall(r"[а-яА-ЯёЁa-zA-Z]+", j)) < 2:
                continue
            j = re.findall(r"[а-яА-ЯёЁa-zA-Z]+", j)[0]
            if j in BLACK and j not in GREY:
                dictionary["black"] += [j]
            if j in GREY:
                dictionary["grey"] += [j]
            if j in PLUS18:
                dictionary["18+"] += [j]

            result.append([i, dictionary])
        ans.append(result)
    return ans
