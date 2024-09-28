import torch
import torchaudio
from transformers import HubertForSequenceClassification, Wav2Vec2FeatureExtractor

from pydub import AudioSegment
from pydub.silence import split_on_silence
import io
from typing import List

from pathlib import Path

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-large-ls960-ft")
model = HubertForSequenceClassification.from_pretrained("xbgoose/hubert-speech-emotion-recognition-russian-dusha-finetuned")
model = model.to(device)

num2emotion = {0: 'neutral', 1: 'angry', 2: 'positive', 3: 'sad', 4: 'other'}

FILE_PATH = Path.cwd().parent.parent / "data" / "external" / "input_car.wav"


def split_audio_into_chunks(filepath: str, silence_range: int = 700) -> List:

    sound = AudioSegment.from_file(filepath)

    # разбиваем аудио по тишине (700ms и больше)
    chunks = split_on_silence(sound, min_silence_len=silence_range, silence_thresh=sound.dBFS-14, keep_silence=silence_range)

    return chunks


def recognize_emotion(chunk) -> str:

    waveform, sample_rate = torchaudio.load(chunk, normalize=True)
    transform = torchaudio.transforms.Resample(sample_rate, 16000)
    waveform = transform(waveform)

    inputs = feature_extractor(
            waveform,
            sampling_rate=feature_extractor.sampling_rate,
            return_tensors="pt",
            padding=True,
            max_length=16000 * 10,
            truncation=True
        )

    inputs = inputs.to(device)
    
    logits = model(inputs['input_values'][0]).logits
    predictions = torch.argmax(logits, dim=-1)
    predicted_emotion = num2emotion[predictions.cpu().numpy()[0]]
  

    return predicted_emotion


def emotions_to_chunks(chunks: List) -> List:

    emotions = []

    for i, audio_chunk in enumerate(chunks, start=1):
        buffer = io.BytesIO()
        audio_chunk.export(buffer, format="wav")
        buffer.seek(0)
        emotions.append(recognize_emotion(buffer))
    return emotions

def emotions_to_chunk(file_path: str) -> List:

    file_path = Path(file_path)

    if not file_path.exists() or not file_path.is_file():
        raise ValueError("Указан некорректный путь к файлу")

    with open(file_path, "rb") as file:
        file_bytes = file.read()
        return recognize_emotion(file_bytes)
    # return emotions


def get_emotional_pipeline(filepath: str) -> List:

    # chunks = split_audio_into_chunks(filepath)
    emotions_list = emotions_to_chunk(filepath)
    return emotions_list

if __name__ == "__main__":
    print(get_emotional_pipeline(FILE_PATH))