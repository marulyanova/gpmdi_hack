import moviepy.editor as mp
import pandas as pd
import whisper
from pydub import AudioSegment


model = whisper.load_model("base")


# def save_audio(path: str):
#     audio = AudioSegment.from_file(path)

#     new_path = path[:-3] + "wav"
#     audio.export(new_path, format="wav")

#     return new_path


def save_audio(path: str):
    video = mp.VideoFileClip(path)
    new_path = path[:-3] + "wav"
    video.audio.write_audiofile(new_path)
    return new_path


def get_text_stats(path: str):
    new_path = save_audio(path)
    result = model.transcribe(new_path)
    return pd.DataFrame(result["segments"])
