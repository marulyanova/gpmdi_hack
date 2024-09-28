# import os
# from pathlib import Path
# import wave

# def extract_wav_chunk(input_file, start_sec, end_sec):
#     """
#     Функция, которая извлекает фрагмент аудиофайла с расширением .wav
#     по заданным начальной и конечной секундам.

#     Параметры:
#     input_file (str или Path) - путь к исходному аудиофайлу .wav
#     start_sec (float) - начальная секунда фрагмента
#     end_sec (float) - конечная секунда фрагмента

#     Возвращает:
#     bytes - байты, содержащие обрезанный фрагмент аудиофайла
#     """
#     input_file = Path(input_file)
#     if not input_file.exists() or not input_file.is_file() or not input_file.suffix.lower() == ".wav":
#         raise ValueError("Указан некорректный путь к файлу .wav")

#     with wave.open(str(input_file), "r") as wav:
#         frame_rate = wav.getframerate()
#         start_frame = int(start_sec * frame_rate)
#         end_frame = int(end_sec * frame_rate)
#         wav.setpos(start_frame)
#         chunk_frames = end_frame - start_frame
#         chunk_data = wav.readframes(chunk_frames)

#     return chunk_data


import os
from pathlib import Path
import wave

def extract_wav_chunk(input_file, start_sec, end_sec, output_file):
    """
    Функция, которая извлекает фрагмент аудиофайла с расширением .wav
    по заданным начальной и конечной секундам и сохраняет его в новый файл.

    Параметры:
    input_file (str или Path) - путь к исходному аудиофайлу .wav
    start_sec (float) - начальная секунда фрагмента
    end_sec (float) - конечная секунда фрагмента
    output_file (str или Path) - путь к новому файлу .wav, куда будет сохранен фрагмент
    """
    input_file = Path(input_file)
    output_file = Path(output_file)

    if not input_file.exists() or not input_file.is_file() or not input_file.suffix.lower() == ".wav":
        raise ValueError("Указан некорректный путь к файлу .wav")

    with wave.open(str(input_file), "r") as wav:
        frame_rate = wav.getframerate()
        start_frame = int(start_sec * frame_rate)
        end_frame = int(end_sec * frame_rate)
        wav.setpos(start_frame)
        chunk_frames = end_frame - start_frame
        chunk_data = wav.readframes(chunk_frames)

    with wave.open(str(output_file), "w") as out_wav:
        out_wav.setnchannels(wav.getnchannels())
        out_wav.setsampwidth(wav.getsampwidth())
        out_wav.setframerate(frame_rate)
        out_wav.writeframes(chunk_data)

    print(f"Фрагмент аудио сохранен в файл: {output_file}")
