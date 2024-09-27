# !pip install audio-extract -q

from audio_extract import extract_audio

def extract_fragment_from_audio(start: str, end: str, input_path: str = None, output_path: str = None) -> None:

    duration = 0.0
    end_min, end_sec = map(int, [end.split(':')[0], end.split(':')[1]])
    start_min, start_sec = map(int, [start.split(':')[0], start.split(':')[1]])
    duration = (end_min - start_min) * 60 - abs(end_sec - start_sec) if end_sec < start_sec else (end_min - start_min) * 60 + abs(end_sec - start_sec)

    Trim audio:
    extract_audio(input_path=input_path,
                  output_path=output_path,
                  start_time=start,
                  duration=duration) # seconds
