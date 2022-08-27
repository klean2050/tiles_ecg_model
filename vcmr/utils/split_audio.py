import os
from tqdm import tqdm

audio_path = "/data/avramidi/video_clip_data_NEW/audios_00_splitted/"

def process_audio(input, output):
    os.system(
        f"ffmpeg -n -loglevel error -threads 0 -i {input} -ar 16000 -ac 1 {output}"
    )

if __name__ == "__main__":
    for code in tqdm(os.listdir(audio_path)[::-1]):
        track_path = os.path.join(audio_path, code)
        new_path = track_path.replace("splitted", "reduced")
        os.makedirs(new_path, exist_ok=True)

        for wav_sample in os.listdir(track_path):
            input = os.path.join(track_path, wav_sample)
            output = os.path.join(new_path, wav_sample)
            process_audio(input, output)