import os
from tqdm import tqdm

audio_path = "/data/avramidi/video_clip_data_NEW/audios_00_clipped/"

def split_audio(source, code):
    new_path = os.path.join(audio_path.replace("clipped", "splitted"), code)
    os.makedirs(new_path, exist_ok=True)
    os.system(f"ffmpeg -n -loglevel error -threads 1 -i {source} -segment_time 00:00:15 -f segment -ar 22050 -ac 1 {new_path}/{code}-%03d.wav")

if __name__ == "__main__":
    for i in tqdm(os.listdir(audio_path)):
        code = i.replace("audio-", "").replace(".wav", "")
        assert len(code) == 11
        source = os.path.join(audio_path, i)
        split_audio(source, code)