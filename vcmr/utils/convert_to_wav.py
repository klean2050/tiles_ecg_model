import os, subprocess


def preprocess_audio(source, target, sample_rate):
    p = subprocess.Popen(
        ["ffmpeg", "-i", source, "-ar", str(sample_rate), target, "-loglevel", "quiet"]
    )
    p.wait()


for i in os.listdir("/data/avramidi/mtg"):
    print(i)
    for j in os.listdir(f"/data/avramidi/mtg/{i}"):
        preprocess_audio(
            f"/data/avramidi/mtg/{i}/{j}", f"/data/avramidi/mtg/{i}/{j[:-3]}wav", 22050
        )
        os.remove(f"/data/avramidi/mtg/{i}/{j}")
    break
