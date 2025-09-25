import os, librosa, numpy as np
from pathlib import Path

RAW = Path("data/raw")
OUT = Path("data/mels")
OUT.mkdir(parents=True, exist_ok=True)
SR = 16000

def save_mel(in_file, out_file):
    y, _ = librosa.load(in_file, sr=SR)
    mel = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=64)
    logmel = librosa.power_to_db(mel)
    np.savez_compressed(out_file, feat=logmel)

def main():
    for cls in ["normal","anomaly"]:
        for f in (RAW/cls).glob("*.wav"):
            out = OUT/f"{cls}_{f.stem}.npz"
            save_mel(f, out)
    print("Mel features stored in", OUT)

if __name__ == "__main__":
    main()
