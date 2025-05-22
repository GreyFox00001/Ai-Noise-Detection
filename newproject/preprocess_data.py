import os
import numpy as np
import librosa
import pandas as pd

# Load dataset metadata
data_path = "UrbanSound8K/metadata/UrbanSound8K.csv"
df = pd.read_csv(data_path)

X, y = [], []
for index, row in df.iterrows():
    file_path = f"UrbanSound8K/audio/fold{row['fold']}/{row['slice_file_name']}"
    if os.path.exists(file_path):
        y_audio, sr = librosa.load(file_path, sr=22050)
        mfccs = librosa.feature.mfcc(y=y_audio, sr=sr, n_mfcc=40, hop_length=512)
        X.append(np.mean(mfccs, axis=1))
        y.append(row["classID"])

# Save processed data
np.save("X.npy", np.array(X))
np.save("y.npy", np.array(y))
print("Preprocessing complete! Features saved.")
