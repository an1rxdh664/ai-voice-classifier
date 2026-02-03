import librosa
import numpy as np
import sys

def extract_features(file_path):
    try:
        # Load audio file
        y, sr = librosa.load(file_path, sr=None, mono=True)
        
        # Ensure we have audio data
        if y is None or len(y) == 0:
            raise ValueError("No audio data found")

        # MFCC - Mean, Std, Var, Mean Difference
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_feat = [
            float(np.mean(mfcc)),
            float(np.std(mfcc)),
            float(np.var(mfcc)),
            float(np.mean(np.diff(mfcc)))
        ]

        # ZCR - Mean, Std, Var, Mean Difference
        zcr = librosa.feature.zero_crossing_rate(y)
        zcr_feat = [
            float(np.mean(zcr)), 
            float(np.std(zcr)),
            float(np.var(zcr)),
            float(np.mean(np.diff(zcr)))
        ]

        # Spectral Centroid - Mean, Std, Var, Mean Difference
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        centroid_feat = [
            float(np.mean(centroid)), 
            float(np.std(centroid)),
            float(np.var(centroid)),
            float(np.mean(np.diff(centroid)))
        ]

        # RMS - Mean, Std, Var, Mean Difference
        rms = librosa.feature.rms(y=y)
        rms_feat = [
            float(np.mean(rms)), 
            float(np.std(rms)),
            float(np.var(rms)),
            float(np.mean(np.diff(rms)))
        ]

        # Pitch (pYIN) - Mean, Std, Var, Mean Difference
        try:
            f0, voiced_flag, _ = librosa.pyin(y, fmin=50, fmax=300, sr=sr)
            pitch = f0[voiced_flag]
            if len(pitch) > 0:
                pitch_feat = [
                    float(np.mean(pitch)), 
                    float(np.std(pitch)), 
                    float(np.var(pitch)),
                    float(np.mean(np.diff(pitch)))
                ]
            else:
                pitch_feat = [0.0, 0.0, 0.0, 0.0]
        except:
            pitch_feat = [0.0, 0.0, 0.0, 0.0]

        return mfcc_feat + zcr_feat + centroid_feat + rms_feat + pitch_feat

    except Exception as e:
        print(f"Feature extraction error: {e}", file=sys.stderr)
        raise