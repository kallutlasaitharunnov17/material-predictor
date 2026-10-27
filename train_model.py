import os
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

MATERIALS = ["iron", "steel", "aluminum", "copper", "plastic"]
FS = 48000
N_POINTS = 512

# ---------------- PHYSICS GENERATORS ----------------
PROPERTIES = {
    "iron":     {"E": 200e9, "yield": 250e6, "ult": 400e6, "fract_stn": 0.25, "emr_amp": 1.20},
    "steel":    {"E": 200e9, "yield": 300e6, "ult": 450e6, "fract_stn": 0.20, "emr_amp": 1.10},
    "aluminum": {"E":  70e9, "yield":  55e6, "ult": 110e6, "fract_stn": 0.12, "emr_amp": 0.70},
    "copper":   {"E": 110e9, "yield":  70e6, "ult": 220e6, "fract_stn": 0.30, "emr_amp": 1.00},
    "plastic":  {"E":   3e9, "yield":  20e6, "ult":  30e6, "fract_stn": 0.50, "emr_amp": 0.30},
}

def stress_strain_curve(mat_name, n=N_POINTS, noise=0.01):
    m = PROPERTIES[mat_name]
    E, sy, su, ef = m["E"], m["yield"], m["ult"], m["fract_stn"]
    e_yield = sy / E
    e_ult   = ef * 0.6
    e_frac  = ef
    strain = np.linspace(0, e_frac, n)
    stress = np.zeros_like(strain)
    for i, e in enumerate(strain):
        if e <= e_yield:
            stress[i] = E * e
        elif e <= e_ult:
            t = (e - e_yield) / (e_ult - e_yield)
            stress[i] = sy + (su - sy) * (1 - (1 - t)**2)
        else:
            t = (e - e_ult) / (e_frac - e_ult)
            stress[i] = su * (1 - 0.15 * t)
    stress += noise * su * np.random.randn(n)
    return strain, stress

def emr_signal(strain, stress, mat_name, n=N_POINTS):
    m = PROPERTIES[mat_name]
    e_yield = m["yield"] / m["E"]
    e_frac  = m["fract_stn"]
    amp     = m["emr_amp"]
    envelope = np.zeros(n)
    for i, e in enumerate(strain):
        if e < e_yield: envelope[i] = 0.02 * np.random.rand()
        elif e < e_frac:
            t = (e - e_yield) / (e_frac - e_yield)
            envelope[i] = amp * np.sin(np.pi * t)
    t = np.arange(n) / FS
    carrier_freq = 5000 + 2000 * np.random.rand()
    carrier = np.sin(2 * np.pi * carrier_freq * t)
    emr = envelope * carrier
    emr += 0.02 * np.random.randn(n)
    return emr

def compute_features(strain, stress, emr):
    dt = 1.0 / FS
    E_emr    = float(np.sum(emr**2) * dt)
    E_stress = float(np.sum(stress**2) * dt)
    E_total  = E_emr + E_stress
    E_ratio  = E_emr / (E_stress + 1e-12)
    max_stress   = float(np.max(stress))
    max_strain   = float(np.max(strain))
    stress_range = float(np.ptp(stress))
    idx_linear = max(4, int(len(strain) * 0.05))
    coeffs = np.polyfit(strain[:idx_linear], stress[:idx_linear], 1)
    E_modulus = float(coeffs[0])
    yield_stress = float(stress[int(len(stress) * 0.1)])
    fract_stress = float(stress[-1])
    emr_mean = float(np.mean(emr))
    emr_std  = float(np.std(emr))
    emr_peak = float(np.max(np.abs(emr)))
    emr_rms  = float(np.sqrt(np.mean(emr**2)))
    emr_fft  = np.abs(np.fft.rfft(emr))
    emr_fsum = float(np.sum(emr_fft))
    emr_fmax = float(np.max(emr_fft))
    emr_farg = float(np.argmax(emr_fft))
    cross = float(np.correlate(emr, stress / (stress.max() + 1e-12), mode='valid')[0])
    return np.array([[E_emr, E_stress, E_total, E_ratio, max_stress, max_strain, stress_range, E_modulus, yield_stress, fract_stress, emr_mean, emr_std, emr_peak, emr_rms, emr_fsum, emr_fmax, emr_farg, cross]])

# ---------------- TRAINING ----------------
def main():
    print("Generating data and training model...")
    X_all, y_all = [], []
    for idx, mat in enumerate(MATERIALS):
        for _ in range(200):
            s, st = stress_strain_curve(mat)
            e = emr_signal(s, st, mat)
            X_all.append(compute_features(s, st, e)[0])
            y_all.append(idx)

    X = np.array(X_all)
    y = np.array(y_all)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf.fit(X_train_s, y_train)

    acc = accuracy_score(y_test, clf.predict(X_test_s))
    print(f"Model trained with {acc*100:.2f}% accuracy!")

    # Save to models folder
    os.makedirs("models", exist_ok=True)
    joblib.dump(clf, "models/model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    with open("models/classes.txt", "w") as f:
        f.write("\n".join(MATERIALS))
    print("Model and scaler saved to 'models/' folder.")

if __name__ == "__main__":
    main()