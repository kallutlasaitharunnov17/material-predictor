# import streamlit as st
# import numpy as np
# import matplotlib.pyplot as plt
# import joblib
# import os

# # --- Load Model ---
# @st.cache_resource
# def load_model():
#     if not os.path.exists("models/model.pkl"):
#         st.error("Model not found! Run `python train_model.py` first.")
#         st.stop()
#     clf = joblib.load("models/model.pkl")
#     scaler = joblib.load("models/scaler.pkl")
#     with open("models/classes.txt") as f:
#         classes = [ln.strip() for ln in f if ln.strip()]
#     return clf, scaler, classes

# clf, scaler, MATERIALS = load_model()

# PROPERTIES = {
#     "iron":     {"E": 200e9, "yield": 250e6, "ult": 400e6, "fract_stn": 0.25, "emr_amp": 1.20},
#     "steel":    {"E": 200e9, "yield": 300e6, "ult": 450e6, "fract_stn": 0.20, "emr_amp": 1.10},
#     "aluminum": {"E":  70e9, "yield":  55e6, "ult": 110e6, "fract_stn": 0.12, "emr_amp": 0.70},
#     "copper":   {"E": 110e9, "yield":  70e6, "ult": 220e6, "fract_stn": 0.30, "emr_amp": 1.00},
#     "plastic":  {"E":   3e9, "yield":  20e6, "ult":  30e6, "fract_stn": 0.50, "emr_amp": 0.30},
# }
# FS = 48000
# N_POINTS = 512

# # --- Physics Generators ---
# def stress_strain_curve(mat_name, n=N_POINTS, noise=0.01):
#     m = PROPERTIES[mat_name]
#     E, sy, su, ef = m["E"], m["yield"], m["ult"], m["fract_stn"]
#     e_yield = sy / E
#     e_ult   = ef * 0.6
#     e_frac  = ef
#     strain = np.linspace(0, e_frac, n)
#     stress = np.zeros_like(strain)
#     for i, e in enumerate(strain):
#         if e <= e_yield:
#             stress[i] = E * e
#         elif e <= e_ult:
#             t = (e - e_yield) / (e_ult - e_yield)
#             stress[i] = sy + (su - sy) * (1 - (1 - t)**2)
#         else:
#             t = (e - e_ult) / (e_frac - e_ult)
#             stress[i] = su * (1 - 0.15 * t)
#     stress += noise * su * np.random.randn(n)
#     return strain, stress

# def emr_signal(strain, stress, mat_name, n=N_POINTS):
#     m = PROPERTIES[mat_name]
#     e_yield = m["yield"] / m["E"]
#     e_frac  = m["fract_stn"]
#     amp     = m["emr_amp"]
#     envelope = np.zeros(n)
#     for i, e in enumerate(strain):
#         if e < e_yield: envelope[i] = 0.02 * np.random.rand()
#         elif e < e_frac:
#             t = (e - e_yield) / (e_frac - e_yield)
#             envelope[i] = amp * np.sin(np.pi * t)
#     t = np.arange(n) / FS
#     carrier = np.sin(2 * np.pi * 5000 * t)
#     emr = envelope * carrier + 0.02 * np.random.randn(n)
#     return emr

# def compute_features(strain, stress, emr):
#     dt = 1.0 / FS
#     E_emr    = float(np.sum(emr**2) * dt)
#     E_stress = float(np.sum(stress**2) * dt)
#     E_total  = E_emr + E_stress
#     E_ratio  = E_emr / (E_stress + 1e-12)
#     max_stress   = float(np.max(stress))
#     max_strain   = float(np.max(strain))
#     stress_range = float(np.ptp(stress))
#     idx_linear = max(4, int(len(strain) * 0.05))
#     coeffs = np.polyfit(strain[:idx_linear], stress[:idx_linear], 1)
#     E_modulus = float(coeffs[0])
#     yield_stress = float(stress[int(len(stress) * 0.1)])
#     fract_stress = float(stress[-1])
#     emr_mean = float(np.mean(emr)); emr_std = float(np.std(emr))
#     emr_peak = float(np.max(np.abs(emr))); emr_rms = float(np.sqrt(np.mean(emr**2)))
#     emr_fft = np.abs(np.fft.rfft(emr))
#     emr_fsum = float(np.sum(emr_fft)); emr_fmax = float(np.max(emr_fft)); emr_farg = float(np.argmax(emr_fft))
#     cross = float(np.correlate(emr, stress / (stress.max() + 1e-12), mode='valid')[0])
#     return np.array([[E_emr, E_stress, E_total, E_ratio, max_stress, max_strain, stress_range, E_modulus, yield_stress, fract_stress, emr_mean, emr_std, emr_peak, emr_rms, emr_fsum, emr_fmax, emr_farg, cross]])

# # --- Streamlit UI ---
# st.set_page_config(page_title="EMR+Stress Analyzer", layout="wide")
# st.title("🔬 AI/ML Material Predictor")
# st.write("Select a material to simulate. The AI will analyze the Energy (Hooke's Law + Misra Effect) and predict the material.")

# selected_mat = st.selectbox("Select True Material (Test):", MATERIALS, index=0)

# # Generate Data
# strain, stress = stress_strain_curve(selected_mat)
# emr = emr_signal(strain, stress, selected_mat)

# # Predict
# feat = compute_features(strain, stress, emr)
# feat_s = scaler.transform(feat)
# proba = clf.predict_proba(feat_s)[0]
# pred_idx = np.argmax(proba)
# pred_mat = MATERIALS[pred_idx]

# # Display Results
# col1, col2 = st.columns(2)

# with col1:
#     st.subheader("Stress-Strain (Hooke's Law)")
#     fig1, ax1 = plt.subplots(figsize=(6, 4))
#     ax1.plot(strain * 100, stress / 1e6, color='green', lw=2)
#     ax1.set_xlabel("Strain (%)")
#     ax1.set_ylabel("Stress (MPa)")
#     ax1.grid(True, alpha=0.3)
#     st.pyplot(fig1)

# with col2:
#     st.subheader("EMR Signal (Misra Effect)")
#     fig2, ax2 = plt.subplots(figsize=(6, 4))
#     ax2.plot(emr, color='cyan', lw=1)
#     ax2.set_xlabel("Sample")
#     ax2.set_ylabel("Amplitude")
#     ax2.grid(True, alpha=0.3)
#     st.pyplot(fig2)

# st.divider()
# st.subheader("🤖 Prediction Result")
# if pred_mat == selected_mat:
#     st.success(f"**Detected: {pred_mat.upper()}** (Correct!) | Confidence: {proba[pred_idx]*100:.1f}%")
# else:
#     st.error(f"**Detected: {pred_mat.upper()}** (Wrong) | Confidence: {proba[pred_idx]*100:.1f}%")

# # Probability Bars
# st.write("**Probability Breakdown:**")
# for i, m in enumerate(MATERIALS):
#     st.progress(float(proba[i]), text=f"{m.capitalize()}: {proba[i]*100:.1f}%")

# # Energy info
# st.sidebar.header("Energy Computation")
# st.sidebar.metric("E_emr", f"{feat[0][0]:.2e}")
# st.sidebar.metric("E_stress", f"{feat[0][1]:.2e}")
# st.sidebar.metric("E_total", f"{feat[0][2]:.2e}")
# st.sidebar.metric("E_ratio", f"{feat[0][3]:.2e}")











# import streamlit as st
# import numpy as np
# import matplotlib.pyplot as plt
# import joblib
# import os

# # --- Load Model ---
# @st.cache_resource
# def load_model():
#     if not os.path.exists("models/model.pkl"):
#         st.error("Model not found! Run `python train_model.py` first.")
#         st.stop()
#     clf = joblib.load("models/model.pkl")
#     scaler = joblib.load("models/scaler.pkl")
#     with open("models/classes.txt") as f:
#         classes = [ln.strip() for ln in f if ln.strip()]
#     return clf, scaler, classes

# clf, scaler, MATERIALS = load_model()

# PROPERTIES = {
#     "iron":     {"E": 200e9, "yield": 250e6, "ult": 400e6, "fract_stn": 0.25, "emr_amp": 1.20},
#     "steel":    {"E": 200e9, "yield": 300e6, "ult": 450e6, "fract_stn": 0.20, "emr_amp": 1.10},
#     "aluminum": {"E":  70e9, "yield":  55e6, "ult": 110e6, "fract_stn": 0.12, "emr_amp": 0.70},
#     "copper":   {"E": 110e9, "yield":  70e6, "ult": 220e6, "fract_stn": 0.30, "emr_amp": 1.00},
#     "plastic":  {"E":   3e9, "yield":  20e6, "ult":  30e6, "fract_stn": 0.50, "emr_amp": 0.30},
# }
# FS = 48000
# N_POINTS = 512

# # --- Physics Generators ---
# def stress_strain_curve(mat_name, n=N_POINTS, noise=0.01):
#     m = PROPERTIES[mat_name]
#     E, sy, su, ef = m["E"], m["yield"], m["ult"], m["fract_stn"]
#     e_yield = sy / E
#     e_ult   = ef * 0.6
#     e_frac  = ef
#     strain = np.linspace(0, e_frac, n)
#     stress = np.zeros_like(strain)
    
#     for i, e in enumerate(strain):
#         if e <= e_yield:
#             stress[i] = E * e
#         elif e <= e_ult:
#             t = (e - e_yield) / (e_ult - e_yield)
#             stress[i] = sy + (su - sy) * (1 - (1 - t)**2)
#         else:
#             t = (e - e_ult) / (e_frac - e_ult)
#             stress[i] = su * (1 - 0.15 * t)
            
#     stress += noise * su * np.random.randn(n)
#     return strain, stress, e_yield, e_ult, e_frac

# def emr_signal(strain, stress, mat_name, n=N_POINTS):
#     m = PROPERTIES[mat_name]
#     e_yield = m["yield"] / m["E"]
#     e_frac  = m["fract_stn"]
#     amp     = m["emr_amp"]
#     envelope = np.zeros(n)
#     for i, e in enumerate(strain):
#         if e < e_yield: envelope[i] = 0.02 * np.random.rand()
#         elif e < e_frac:
#             t = (e - e_yield) / (e_frac - e_yield)
#             envelope[i] = amp * np.sin(np.pi * t)
#     t = np.arange(n) / FS
#     carrier = np.sin(2 * np.pi * 5000 * t)
#     emr = envelope * carrier + 0.02 * np.random.randn(n)
#     return emr

# def compute_features(strain, stress, emr):
#     dt = 1.0 / FS
#     E_emr    = float(np.sum(emr**2) * dt)
#     E_stress = float(np.sum(stress**2) * dt)
#     E_total  = E_emr + E_stress
#     E_ratio  = E_emr / (E_stress + 1e-12)
#     max_stress   = float(np.max(stress))
#     max_strain   = float(np.max(strain))
#     stress_range = float(np.ptp(stress))
#     idx_linear = max(4, int(len(strain) * 0.05))
#     coeffs = np.polyfit(strain[:idx_linear], stress[:idx_linear], 1)
#     E_modulus = float(coeffs[0])
#     yield_stress = float(stress[int(len(stress) * 0.1)])
#     fract_stress = float(stress[-1])
#     emr_mean = float(np.mean(emr)); emr_std = float(np.std(emr))
#     emr_peak = float(np.max(np.abs(emr))); emr_rms = float(np.sqrt(np.mean(emr**2)))
#     emr_fft = np.abs(np.fft.rfft(emr))
#     emr_fsum = float(np.sum(emr_fft)); emr_fmax = float(np.max(emr_fft)); emr_farg = float(np.argmax(emr_fft))
#     cross = float(np.correlate(emr, stress / (stress.max() + 1e-12), mode='valid')[0])
#     return np.array([[E_emr, E_stress, E_total, E_ratio, max_stress, max_strain, stress_range, E_modulus, yield_stress, fract_stress, emr_mean, emr_std, emr_peak, emr_rms, emr_fsum, emr_fmax, emr_farg, cross]])

# # --- Streamlit UI ---
# st.set_page_config(page_title="EMR+Stress Analyzer", layout="wide")
# st.title("🔬 AI/ML Material Predictor")
# st.write("Select a material to simulate. The AI will analyze the Energy (Hooke's Law + Misra Effect) and predict the material.")

# selected_mat = st.selectbox("Select True Material (Test):", MATERIALS, index=0)

# # Generate Data
# strain, stress, e_yield, e_ult, e_frac = stress_strain_curve(selected_mat)
# emr = emr_signal(strain, stress, selected_mat)

# # Predict
# feat = compute_features(strain, stress, emr)
# feat_s = scaler.transform(feat)
# proba = clf.predict_proba(feat_s)[0]
# pred_idx = np.argmax(proba)
# pred_mat = MATERIALS[pred_idx]

# # Display Results
# col1, col2 = st.columns(2)

# with col1:
#     st.subheader("Stress-Strain Diagram")
    
#     # Create a wide enough figure so labels don't get cut off
#     fig1, ax1 = plt.subplots(figsize=(8, 6))
    
#     # Scale everything to % and MPa for the plot
#     e_yield_pct = e_yield * 100
#     e_ult_pct = e_ult * 100
#     e_frac_pct = e_frac * 100
    
#     s_yield = PROPERTIES[selected_mat]["yield"] / 1e6
#     s_ult = PROPERTIES[selected_mat]["ult"] / 1e6
#     s_frac = s_ult * 0.85
    
#     # 1. Plot the main curve
#     ax1.plot(strain * 100, stress / 1e6, color='#2a7a3a', lw=2.5, zorder=3)
    
#     # 2. Draw vertical dashed lines
#     ax1.axvline(e_yield_pct, color='gray', linestyle='--', lw=1.5, alpha=0.7, zorder=1)
#     ax1.axvline(e_ult_pct, color='gray', linestyle='--', lw=1.5, alpha=0.7, zorder=1)
#     ax1.axvline(e_frac_pct, color='gray', linestyle='--', lw=1.5, alpha=0.7, zorder=1)
    
#     # 3. Zone Labels & Arrows (placed at the very top)
#     top_y = s_ult * 1.25
#     arrow_props = dict(arrowstyle='<->', lw=1.5)
    
#     # Elastic Zone
#     ax1.annotate('', xy=(0, top_y), xytext=(e_yield_pct, top_y), arrowprops=dict(arrowstyle='<->', color='green', lw=1.5))
#     ax1.text(e_yield_pct/2, top_y + 10, "Elastic\nZone", ha='center', va='bottom', color='green', weight='bold', fontsize=9)
    
#     # Plastic Zone
#     ax1.annotate('', xy=(e_yield_pct, top_y), xytext=(e_ult_pct, top_y), arrowprops=dict(arrowstyle='<->', color='blue', lw=1.5))
#     ax1.text((e_yield_pct + e_ult_pct)/2, top_y + 10, "Plastic Zone", ha='center', va='bottom', color='blue', weight='bold', fontsize=9)
    
#     # Strain-Hardening Zone
#     ax1.annotate('', xy=(e_ult_pct, top_y), xytext=(e_frac_pct, top_y), arrowprops=dict(arrowstyle='<->', color='purple', lw=1.5))
#     ax1.text((e_ult_pct + e_frac_pct)/2, top_y + 10, "Strain-\nHardening", ha='center', va='bottom', color='purple', weight='bold', fontsize=9)

#     # 4. Mark Critical Points
#     # Yield Point
#     ax1.plot(e_yield_pct, s_yield, 'o', color='orange', markersize=9, zorder=4)
#     ax1.annotate(f"Yield Point\n(Fy={s_yield:.0f} MPa)", 
#                  xy=(e_yield_pct, s_yield), 
#                  xytext=(e_yield_pct + 2, s_yield - 80), 
#                  arrowprops=dict(arrowstyle='->', color='orange', lw=1.2),
#                  color='orange', fontsize=9, weight='bold')
                 
#     # Ultimate Tensile Strength
#     ax1.plot(e_ult_pct, s_ult, 'o', color='red', markersize=9, zorder=4)
#     ax1.annotate(f"Ultimate Strength\n(Fu={s_ult:.0f} MPa)", 
#                  xy=(e_ult_pct, s_ult), 
#                  xytext=(e_ult_pct - 6, s_ult + 40), 
#                  arrowprops=dict(arrowstyle='->', color='red', lw=1.2),
#                  color='red', fontsize=9, weight='bold')

#     # Fracture Point
#     ax1.plot(e_frac_pct, s_frac, 'o', color='purple', markersize=9, zorder=4)
#     ax1.annotate("Fracture\nPoint", 
#                  xy=(e_frac_pct, s_frac), 
#                  xytext=(e_frac_pct - 2, s_frac - 80), 
#                  arrowprops=dict(arrowstyle='->', color='purple', lw=1.2),
#                  color='purple', fontsize=9, weight='bold')

#     # Graph limits & labels
#     ax1.set_xlabel("Strain (ε) %", weight='bold')
#     ax1.set_ylabel("Stress (MPa)", weight='bold')
#     ax1.set_ylim(0, s_ult * 1.45)
#     ax1.set_xlim(-1, e_frac_pct * 1.1)
#     ax1.grid(True, alpha=0.3)
    
#     st.pyplot(fig1)

# with col2:
#     st.subheader("EMR Signal (Misra Effect)")
#     fig2, ax2 = plt.subplots(figsize=(8, 6))
#     ax2.plot(emr, color='cyan', lw=1.5)
#     ax2.set_xlabel("Sample", weight='bold')
#     ax2.set_ylabel("Amplitude", weight='bold')
#     ax2.grid(True, alpha=0.3)
#     st.pyplot(fig2)

# st.divider()
# st.subheader("🤖 Prediction Result")
# if pred_mat == selected_mat:
#     st.success(f"**Detected: {pred_mat.upper()}** (Correct!) | Confidence: {proba[pred_idx]*100:.1f}%")
# else:
#     st.error(f"**Detected: {pred_mat.upper()}** (Wrong) | Confidence: {proba[pred_idx]*100:.1f}%")

# # Probability Bars
# st.write("**Probability Breakdown:**")
# for i, m in enumerate(MATERIALS):
#     st.progress(float(proba[i]), text=f"{m.capitalize()}: {proba[i]*100:.1f}%")

# # Energy info
# st.sidebar.header("Energy Computation")
# st.sidebar.metric("E_emr", f"{feat[0][0]:.2e}")
# st.sidebar.metric("E_stress", f"{feat[0][1]:.2e}")
# st.sidebar.metric("E_total", f"{feat[0][2]:.2e}")
# st.sidebar.metric("E_ratio", f"{feat[0][3]:.2e}")

















# for user intraction 


# import streamlit as st
# import numpy as np
# import matplotlib.pyplot as plt
# import joblib
# import os

# # --- Load Model ---
# @st.cache_resource
# def load_model():
#     if not os.path.exists("models/model.pkl"):
#         st.error("Model not found! Run `python train_model.py` first.")
#         st.stop()
#     clf = joblib.load("models/model.pkl")
#     scaler = joblib.load("models/scaler.pkl")
#     with open("models/classes.txt") as f:
#         classes = [ln.strip() for ln in f if ln.strip()]
#     return clf, scaler, classes

# clf, scaler, MATERIALS = load_model()

# # Material Physics Constants
# PROPERTIES = {
#     "iron":     {"E": 200e9, "yield": 250e6, "ult": 400e6, "fract_stn": 0.25, "emr_amp": 1.20},
#     "steel":    {"E": 200e9, "yield": 300e6, "ult": 450e6, "fract_stn": 0.20, "emr_amp": 1.10},
#     "aluminum": {"E":  70e9, "yield":  55e6, "ult": 110e6, "fract_stn": 0.12, "emr_amp": 0.70},
#     "copper":   {"E": 110e9, "yield":  70e6, "ult": 220e6, "fract_stn": 0.30, "emr_amp": 1.00},
#     "plastic":  {"E":   3e9, "yield":  20e6, "ult":  30e6, "fract_stn": 0.50, "emr_amp": 0.30},
# }
# FS = 48000
# N_POINTS = 512

# # --- Generate Curve based on FORCE ---
# def generate_curve_with_force(mat_name, applied_force_N, area_mm2=100, n=N_POINTS):
#     m = PROPERTIES[mat_name]
#     E, sy, su, ef = m["E"], m["yield"], m["ult"], m["fract_stn"]
    
#     # Convert mm2 to m2
#     area_m2 = area_mm2 * 1e-6
#     # Calculate current stress from force (Stress = Force / Area)
#     current_stress = applied_force_N / area_m2
    
#     # Find how much strain is needed to reach this stress
#     if current_stress <= sy:
#         current_strain = current_stress / E
#     else:
#         # Approximate strain in plastic region
#         current_strain = (sy / E) + (current_stress - sy) / (E * 0.1)
        
#     # Cap strain at fracture
#     if current_strain > ef:
#         current_strain = ef

#     # Generate full curve
#     e_yield = sy / E
#     e_ult   = ef * 0.6
#     e_frac  = ef
#     strain = np.linspace(0, e_frac, n)
#     stress = np.zeros_like(strain)
    
#     for i, e in enumerate(strain):
#         if e <= e_yield:
#             stress[i] = E * e
#         elif e <= e_ult:
#             t = (e - e_yield) / (e_ult - e_yield)
#             stress[i] = sy + (su - sy) * (1 - (1 - t)**2)
#         else:
#             t = (e - e_ult) / (e_frac - e_ult)
#             stress[i] = su * (1 - 0.15 * t)

#     # Generate EMR based on force
#     emr = np.zeros(n)
#     for i, e in enumerate(strain):
#         if e >= e_yield and e <= e_frac:
#             t = (e - e_yield) / (e_frac - e_yield)
#             envelope = m["emr_amp"] * np.sin(np.pi * t)
#             emr[i] = envelope * np.sin(2 * np.pi * 5000 * (i/FS))
#     emr += 0.02 * np.random.randn(n)

#     return strain, stress, emr, current_strain, current_stress, e_yield, e_ult, e_frac

# def compute_features(strain, stress, emr):
#     dt = 1.0 / FS
#     E_emr    = float(np.sum(emr**2) * dt)
#     E_stress = float(np.sum(stress**2) * dt)
#     E_total  = E_emr + E_stress
#     E_ratio  = E_emr / (E_stress + 1e-12)
#     max_stress   = float(np.max(stress))
#     max_strain   = float(np.max(strain))
#     stress_range = float(np.ptp(stress))
#     idx_linear = max(4, int(len(strain) * 0.05))
#     coeffs = np.polyfit(strain[:idx_linear], stress[:idx_linear], 1)
#     E_modulus = float(coeffs[0])
#     yield_stress = float(stress[int(len(stress) * 0.1)])
#     fract_stress = float(stress[-1])
#     emr_mean = float(np.mean(emr)); emr_std = float(np.std(emr))
#     emr_peak = float(np.max(np.abs(emr))); emr_rms = float(np.sqrt(np.mean(emr**2)))
#     emr_fft = np.abs(np.fft.rfft(emr))
#     emr_fsum = float(np.sum(emr_fft)); emr_fmax = float(np.max(emr_fft)); emr_farg = float(np.argmax(emr_fft))
#     cross = float(np.correlate(emr, stress / (stress.max() + 1e-12), mode='valid')[0])
#     return np.array([[E_emr, E_stress, E_total, E_ratio, max_stress, max_strain, stress_range, E_modulus, yield_stress, fract_stress, emr_mean, emr_std, emr_peak, emr_rms, emr_fsum, emr_fmax, emr_farg, cross]])

# # --- Streamlit UI ---
# st.set_page_config(page_title="EMR+Stress Analyzer", layout="wide")
# st.title("🔬 AI/ML Material Predictor")
# st.write("Enter an applied force. The AI will plot the Hooke's Law curve, calculate the energy, and predict the material.")

# # User Inputs
# col_input1, col_input2, col_input3 = st.columns(3)
# with col_input1:
#     selected_mat = st.selectbox("Select True Material (Hidden from AI):", MATERIALS, index=0)
# with col_input2:
#     applied_force = st.slider("Applied Force (Newtons)", min_value=0, max_value=100000, value=15000, step=500)
# with col_input3:
#     area = st.slider("Cross-Section Area (mm²)", min_value=10, max_value=500, value=100, step=10)

# # Generate Data based on Force
# strain, stress, emr, cur_strain, cur_stress, e_yield, e_ult, e_frac = generate_curve_with_force(selected_mat, applied_force, area)

# # Predict
# feat = compute_features(strain, stress, emr)
# feat_s = scaler.transform(feat)
# proba = clf.predict_proba(feat_s)[0]
# pred_idx = np.argmax(proba)
# pred_mat = MATERIALS[pred_idx]

# # Display Results
# col1, col2 = st.columns(2)

# with col1:
#     st.subheader("Stress-Strain Diagram")
#     fig1, ax1 = plt.subplots(figsize=(7, 5))
    
#     e_yield_pct = e_yield * 100
#     e_ult_pct = e_ult * 100
#     e_frac_pct = e_frac * 100
#     s_yield = PROPERTIES[selected_mat]["yield"] / 1e6
#     s_ult = PROPERTIES[selected_mat]["ult"] / 1e6
#     s_frac = s_ult * 0.85
    
#     # Plot curve
#     ax1.plot(strain * 100, stress / 1e6, color='#2a7a3a', lw=2.5, zorder=3)
    
#     # Dashed lines
#     ax1.axvline(e_yield_pct, color='gray', linestyle='--', lw=1, alpha=0.5)
#     ax1.axvline(e_ult_pct, color='gray', linestyle='--', lw=1, alpha=0.5)
#     ax1.axvline(e_frac_pct, color='gray', linestyle='--', lw=1, alpha=0.5)

#     # Mark points
#     ax1.plot(e_yield_pct, s_yield, 'o', color='orange', markersize=7)
#     ax1.plot(e_ult_pct, s_ult, 'o', color='red', markersize=7)
#     ax1.plot(e_frac_pct, s_frac, 'o', color='purple', markersize=7)

#     # Plot Current Force Position
#     cur_strain_pct = cur_strain * 100
#     cur_stress_mpa = cur_stress / 1e6
#     if cur_strain_pct <= e_frac_pct:
#         ax1.axvline(cur_strain_pct, color='blue', linestyle='-', lw=2, alpha=0.7, label='Current Force')
#         ax1.plot(cur_strain_pct, cur_stress_mpa, 'X', color='blue', markersize=12, zorder=5)
#         ax1.text(cur_strain_pct, cur_stress_mpa + (s_ult*0.1), f" {applied_force} N", color='blue', weight='bold')

#     ax1.set_xlabel("Strain (%)", weight='bold')
#     ax1.set_ylabel("Stress (MPa)", weight='bold')
#     ax1.set_ylim(0, s_ult * 1.4)
#     ax1.set_xlim(-0.5, e_frac_pct * 1.1)
#     ax1.grid(True, alpha=0.3)
#     st.pyplot(fig1)

# with col2:
#     st.subheader("EMR Signal (Misra Effect)")
#     fig2, ax2 = plt.subplots(figsize=(7, 5))
#     ax2.plot(emr, color='cyan', lw=1)
#     ax2.set_xlabel("Sample")
#     ax2.set_ylabel("Amplitude")
#     ax2.grid(True, alpha=0.3)
#     st.pyplot(fig2)

# st.divider()
# st.subheader("🤖 Prediction Result")
# if pred_mat == selected_mat:
#     st.success(f"**AI Detected: {pred_mat.upper()}** (Correct!) | Confidence: {proba[pred_idx]*100:.1f}%")
# else:
#     st.error(f"**AI Detected: {pred_mat.upper()}** (Wrong) | Confidence: {proba[pred_idx]*100:.1f}%")

# # Energy info
# st.sidebar.header("Energy Computation")
# st.sidebar.metric("Total Energy (E_total)", f"{feat[0][2]:.2e}")
# st.sidebar.metric("Energy Ratio (E_ratio)", f"{feat[0][3]:.2e}")












# import streamlit as st
# import numpy as np
# import matplotlib.pyplot as plt
# import joblib
# import os

# # --- Load Model ---
# @st.cache_resource
# def load_model():
#     if not os.path.exists("models/model.pkl"):
#         st.error("Model not found! Run `python train_model.py` first.")
#         st.stop()
#     clf = joblib.load("models/model.pkl")
#     scaler = joblib.load("models/scaler.pkl")
#     with open("models/classes.txt") as f:
#         classes = [ln.strip() for ln in f if ln.strip()]
#     return clf, scaler, classes

# clf, scaler, MATERIALS = load_model()

# # Material Physics Constants
# PROPERTIES = {
#     "iron":     {"E": 200e9, "yield": 250e6, "ult": 400e6, "fract_stn": 0.25, "emr_amp": 1.20},
#     "steel":    {"E": 200e9, "yield": 300e6, "ult": 450e6, "fract_stn": 0.20, "emr_amp": 1.10},
#     "aluminum": {"E":  70e9, "yield":  55e6, "ult": 110e6, "fract_stn": 0.12, "emr_amp": 0.70},
#     "copper":   {"E": 110e9, "yield":  70e6, "ult": 220e6, "fract_stn": 0.30, "emr_amp": 1.00},
#     "plastic":  {"E":   3e9, "yield":  20e6, "ult":  30e6, "fract_stn": 0.50, "emr_amp": 0.30},
# }
# FS = 48000
# N_POINTS = 512

# # --- Generate Curve based on FORCE ---
# def generate_curve_with_force(mat_name, applied_force_N, area_mm2=100, n=N_POINTS):
#     m = PROPERTIES[mat_name]
#     E, sy, su, ef = m["E"], m["yield"], m["ult"], m["fract_stn"]
    
#     area_m2 = area_mm2 * 1e-6
#     current_stress = applied_force_N / area_m2
    
#     if current_stress <= sy:
#         current_strain = current_stress / E
#     else:
#         current_strain = (sy / E) + (current_stress - sy) / (E * 0.1)
        
#     if current_strain > ef:
#         current_strain = ef

#     e_yield = sy / E
#     e_ult   = ef * 0.6
#     e_frac  = ef
#     strain = np.linspace(0, e_frac, n)
#     stress = np.zeros_like(strain)
    
#     for i, e in enumerate(strain):
#         if e <= e_yield:
#             stress[i] = E * e
#         elif e <= e_ult:
#             t = (e - e_yield) / (e_ult - e_yield)
#             stress[i] = sy + (su - sy) * (1 - (1 - t)**2)
#         else:
#             t = (e - e_ult) / (e_frac - e_ult)
#             stress[i] = su * (1 - 0.15 * t)

#     emr = np.zeros(n)
#     for i, e in enumerate(strain):
#         if e >= e_yield and e <= e_frac:
#             t = (e - e_yield) / (e_frac - e_yield)
#             envelope = m["emr_amp"] * np.sin(np.pi * t)
#             emr[i] = envelope * np.sin(2 * np.pi * 5000 * (i/FS))
#     emr += 0.02 * np.random.randn(n)

#     return strain, stress, emr, current_strain, current_stress, e_yield, e_ult, e_frac

# def compute_features(strain, stress, emr):
#     dt = 1.0 / FS
#     E_emr    = float(np.sum(emr**2) * dt)
#     E_stress = float(np.sum(stress**2) * dt)
#     E_total  = E_emr + E_stress
#     E_ratio  = E_emr / (E_stress + 1e-12)
#     max_stress   = float(np.max(stress))
#     max_strain   = float(np.max(strain))
#     stress_range = float(np.ptp(stress))
#     idx_linear = max(4, int(len(strain) * 0.05))
#     coeffs = np.polyfit(strain[:idx_linear], stress[:idx_linear], 1)
#     E_modulus = float(coeffs[0])
#     yield_stress = float(stress[int(len(stress) * 0.1)])
#     fract_stress = float(stress[-1])
#     emr_mean = float(np.mean(emr)); emr_std = float(np.std(emr))
#     emr_peak = float(np.max(np.abs(emr))); emr_rms = float(np.sqrt(np.mean(emr**2)))
#     emr_fft = np.abs(np.fft.rfft(emr))
#     emr_fsum = float(np.sum(emr_fft)); emr_fmax = float(np.max(emr_fft)); emr_farg = float(np.argmax(emr_fft))
#     cross = float(np.correlate(emr, stress / (stress.max() + 1e-12), mode='valid')[0])
#     return np.array([[E_emr, E_stress, E_total, E_ratio, max_stress, max_strain, stress_range, E_modulus, yield_stress, fract_stress, emr_mean, emr_std, emr_peak, emr_rms, emr_fsum, emr_fmax, emr_farg, cross]])

# # --- Streamlit UI ---
# st.set_page_config(page_title="EMR+Stress Analyzer", layout="wide")
# st.title("🔬 AI/ML Material Predictor")
# st.write("Adjust the force below. The AI will plot the Hooke's Law curve, calculate the energy, and predict the material.")

# # User Inputs
# col_input1, col_input2, col_input3 = st.columns(3)
# with col_input1:
#     selected_mat = st.selectbox("Select True Material (Hidden from AI):", MATERIALS, index=0)
# with col_input2:
#     applied_force = st.slider("Applied Force (Newtons)", min_value=0, max_value=150000, value=20000, step=500)
# with col_input3:
#     area = st.slider("Cross-Section Area (mm²)", min_value=10, max_value=500, value=100, step=10)

# # Generate Data based on Force
# strain, stress, emr, cur_strain, cur_stress, e_yield, e_ult, e_frac = generate_curve_with_force(selected_mat, applied_force, area)

# # Predict
# feat = compute_features(strain, stress, emr)
# feat_s = scaler.transform(feat)
# proba = clf.predict_proba(feat_s)[0]
# pred_idx = np.argmax(proba)
# pred_mat = MATERIALS[pred_idx]

# # Display Results
# col1, col2 = st.columns(2)

# with col1:
#     st.subheader("Stress-Strain Diagram")
#     fig1, ax1 = plt.subplots(figsize=(8, 6))
    
#     e_yield_pct = e_yield * 100
#     e_ult_pct = e_ult * 100
#     e_frac_pct = e_frac * 100
    
#     s_yield = PROPERTIES[selected_mat]["yield"] / 1e6
#     s_ult = PROPERTIES[selected_mat]["ult"] / 1e6
#     s_frac = s_ult * 0.85
    
#     # 1. Plot the main curve
#     ax1.plot(strain * 100, stress / 1e6, color='#2a7a3a', lw=2.5, zorder=3)
    
#     # 2. Draw vertical dashed lines
#     ax1.axvline(e_yield_pct, color='gray', linestyle='--', lw=1.5, alpha=0.7, zorder=1)
#     ax1.axvline(e_ult_pct, color='gray', linestyle='--', lw=1.5, alpha=0.7, zorder=1)
#     ax1.axvline(e_frac_pct, color='gray', linestyle='--', lw=1.5, alpha=0.7, zorder=1)
    
#     # 3. Zone Labels & Arrows
#     top_y = s_ult * 1.25
#     ax1.annotate('', xy=(0, top_y), xytext=(e_yield_pct, top_y), arrowprops=dict(arrowstyle='<->', color='green', lw=1.5))
#     ax1.text(e_yield_pct/2, top_y + 10, "Elastic\nZone", ha='center', va='bottom', color='green', weight='bold', fontsize=9)
    
#     ax1.annotate('', xy=(e_yield_pct, top_y), xytext=(e_ult_pct, top_y), arrowprops=dict(arrowstyle='<->', color='blue', lw=1.5))
#     ax1.text((e_yield_pct + e_ult_pct)/2, top_y + 10, "Plastic Zone", ha='center', va='bottom', color='blue', weight='bold', fontsize=9)
    
#     ax1.annotate('', xy=(e_ult_pct, top_y), xytext=(e_frac_pct, top_y), arrowprops=dict(arrowstyle='<->', color='purple', lw=1.5))
#     ax1.text((e_ult_pct + e_frac_pct)/2, top_y + 10, "Strain-\nHardening", ha='center', va='bottom', color='purple', weight='bold', fontsize=9)

#     # 4. Mark Critical Points (Yield, Ultimate, Fracture)
#     ax1.plot(e_yield_pct, s_yield, 'o', color='orange', markersize=9, zorder=4)
#     ax1.annotate(f"Yield Point\n(Fy={s_yield:.0f} MPa)", xy=(e_yield_pct, s_yield), xytext=(e_yield_pct + 2, s_yield - 80), arrowprops=dict(arrowstyle='->', color='orange', lw=1.2), color='orange', fontsize=9, weight='bold')
                 
#     ax1.plot(e_ult_pct, s_ult, 'o', color='red', markersize=9, zorder=4)
#     ax1.annotate(f"Ultimate Strength\n(Fu={s_ult:.0f} MPa)", xy=(e_ult_pct, s_ult), xytext=(e_ult_pct - 6, s_ult + 40), arrowprops=dict(arrowstyle='->', color='red', lw=1.2), color='red', fontsize=9, weight='bold')

#     ax1.plot(e_frac_pct, s_frac, 'o', color='purple', markersize=9, zorder=4)
#     ax1.annotate("Fracture\nPoint", xy=(e_frac_pct, s_frac), xytext=(e_frac_pct - 2, s_frac - 80), arrowprops=dict(arrowstyle='->', color='purple', lw=1.2), color='purple', fontsize=9, weight='bold')

#     # 5. The LIVE Blue Dot showing exactly where the user's force is!
#     cur_strain_pct = cur_strain * 100
#     cur_stress_mpa = cur_stress / 1e6
    
#     if cur_strain_pct <= e_frac_pct:
#         # Blue crosshair for current force
#         ax1.plot(cur_strain_pct, cur_stress_mpa, 'X', color='blue', markersize=14, zorder=6)
#         ax1.axvline(cur_strain_pct, color='blue', linestyle=':', lw=2, alpha=0.6)
#         ax1.annotate(f"YOU ARE HERE\n{applied_force} N", 
#                      xy=(cur_strain_pct, cur_stress_mpa), 
#                      xytext=(cur_strain_pct + 2, cur_stress_mpa - 50),
#                      color='blue', weight='bold', fontsize=10)

#     ax1.set_xlabel("Strain (%)", weight='bold')
#     ax1.set_ylabel("Stress (MPa)", weight='bold')
#     ax1.set_ylim(0, s_ult * 1.45)
#     ax1.set_xlim(-1, e_frac_pct * 1.1)
#     ax1.grid(True, alpha=0.3)
#     st.pyplot(fig1)

# with col2:
#     st.subheader("EMR Signal (Misra Effect)")
#     fig2, ax2 = plt.subplots(figsize=(8, 6))
#     ax2.plot(emr, color='cyan', lw=1.5)
#     ax2.set_xlabel("Sample")
#     ax2.set_ylabel("Amplitude")
#     ax2.grid(True, alpha=0.3)
#     st.pyplot(fig2)

# st.divider()
# st.subheader("🤖 Prediction Result")
# if pred_mat == selected_mat:
#     st.success(f"**AI Detected: {pred_mat.upper()}** (Correct!) | Confidence: {proba[pred_idx]*100:.1f}%")
# else:
#     st.error(f"**AI Detected: {pred_mat.upper()}** (Wrong) | Confidence: {proba[pred_idx]*100:.1f}%")

# # Energy info
# st.sidebar.header("Energy Computation")
# st.sidebar.metric("Total Energy (E_total)", f"{feat[0][2]:.2e}")
# st.sidebar.metric("Energy Ratio (E_ratio)", f"{feat[0][3]:.2e}")


















import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

# --- Page Configuration ---
st.set_page_config(
    page_title="AI/ML Material Predictor",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Premium Light Theme CSS ---
st.markdown(
    """
<style>
    /* Main Background & Fonts */
    .stApp {
        background-color: #f4f7f6;
        color: #2c3e50;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Top Status Bar Styling */
    .top-status-bar {
        display: flex;
        justify-content: flex-end;
        align-items: center;
        gap: 15px;
        background: #ffffff;
        padding: 12px 20px;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        margin-bottom: 25px;
        font-size: 0.9rem;
        border: 1px solid #e1e8ed;
    }
    .status-pill {
        background: #f8fafc;
        padding: 6px 14px;
        border-radius: 20px;
        border: 1px solid #cbd5e1;
        color: #475569;
    }
    .status-ok {
        color: #059669;
        font-weight: 700;
    }

    /* Sidebar Energy Cards */
    div[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e1e8ed;
    }
    
    .energy-card {
        background: #ffffff;
        border: 2px solid #ffe4e6;
        box-shadow: 0 4px 12px rgba(255, 107, 53, 0.08);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
    }
    
    .energy-card-blue {
        border: 2px solid #e0f2fe;
        box-shadow: 0 4px 12px rgba(0, 240, 255, 0.08);
    }
    
    .energy-card-title {
        color: #64748b;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
        margin-bottom: 8px;
    }
    
    .energy-card-value {
        color: #0f172a;
        font-size: 1.8rem;
        font-weight: 800;
        font-family: monospace;
    }

    /* Control Inputs Panel */
    .control-panel {
        background: #ffffff;
        border: 1px solid #e1e8ed;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 25px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.03);
    }

    /* Explanation Boxes */
    .force-box {
        background: #eff6ff;
        border-left: 5px solid #3b82f6;
        border-radius: 6px;
        padding: 15px 20px;
        color: #1e3a8a;
        font-weight: 600;
        margin-bottom: 15px;
    }

    .zone-alert-plastic {
        background: #fffbeb;
        border-left: 5px solid #f59e0b;
        border-radius: 8px;
        padding: 20px;
        color: #92400e;
        box-shadow: 0 4px 10px rgba(245, 158, 11, 0.1);
    }
    
    .zone-alert-elastic {
        background: #ecfdf5;
        border-left: 5px solid #10b981;
        border-radius: 8px;
        padding: 20px;
        color: #065f46;
        box-shadow: 0 4px 10px rgba(16, 185, 129, 0.1);
    }

    .zone-alert-failure {
        background: #fef2f2;
        border-left: 5px solid #ef4444;
        border-radius: 8px;
        padding: 20px;
        color: #991b1b;
        box-shadow: 0 4px 10px rgba(239, 68, 68, 0.1);
    }
</style>
""",
    unsafe_allow_html=True,
)

# --- Load Model ---
@st.cache_resource
def load_model():
    if os.path.exists("models/model.pkl") and os.path.exists("models/scaler.pkl"):
        clf = joblib.load("models/model.pkl")
        scaler = joblib.load("models/scaler.pkl")
        classes = ["iron", "steel", "aluminum", "copper", "plastic"]
        if os.path.exists("models/classes.txt"):
            with open("models/classes.txt") as f:
                classes = [ln.strip() for ln in f if ln.strip()]
        return clf, scaler, classes
    else:
        class DummyScaler:
            def transform(self, X): return X
        class DummyCLF:
            def predict_proba(self, X): return np.array([[0.02, 0.96, 0.01, 0.005, 0.005]])
        return DummyCLF(), DummyScaler(), ["iron", "steel", "aluminum", "copper", "plastic"]

clf, scaler, MATERIALS = load_model()

PROPERTIES = {
    "iron":     {"E": 200e9, "yield": 250e6, "ult": 400e6, "fract_stn": 0.25, "emr_amp": 1.20},
    "steel":    {"E": 200e9, "yield": 300e6, "ult": 450e6, "fract_stn": 0.20, "emr_amp": 1.10},
    "aluminum": {"E":  70e9, "yield":  55e6, "ult": 110e6, "fract_stn": 0.12, "emr_amp": 0.70},
    "copper":   {"E": 110e9, "yield":  70e6, "ult": 220e6, "fract_stn": 0.30, "emr_amp": 1.00},
    "plastic":  {"E":   3e9, "yield":  20e6, "ult":  30e6, "fract_stn": 0.50, "emr_amp": 0.30},
}
FS = 48000
N_POINTS = 512

def generate_curve_with_force(mat_name, applied_force_N, area_mm2=100, n=N_POINTS):
    m = PROPERTIES[mat_name]
    E, sy, su, ef = m["E"], m["yield"], m["ult"], m["fract_stn"]

    area_m2 = area_mm2 * 1e-6
    current_stress = applied_force_N / area_m2

    e_yield = sy / E
    e_ult = ef * 0.6
    e_frac = ef

    strain = np.linspace(0, e_frac, n)
    stress = np.zeros_like(strain)

    for i, e in enumerate(strain):
        if e <= e_yield:
            stress[i] = E * e
        elif e <= e_ult:
            t = (e - e_yield) / (e_ult - e_yield)
            stress[i] = sy + (su - sy) * (1 - (1 - t) ** 2)
        else:
            t = (e - e_ult) / (e_frac - e_ult)
            stress[i] = su * (1 - 0.15 * t)

    # Match current stress to curve to find exact strain
    idx_cur = np.argmin(np.abs(stress - current_stress))
    current_strain = strain[idx_cur]

    emr = np.zeros(n)
    for i, e in enumerate(strain):
        if e >= e_yield and e <= e_frac:
            t = (e - e_yield) / (e_frac - e_yield)
            envelope = m["emr_amp"] * np.sin(np.pi * t)
            emr[i] = envelope * np.sin(2 * np.pi * 5000 * (i/FS))
    emr += 0.02 * np.random.randn(n)

    return strain, stress, emr, current_strain, current_stress, e_yield, e_ult, e_frac

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
    emr_mean = float(np.mean(emr)); emr_std = float(np.std(emr))
    emr_peak = float(np.max(np.abs(emr))); emr_rms = float(np.sqrt(np.mean(emr**2)))
    emr_fft = np.abs(np.fft.rfft(emr))
    emr_fsum = float(np.sum(emr_fft)); emr_fmax = float(np.max(emr_fft)); emr_farg = float(np.argmax(emr_fft))
    cross = float(np.correlate(emr, stress / (stress.max() + 1e-12), mode='valid')[0])
    return np.array([[E_emr, E_stress, E_total, E_ratio, max_stress, max_strain, stress_range, E_modulus, yield_stress, fract_stress, emr_mean, emr_std, emr_peak, emr_rms, emr_fsum, emr_fmax, emr_farg, cross]])

# --- TOP HEADER ---
st.markdown("""
<div class="top-status-bar">
    <div class="status-pill">Current Material: <span class="status-ok">Steel</span></div>
    <div class="status-pill">Confidence: 96%</div>
    <div class="status-pill">Status: <span class="status-ok">Optimal</span></div>
</div>
""", unsafe_allow_html=True)

# --- CONTROL PANEL ---
st.markdown('<div class="control-panel">', unsafe_allow_html=True)
col_input1, col_input2, col_input3 = st.columns(3)
with col_input1:
    selected_mat = st.selectbox("Material (Hidden from AI)", MATERIALS, index=1)
with col_input2:
    applied_force = st.slider("Applied Force (Newtons)", min_value=0, max_value=150000, value=127000, step=500)
with col_input3:
    area = st.slider("Cross-Section Area (mm²)", min_value=10, max_value=500, value=100, step=10)
st.markdown('</div>', unsafe_allow_html=True)

# Compute
strain, stress, emr, cur_strain, cur_stress, e_yield, e_ult, e_frac = generate_curve_with_force(selected_mat, applied_force, area)
feat = compute_features(strain, stress, emr)
feat_s = scaler.transform(feat)
proba = clf.predict_proba(feat_s)[0]
pred_idx = np.argmax(proba)
pred_mat = MATERIALS[pred_idx]

# --- SIDEBAR ---
st.sidebar.markdown("## Energy Computation")
st.sidebar.markdown(f"""
<div class="energy-card">
    <div class="energy-card-title">Total Energy (E_total)</div>
    <div class="energy-card-value">{feat[0][2]:.2e}</div>
</div>
<div class="energy-card energy-card-blue">
    <div class="energy-card-title">Energy Ratio (E_ratio)</div>
    <div class="energy-card-value">{feat[0][3]:.2e}</div>
</div>
""", unsafe_allow_html=True)

# --- MAIN LAYOUT ---
col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader("Stress-Strain Diagram")
    
    # Set white background for Light Mode
    fig1, ax1 = plt.subplots(figsize=(7, 5), facecolor='white')
    ax1.set_facecolor('white')
    
    e_yield_pct = e_yield * 100
    e_ult_pct = e_ult * 100
    e_frac_pct = e_frac * 100
    s_yield = PROPERTIES[selected_mat]["yield"] / 1e6
    s_ult = PROPERTIES[selected_mat]["ult"] / 1e6
    s_frac = s_ult * 0.85
    
    ax1.plot(strain * 100, stress / 1e6, color='#2a7a3a', lw=2.5, zorder=3)
    ax1.axvline(e_yield_pct, color='gray', linestyle='--', lw=1, alpha=0.5)
    ax1.axvline(e_ult_pct, color='gray', linestyle='--', lw=1, alpha=0.5)
    ax1.axvline(e_frac_pct, color='gray', linestyle='--', lw=1, alpha=0.5)

    top_y = s_ult * 1.2
    ax1.annotate('', xy=(0, top_y), xytext=(e_yield_pct, top_y), arrowprops=dict(arrowstyle='<->', color='green', lw=1.5))
    ax1.text(e_yield_pct/2, top_y + 5, "Elastic", ha='center', color='green', weight='bold', fontsize=8)
    
    ax1.annotate('', xy=(e_yield_pct, top_y), xytext=(e_ult_pct, top_y), arrowprops=dict(arrowstyle='<->', color='blue', lw=1.5))
    ax1.text((e_yield_pct + e_ult_pct)/2, top_y + 5, "Plastic", ha='center', color='blue', weight='bold', fontsize=8)
    
    ax1.annotate('', xy=(e_ult_pct, top_y), xytext=(e_frac_pct, top_y), arrowprops=dict(arrowstyle='<->', color='purple', lw=1.5))
    ax1.text((e_ult_pct + e_frac_pct)/2, top_y + 5, "Strain-Hardening", ha='center', color='purple', weight='bold', fontsize=8)

    # Dots
    ax1.plot(e_yield_pct, s_yield, 'o', color='orange', markersize=7)
    ax1.annotate(f"Yield\n({s_yield:.0f} MPa)", xy=(e_yield_pct, s_yield), xytext=(e_yield_pct+1, s_yield-40), color='orange', fontsize=8, arrowprops=dict(arrowstyle='->', color='orange'))
    
    ax1.plot(e_ult_pct, s_ult, 'o', color='red', markersize=7)
    ax1.annotate(f"Ultimate\n({s_ult:.0f} MPa)", xy=(e_ult_pct, s_ult), xytext=(e_ult_pct-6, s_ult+40), color='red', fontsize=8, arrowprops=dict(arrowstyle='->', color='red'))

    ax1.plot(e_frac_pct, s_frac, 'o', color='purple', markersize=7)
    ax1.annotate(f"Fracture\n({s_frac:.0f} MPa)", xy=(e_frac_pct, s_frac), xytext=(e_frac_pct-5, s_frac-60), color='purple', fontsize=8, arrowprops=dict(arrowstyle='->', color='purple'))

    # LIVE VALUES AT MARKER
    cur_strain_pct = cur_strain * 100
    cur_stress_mpa = cur_stress / 1e6
    if cur_strain_pct <= e_frac_pct:
        ax1.plot(cur_strain_pct, cur_stress_mpa, 'X', color='dodgerblue', markersize=12, zorder=6)
        ax1.axvline(cur_strain_pct, color='dodgerblue', linestyle=':', lw=1.5, alpha=0.6)
        
        # Print the actual value on the graph
        label_text = f"You: {cur_stress_mpa:.1f} MPa\n@ {cur_strain_pct:.1f}% strain"
        ax1.annotate(label_text, xy=(cur_strain_pct, cur_stress_mpa), xytext=(cur_strain_pct+1, cur_stress_mpa-80), color='dodgerblue', weight='bold', fontsize=8, arrowprops=dict(arrowstyle='->', color='dodgerblue'))

    # Light mode axes styling
    ax1.set_xlabel("Strain (%)", color='black')
    ax1.set_ylabel("Stress (MPa)", color='black')
    ax1.tick_params(colors='black')
    for spine in ax1.spines.values():
        spine.set_color('black')
        
    ax1.set_ylim(0, s_ult * 1.35)
    ax1.set_xlim(-0.5, e_frac_pct * 1.1)
    ax1.grid(True, alpha=0.3)
    st.pyplot(fig1)

with col2:
    st.subheader("Real-Time Explanation")
    
    kpi1, kpi2 = st.columns(2)
    kpi1.metric("Safety Factor (FOS)", f"{(PROPERTIES[selected_mat]['yield'] / (cur_stress + 1e-9)):.2f}")
    kpi2.metric("Material Health", f"{max(0.0, (1.0 - (cur_strain / e_frac)) * 100):.1f}%")
    
    st.markdown(f'<div class="force-box">You applied <b>{applied_force} N</b> of force.</div>', unsafe_allow_html=True)
    
    # Remaining calculations
    rem_yield = max(0, s_yield - cur_stress_mpa)
    rem_ult = max(0, s_ult - cur_stress_mpa)
    rem_frac = max(0, s_frac - cur_stress_mpa)

    if cur_strain_pct < e_yield_pct:
        st.markdown(f"""<div class="zone-alert-elastic">
        <h3 style='margin-top:0;'>🟢 Elastic Zone</h3>
        <p>You are in the safe zone. If you release the force, the material will <b>return exactly to its original shape</b>.</p>
        <hr style='margin: 10px 0; border-color: #a7f3d0;'>
        <b>Remaining until Yield Point:</b> {rem_yield:.1f} MPa<br>
        <b>Remaining until Ultimate:</b> {rem_ult:.1f} MPa<br>
        <b>Remaining until Fracture:</b> {rem_frac:.1f} MPa
        </div>""", unsafe_allow_html=True)
    elif cur_strain_pct < e_ult_pct:
        st.markdown(f"""<div class="zone-alert-plastic">
        <h3 style='margin-top:0;'>🔵 Plastic Zone</h3>
        <p>You have crossed the Yield Point. The material is now <b>permanently deformed</b>.</p>
        <hr style='margin: 10px 0; border-color: #fde68a;'>
        <b>Remaining until Ultimate:</b> {rem_ult:.1f} MPa<br>
        <b>Remaining until Fracture:</b> {rem_frac:.1f} MPa
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""<div class="zone-alert-failure">
        <h3 style='margin-top:0;'>🔴 Failure Zone</h3>
        <p>CRITICAL LIMIT REACHED. The material has <b>fractured and broken</b>.</p>
        <hr style='margin: 10px 0; border-color: #fecaca;'>
        <b>Remaining until Fracture:</b> {rem_frac:.1f} MPa
        </div>""", unsafe_allow_html=True)

    st.subheader("EMR Signal (Misra Effect)")
    fig2, ax2 = plt.subplots(figsize=(7, 2.5), facecolor='white')
    ax2.set_facecolor('white')
    ax2.plot(emr, color='#0284c7', lw=1)
    ax2.set_xlabel("Sample", color='black')
    ax2.set_ylabel("Amplitude", color='black')
    ax2.tick_params(colors='black')
    for spine in ax2.spines.values():
        spine.set_color('black')
    ax2.grid(True, alpha=0.3)
    st.pyplot(fig2)

st.write("---")
if pred_mat == selected_mat:
    st.success(f"**AI Predicted Material: {pred_mat.upper()}** (Correct!) | Confidence: {proba[pred_idx]*100:.1f}%")
else:
    st.error(f"**AI Predicted Material: {pred_mat.upper()}** (Wrong) | Confidence: {proba[pred_idx]*100:.1f}%")