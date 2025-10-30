# app_v4.0.py
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ---------------------------
# Config / translation maps
# ---------------------------
# disease -> Sinhala (lowercase keys)
disease_si = {
    "parvovirus": "පාර්වෝ වෛරස් රෝගය (බල්ලා)",
    "distemper": "ඩිස්ටෙම්පර් රෝගය (බල්ලා)",
    "skin_allergy": "සම අසාත්මිකතාව (බල්ලා / පූසා)",
    "mastitis": "ස්තන ආසාදනය (ගවයා)",
    "foot_and_mouth": "පා සහ මුව රෝගය (ගවයා)",
    "bovine_diarrhea": "ගවයාගේ දියවැඩියාව",
    "bovine_tuberculosis": "ගවයාගේ ටියුබර්කියුලෝසිස් (TB)",
    "feline_allergy": "පූසාගේ සම අසාත්මිකතාව",
    "feline_arthritis": "පූසාගේ අස්ථි රෝගය",
    "feline_panleukopenia": "පූසාගේ පෑන්ලියුකෝපෙනියා",
    "feline_uti": "පූසාගේ මූත්‍රා පද්ධති ආසාදනය (UTI)",
    "arthritis": "අස්ථි සන්ධි රෝගය (බල්ලා / ගවයා)",
    "ketosis": "කීටෝසිස් (ගවයා)",
    "kidney_disease": "වකුගඩු රෝගය (බල්ලා / පූසා)",
    "rhinotracheitis": "රයිනෝට්‍රැකයිටිස් (පූසා)",
}

# mapping disease -> species (for filtering). Values may be single species or underscore-separated set.
disease_species = {
    "Parvovirus": "dog",
    "Distemper": "dog",
    "Skin_Allergy": "dog_cat",
    "Arthritis": "dog_cow",
    "Kidney_Disease": "dog_cat",
    "Feline_Panleukopenia": "cat",
    "Rhinotracheitis": "cat",
    "Feline_UTI": "cat",
    "Feline_Allergy": "cat",
    "Feline_Arthritis": "cat",
    "Mastitis": "cow",
    "Foot_and_Mouth": "cow",
    "Bovine_Diarrhea": "cow",
    "Bovine_Tuberculosis": "cow",
    "Ketosis": "cow",
}

# species keywords to detect in user input (Sinhala + English)
species_keywords = {
    "dog": ["dog", "බල්ලා", "canine"],
    "cat": ["cat", "පූසා", "feline"],
    "cow": ["cow", "ගවයා", "cattle", "ගෝමිය"],
}

# ---------------------------
# Load model components
# ---------------------------
try:
    model = joblib.load("real_animal_disease_model.joblib")
    le = joblib.load("label_encoder.joblib")
    vectorizer = joblib.load("vectorizer.joblib")
except Exception as e:
    st.error(
        "Model files not found or failed to load. Ensure the files are in this folder:\n"
        "real_animal_disease_model.joblib\nlabel_encoder.joblib\nvectorizer.joblib"
    )
    st.stop()

# ---------------------------
# Streamlit config + mobile CSS
# ---------------------------
st.set_page_config(page_title="🐾 AI Vet Diagnosis v4.0", page_icon="🐶", layout="centered")
st.markdown(
    """
<style>
    @media (max-width: 600px) {
        h1 {font-size: 28px !important;}
        h3 {font-size: 20px !important;}
        p, div, input, button, label {font-size: 16px !important;}
    }
    .alert {
        animation: blink 1s infinite;
        background-color: #E74C3C;
        color: white;
        text-align: center;
        padding: 10px;
        border-radius: 10px;
        margin-top: 10px;
    }
    @keyframes blink {50% { background-color: #C0392B; }}
    .small-muted {font-size:13px;color:#6c757d;}
</style>
""",
    unsafe_allow_html=True,
)
st.markdown("<h1 style='text-align:center;color:#1B2631;'>🐾 AI Animal Disease Diagnosis v4.0</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center;' class='small-muted'>Type animal + symptoms in English or Sinhala (eg: 'dog fever vomiting' or 'බල්ලා උණ වමන')</p>",
    unsafe_allow_html=True,
)

# ---------------------------
# Input area
# ---------------------------
user_input = st.text_input(
    "👉 Type here / මෙතන ලියා දෙන්න:",
    "",
    placeholder="dog fever vomiting diarrhea  — or —  බල්ලා උණ වමන",
)

if st.button("🔍 Diagnose / විශ්ලේෂණය කරන්න"):
    if not user_input.strip():
        st.warning("Please enter symptoms / කරුණාකර රෝග ලක්ෂණ ඇතුලත් කරන්න.")
    else:
        # vectorize and predict
        X = vectorizer.transform([user_input.lower()])
        probs = model.predict_proba(X)[0]  # order corresponds to model.classes_
        # map model.classes_ to disease names (label encoder inverse)
        try:
            classes = le.inverse_transform(model.classes_)
        except Exception:
            # fallback: le.classes_ (should normally work)
            classes = le.classes_

        # create dataframe of all classes + probs
        all_df = pd.DataFrame({"disease": classes, "prob": probs})
        all_df = all_df.sort_values("prob", ascending=False).reset_index(drop=True)

        # validation: if max prob < threshold -> likely invalid symptoms
        if all_df.loc[0, "prob"] < 0.10:
            st.error("❌ කරුණාකර නිවැරදි රෝග ලක්ෂණ ඇතුළත් කරන්න! / Please enter valid disease symptoms.")
        else:
            # detect species from input
            txt = user_input.lower()
            detected_species = None
            for sp, keys in species_keywords.items():
                for k in keys:
                    if k in txt:
                        detected_species = sp
                        break
                if detected_species:
                    break

            # Post-prediction species-aware filtering
            adjusted = []
            for _, row in all_df.iterrows():
                d = row["disease"]
                p = float(row["prob"])
                dspecies = disease_species.get(d, None)
                if detected_species and dspecies:
                    # if disease is multi-species like 'dog_cat' -> split and allow multiple
                    if "_" in dspecies:
                        parts = dspecies.split("_")
                        if detected_species not in parts:
                            # penalize strongly
                            p *= 0.05
                    else:
                        if detected_species != dspecies:
                            p *= 0.05
                adjusted.append((d, p))

            total = sum([p for _, p in adjusted])
            if total <= 0:
                # fallback to original probs normalized
                adjusted = [(d, p) for d, p in adjusted]
                total = sum([p for _, p in adjusted]) or 1.0
            adjusted = [(d, p / total) for d, p in adjusted]

            # sort and pick top-3
            adjusted_sorted = sorted(adjusted, key=lambda x: x[1], reverse=True)
            top3 = adjusted_sorted[:3]

            # Display results
            st.markdown("<h3 style='color:#2E86C1;'>🩺 Sambhāvanā Rōga (Top-3 Probable Diseases):</h3>", unsafe_allow_html=True)
            chart_rows = []
            alert_triggered = False

            for i, (disease, prob) in enumerate(top3):
                conf = prob * 100
                key = disease.strip().lower()
                siname = disease_si.get(key, "— සිංහල නාමය නොමැත —")
                bilingual = f"{disease} ({siname})"

                if conf >= 50:
                    color = "#E74C3C"
                    advice = "🔴 👉 Visit a qualified vet immediately. වෛද්‍යවරයෙකු වෙත ගිය යුතුය."
                    alert_triggered = True
                elif conf >= 20:
                    color = "#F39C12"
                    advice = "🟠 Possible but less likely. Monitor carefully. සතුන්ගේ තත්ත්වය නිරීක්ෂණය කරන්න."
                else:
                    color = "#27AE60"
                    advice = "🟢 Mild risk. Observe for new symptoms. අඩු අවදානමක් ඇත."

                # render card
                st.markdown(
                    f"""
                    <div style="background-color:{color}15; border-left:6px solid {color}; border-radius:12px; padding:12px; margin-bottom:10px;">
                        <h4 style="color:{color}; font-size:20px; margin-bottom:6px;">{i+1}. {bilingual} — {conf:.1f}%</h4>
                        <p style="font-size:15px; color:{color}; font-weight:500;">{advice}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                chart_rows.append({"Disease": bilingual, "Confidence (%)": conf, "Color": color})

            # Notification banner for serious cases
            if alert_triggered:
                st.markdown("<div class='alert'>🚨 Serious condition detected! වහාම වෛද්‍යවරයෙකු වෙත ගිය යුතුය!</div>", unsafe_allow_html=True)

            # Chart
            df_chart = pd.DataFrame(chart_rows)
            fig = px.bar(
                df_chart,
                x="Disease",
                y="Confidence (%)",
                color="Color",
                color_discrete_map={c: c for c in df_chart["Color"]},
                text=df_chart["Confidence (%)"].round(1).astype(str) + "%",
                title="📊 Confidence Comparison (විශ්වාස තත්ත්වය)",
            )
            fig.update_traces(textposition="outside", marker_line_width=1.2, marker_line_color="#2C3E50")
            fig.update_layout(
                title_font_size=18,
                yaxis=dict(title="Confidence (%)"),
                xaxis=dict(title="Disease (රෝගය)", tickangle=-30),
                showlegend=False,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, use_container_width=True)

            # Save diagnosis history (append)
            try:
                hist_file = "diagnosis_history.csv"
                preds_text = ";".join([f"{r['Disease']}|{r['Confidence (%)']:.1f}%" for r in chart_rows])
                history_row = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "input": user_input,
                    "species_detected": detected_species or "",
                    "predictions": preds_text,
                }
                if not os.path.exists(hist_file):
                    pd.DataFrame([history_row]).to_csv(hist_file, index=False)
                else:
                    pd.DataFrame([history_row]).to_csv(hist_file, mode="a", header=False, index=False)
                st.caption("🔒 Diagnosis saved to diagnosis_history.csv")
            except Exception:
                pass

            st.success("✅ Analysis Complete / විශ්ලේෂණය අවසන්!")
