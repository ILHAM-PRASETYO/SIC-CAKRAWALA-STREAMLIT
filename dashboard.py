import streamlit as st
import paho.mqtt.client as mqtt
import pandas as pd
import numpy as np
from datetime import datetime
import time
import json
import threading

# Inisialisasi session state
for key in ["df_face", "df_voice", "df_brankas"]:
    if key not in st.session_state:
        if key == "df_brankas":
            st.session_state[key] = pd.DataFrame(columns=[
                "Timestamp", "Status Brankas", "Jarak (cm)", "PIR", "Prediksi Wajah", "Prediksi Suara", "Label Prediksi"
            ])
        else:
            st.session_state[key] = pd.DataFrame(columns=["Timestamp", "Hasil Prediksi", "Akurasi (%)", "Status", "Keterangan"])

if 'last_face_time' not in st.session_state:
    st.session_state.last_face_time = None

if 'last_voice_time' not in st.session_state:
    st.session_state.last_voice_time = None

if 'photo_url' not in st.session_state:
    st.session_state.photo_url = "https://via.placeholder.com/640x480?text=No+Photo+Yet"

# Fungsi untuk menggabungkan prediksi dan sensor
def generate_final_prediction(row):
    wajah = row.get("Prediksi Wajah", "")
    suara = row.get("Prediksi Suara", "")
    jarak = row.get("Jarak (cm)", np.nan)
    pir = row.get("PIR", np.nan)
    status = row.get("Status Brankas", "")

    if "Dibuka Paksa" in status:
        return "⚠ Dibobol!"
    if wajah == "Wajah Tidak Dikenal" or suara == "Suara Mencurigakan":
        return "🚨 Mencurigakan!"
    if wajah == "Wajah Dikenal" and suara == "Suara Sah":
        return "✅ Sah & Aman"
    if pd.notna(jarak) and jarak < 30:
        return "👀 Aktivitas Terdeteksi"
    if pd.notna(pir) and pir == 1:
        return "👀 Gerakan Terdeteksi"
    return "✅ Aman"

# Callback MQTT
def on_mqtt_message(client, userdata, msg):
    topic = msg.topic
    try:
        payload = msg.payload.decode("utf-8").strip()
    except UnicodeDecodeError:
        return

    if topic == "data/status/kontrol":
        new_row = {
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Status Brankas": payload,
            "Jarak (cm)": np.nan,
            "PIR": np.nan,
            "Prediksi Wajah": "any",
            "Prediksi Suara": "any",
            "Label Prediksi": "Belum Diproses"
        }
        st.session_state.df_brankas = pd.concat([
            st.session_state.df_brankas, pd.DataFrame([new_row])
        ], ignore_index=True)

    elif topic == "data/ldr/kontrol":
        try:
            distance = float(payload)
            if not st.session_state.df_brankas.empty:
                st.session_state.df_brankas.loc[st.session_state.df_brankas.index[-1], 'Jarak (cm)'] = distance
        except:
            pass

    elif topic == "data/pir/kontrol":
        try:
            pir_val = int(payload)
            if not st.session_state.df_brankas.empty:
                st.session_state.df_brankas.loc[st.session_state.df_brankas.index[-1], 'PIR'] = pir_val
        except:
            pass

    elif topic == "/ai/face/result":
        if not st.session_state.df_brankas.empty:
            st.session_state.df_brankas.loc[st.session_state.df_brankas.index[-1], 'Prediksi Wajah'] = payload

    elif topic == "/ai/voice/result":
        if not st.session_state.df_brankas.empty:
            st.session_state.df_brankas.loc[st.session_state.df_brankas.index[-1], 'Prediksi Suara'] = payload

    elif topic == "/iot/camera/photo":
        st.session_state.photo_url = f"{payload}?t={int(time.time())}"

# Inisialisasi MQTT
mqtt_client = mqtt.Client()
mqtt_client.on_message = on_mqtt_message
mqtt_client.connect("localhost", 1883, 60)
mqtt_client.subscribe([
    ("data/status/kontrol", 0),
    ("data/ldr/kontrol", 0),
    ("data/pir/kontrol", 0),
    ("/ai/face/result", 0),
    ("/ai/voice/result", 0),
    ("/iot/camera/photo", 0)
])

def mqtt_loop():
    mqtt_client.loop_forever()

mqtt_thread = threading.Thread(target=mqtt_loop, daemon=True)
mqtt_thread.start()

# Fungsi baca hasil ML baru
def load_new_ml_results():
    try:
        with open("results.json", "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        return {"face": [], "voice": []}

    new_face = []
    new_voice = []

    for item in data["face"]:
        item_time = datetime.strptime(item["Timestamp"], "%Y-%m-%d %H:%M:%S")
        if st.session_state.last_face_time is None or item_time > st.session_state.last_face_time:
            new_face.append(item)

    for item in data["voice"]:
        item_time = datetime.strptime(item["Timestamp"], "%Y-%m-%d %H:%M:%S")
        if st.session_state.last_voice_time is None or item_time > st.session_state.last_voice_time:
            new_voice.append(item)

    if new_face:
        st.session_state.last_face_time = datetime.strptime(new_face[-1]["Timestamp"], "%Y-%m-%d %H:%M:%S")
    if new_voice:
        st.session_state.last_voice_time = datetime.strptime(new_voice[-1]["Timestamp"], "%Y-%m-%d %H:%M:%S")

    return {"face": new_face, "voice": new_voice}

# Tambahkan hasil ML baru ke log
new_results = load_new_ml_results()

for item in new_results["face"]:
    st.session_state.df_face = pd.concat([
        st.session_state.df_face, pd.DataFrame([item])
    ], ignore_index=True)

for item in new_results["voice"]:
    st.session_state.df_voice = pd.concat([
        st.session_state.df_voice, pd.DataFrame([item])
    ], ignore_index=True)

# Update label prediksi di df_brankas
if not st.session_state.df_brankas.empty:
    st.session_state.df_brankas["Label Prediksi"] = st.session_state.df_brankas.apply(
        generate_final_prediction, axis=1
    )

# UI Dashboard
st.title("🔒 Dashboard Monitoring Brankas & AI")
tab1, tab2, tab3 = st.tabs(["🏠 Brankas", "🖼️ ML Gambar", "🔊 ML Audio"])

# Tab Brankas
with tab1:
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📷 Ambil Foto"):
            mqtt_client.publish("/iot/camera/trigger", "capture")
    with col2:
        if st.button("🔇 Matikan Alarm"):
            mqtt_client.publish("data/allert/control", "off")
    with col3:
        if st.button("🔄 Refresh Gambar"):
            st.session_state.photo_url = f"{st.session_state.photo_url.split('?')[0]}?t={int(time.time())}"

    st.subheader("Foto Terbaru")
    st.image(st.session_state.photo_url, use_column_width=True)

    st.subheader("Log Brankas")
    st.dataframe(st.session_state.df_brankas.tail(10))

# Tab ML Gambar
with tab2:
    st.subheader("Log Prediksi Wajah")
    st.dataframe(st.session_state.df_face.tail(10))

# Tab ML Suara
with tab3:
    st.subheader("Log Prediksi Suara")
    st.dataframe(st.session_state.df_voice.tail(10))

# Refresh otomatis
time.sleep(2)
st.rerun()