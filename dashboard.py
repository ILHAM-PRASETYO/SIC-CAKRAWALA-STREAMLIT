import streamlit as st
import paho.mqtt.client as mqtt
import pandas as pd
import numpy as np
from datetime import datetime
import time
import json
import threading

# ====================================================================
# KONFIGURASI MQTT
# ====================================================================
MQTT_SERVER = "broker.hivemq.com" 
MQTT_PORT = 1883

# --- Topik (Subscription: Data dari ESP32/ML Server) ---
TOPIC_STATUS_BRANKAS = "data/status/kontrol" # Status umum brankas (Aman/Terbuka/Dibobol)
TOPIC_DIST = "data/dist/kontrol"            # Sensor Jarak Ultrasonik
TOPIC_PIR = "data/pir/kontrol"               # Sensor PIR
TOPIC_ML_FACE_RESULT = "ai/face/result"     # Hasil Prediksi Wajah (dari ML Server)
TOPIC_ML_VOICE_RESULT = "ai/voice/result"   # Hasil Prediksi Suara (dari ML Server)
TOPIC_CAM_PHOTO_URL = "/iot/camera/photo"    # URL Foto Terbaru
TOPIC_AUDIO_LINK = "data/audio/link"

# --- Topik (Publication: Perintah ke ESP32/Camera) ---
TOPIC_CAM_TRIGGER = "/iot/camera/trigger"    # Perintah Ambil Foto
TOPIC_ALARM_CONTROL = "data/alarm/kontrol"   # Perintah Matikan/Nyalakan Alarm

# ====================================================================
# INISIALISASI STREAMLIT SESSION STATE
# ====================================================================
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

if 'audio_url' not in st.session_state:
    st.session_state.audio_url = None
# ====================================================================
# FUNGSI LOGIKA DAN CALLBACK MQTT
# ====================================================================

# Fungsi untuk menggabungkan prediksi dan sensor
def generate_final_prediction(row):
    wajah = row.get("Prediksi Wajah", "")
    suara = row.get("Prediksi Suara", "")
    jarak = row.get("Jarak (cm)", np.nan)
    pir = row.get("PIR", np.nan)
    status = row.get("Status Brankas", "")

    if "Brangkas Dibuka Paksa" in status:
        return "⚠ Dibobol!"
    # Asumsi: Prediksi ML wajah/suara yang tidak dikenali
    if wajah == "Unknown" or suara == "Not_User" or wajah == "OTHER_FACES": 
        return "🚨 Mencurigakan!"
    if "Terbuka Secara Aman" in status:
        return "✅ Sah & Aman"
    if pd.notna(jarak) and jarak > 0 and jarak < 25: # Jarak > 0 untuk menghilangkan pembacaan sensor yang gagal (0)
        return "👀 Aktivitas Dekat"
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

    if topic == TOPIC_STATUS_BRANKAS:
        new_row = {
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Status Brankas": payload,
            "Jarak (cm)": np.nan,
            "PIR": np.nan,
            "Prediksi Wajah": "Menunggu...",
            "Prediksi Suara": "Menunggu...",
            "Label Prediksi": "Belum Diproses"
        }
        st.session_state.df_brankas = pd.concat([
            st.session_state.df_brankas, pd.DataFrame([new_row])
        ], ignore_index=True)

    elif topic == TOPIC_DIST:
        try:
            distance = float(payload)
            if not st.session_state.df_brankas.empty:
                st.session_state.df_brankas.loc[st.session_state.df_brankas.index[-1], 'Jarak (cm)'] = distance
        except:
            pass

    elif topic == TOPIC_PIR:
        try:
            pir_val = int(payload)
            if not st.session_state.df_brankas.empty:
                st.session_state.df_brankas.loc[st.session_state.df_brankas.index[-1], 'PIR'] = pir_val
        except:
            pass

    elif topic == TOPIC_ML_FACE_RESULT:
        if not st.session_state.df_brankas.empty:
            st.session_state.df_brankas.loc[st.session_state.df_brankas.index[-1], 'Prediksi Wajah'] = payload

    elif topic == TOPIC_ML_VOICE_RESULT:
        if not st.session_state.df_brankas.empty:
            st.session_state.df_brankas.loc[st.session_state.df_brankas.index[-1], 'Prediksi Suara'] = payload

    elif topic == TOPIC_CAM_PHOTO_URL:
        st.session_state.photo_url = f"{payload}?t={int(time.time())}"
    
    elif topic == TOPIC_AUDIO_LINK:
        st.session_state.audio_url = f"{payload}?t={int(time.time())}" 
        st.info(f"Link audio baru diterima: {st.session_state.audio_url}") #

# ====================================================================
# INISIALISASI & LOOP MQTT
# ====================================================================

mqtt_client = mqtt.Client()
mqtt_client.on_message = on_mqtt_message
# 🛑 FIX UTAMA: Menggunakan broker.hivemq.com alih-alih localhost
mqtt_client.connect(MQTT_SERVER, MQTT_PORT, 60) 
mqtt_client.subscribe([
    (TOPIC_STATUS_BRANKAS, 0),
    (TOPIC_DIST, 0),
    (TOPIC_PIR, 0),
    (TOPIC_ML_FACE_RESULT, 0),
    (TOPIC_ML_VOICE_RESULT, 0),
    (TOPIC_CAM_PHOTO_URL, 0),
    (TOPIC_AUDIO_LINK, 0)
])

def mqtt_loop():
    mqtt_client.loop_forever()

mqtt_thread = threading.Thread(target=mqtt_loop, daemon=True)
mqtt_thread.start()

# ====================================================================
# FUNGSI TAMBAHAN LOG ML (results.json)
# ====================================================================

def load_new_ml_results():
    # Fungsi ini dipertahankan meski data ML kini harusnya lewat MQTT
    # Tapi ini bisa digunakan untuk historical data dari file
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

# ====================================================================
# UI Dashboard
# ====================================================================
st.title("🔒 Dashboard Monitoring Brankas & AI")
tab1, tab2, tab3 = st.tabs(["🏠 Brankas", "🖼️ ML Gambar", "🔊 ML Audio"])

# Tab Brankas
with tab1:
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📷 Ambil Foto"):
            # Menggunakan konstanta TOPIC_CAM_TRIGGER
            mqtt_client.publish(TOPIC_CAM_TRIGGER, "capture")
    with col2:
        if st.button("🔇 Matikan Alarm"):
            # Menggunakan konstanta TOPIC_ALARM_CONTROL dan pesan "OFF" (huruf kapital)
            mqtt_client.publish(TOPIC_ALARM_CONTROL, "OFF")
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
    st.subheader("Audio Terbaru untuk Analisis")
    if st.session_state.audio_url:
        # Gunakan st.audio() untuk menampilkan pemutar
        st.audio(st.session_state.audio_url, format='audio/wav')
    else:
        st.info("Menunggu rekaman audio terbaru...")

    st.subheader("Log Prediksi Suara")
    st.dataframe(st.session_state.df_voice.tail(10))
# Refresh otomatis
time.sleep(2)
st.rerun()
