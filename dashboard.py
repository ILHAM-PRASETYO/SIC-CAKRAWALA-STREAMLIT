import streamlit as st
import paho.mqtt.client as mqtt
import pandas as pd
import numpy as np
from datetime import datetime
import time
import json
import threading
import plotly.graph_objects as go 

# ====================================================================
# KONFIGURASI MQTT (TIDAK BERUBAH)
# ====================================================================
MQTT_SERVER = "broker.hivemq.com" 
MQTT_PORT = 1883

# --- Topik (Subscription: Data dari ESP32/ML Server) ---
TOPIC_STATUS_BRANKAS = "data/status/kontrol"
TOPIC_DIST = "data/dist/kontrol"
TOPIC_PIR = "data/pir/kontrol" 
TOPIC_ML_FACE_RESULT = "ai/face/result"
TOPIC_ML_VOICE_RESULT = "ai/voice/result"
TOPIC_CAM_PHOTO_URL = "/iot/camera/photo"
TOPIC_AUDIO_LINK = "data/audio/link"

# --- Topik (Publication: Perintah ke ESP32/Camera) ---
TOPIC_CAM_TRIGGER = "/iot/camera/trigger"
TOPIC_ALARM_CONTROL = "data/alarm/kontrol"
# ====================================================================
# INISIALISASI STREAMLIT SESSION STATE (TIDAK BERUBAH)
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
# FUNGSI LOGIKA DAN CALLBACK MQTT (TIDAK BERUBAH)
# ====================================================================

# Fungsi untuk menggabungkan prediksi dan sensor (TIDAK BERUBAH)
def generate_final_prediction(row):
    wajah = row.get("Prediksi Wajah", "")
    suara = row.get("Prediksi Suara", "")
    jarak = row.get("Jarak (cm)", np.nan)
    pir = row.get("PIR", np.nan)
    status = row.get("Status Brankas", "")

    if "Brangkas Dibuka Paksa" in status:
        return "⚠ Dibobol!"
    if wajah == "Unknown" or suara == "Not_User" or wajah == "OTHER_FACES": 
        return "🚨 Mencurigakan!"
    if "Terbuka Secara Aman" in status:
        return "✅ Sah & Aman"
    if pd.notna(jarak) and jarak > 0 and jarak < 25: 
        return "👀 Aktivitas Dekat"
    if pd.notna(pir) and pir == 1:
        return "👀 Gerakan Terdeteksi"
    return "✅ Aman"

# Callback MQTT (TIDAK BERUBAH)
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
        
    elif not st.session_state.df_brankas.empty:
        last_index = st.session_state.df_brankas.index[-1]
        
        if topic == TOPIC_DIST:
            try:
                distance = float(payload)
                st.session_state.df_brankas.loc[last_index, 'Jarak (cm)'] = distance
            except ValueError:
                pass

        elif topic == TOPIC_PIR:
            try:
                pir_val = int(payload)
                st.session_state.df_brankas.loc[last_index, 'PIR'] = pir_val
            except ValueError:
                pass

        elif topic == TOPIC_ML_FACE_RESULT:
            st.session_state.df_brankas.loc[last_index, 'Prediksi Wajah'] = payload

        elif topic == TOPIC_ML_VOICE_RESULT:
            st.session_state.df_brankas.loc[last_index, 'Prediksi Suara'] = payload

    elif topic == TOPIC_CAM_PHOTO_URL:
        st.session_state.photo_url = f"{payload}?t={int(time.time())}"
        
    elif topic == TOPIC_AUDIO_LINK:
        st.session_state.audio_url = f"{payload}?t={int(time.time())}" 
        st.info(f"Link audio baru diterima: {st.session_state.audio_url}")

# ====================================================================
# INISIALISASI & LOOP MQTT (TIDAK BERUBAH)
# ====================================================================
mqtt_client = mqtt.Client()
mqtt_client.on_message = on_mqtt_message
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
# FUNGSI TAMBAHAN LOG ML (results.json) - Dipertahankan
# ====================================================================
def load_new_ml_results():
    # ... (kode load_new_ml_results Anda) ...
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
# UI Dashboard (Struktur Direvisi)
# ====================================================================
st.title("🔒 Dashboard Monitoring Brankas & AI")

# --- BAGIAN ATAS (CHART & LOG UTAMA) ---
chart_col, log_col = st.columns([2, 1])

with chart_col:
    st.header("Live Chart (Jarak & PIR)")
    df_plot = st.session_state.df_brankas.tail(200).copy()
    
    # Hapus baris dengan nilai NaN pada Jarak dan PIR agar chart bersih
    df_plot.dropna(subset=['Jarak (cm)', 'PIR'], inplace=True) 

    if not df_plot.empty:
        fig = go.Figure()
        
        # Scatter plot untuk Jarak (cm)
        fig.add_trace(go.Scatter(
            x=df_plot["Timestamp"], 
            y=df_plot["Jarak (cm)"], 
            mode="lines+markers", 
            name="Jarak (cm)"
        ))
        
        # Scatter plot untuk PIR (Nilai 0 atau 1)
        fig.add_trace(go.Scatter(
            x=df_plot["Timestamp"], 
            y=df_plot["PIR"], 
            mode="lines+markers", 
            name="PIR (Gerakan)", 
            yaxis="y2"
        ))

        fig.update_layout(
            yaxis=dict(title="Jarak (cm)"),
            yaxis2=dict(
                title="PIR (0=Aman, 1=Gerak)", 
                overlaying="y", 
                side="right", 
                showgrid=False,
                range=[-0.1, 1.1] 
            ),
            height=520,
            legend=dict(x=0, y=1.1, orientation="h")
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Menunggu data Jarak/PIR untuk menampilkan chart.")
    
    # === FITUR DOWNLOAD CSV (Ditempatkan di bawah chart) ===
    csv_data = st.session_state.df_brankas.to_csv(index=False).encode("utf-8")
    is_data_available = not st.session_state.df_brankas.empty
    
    if is_data_available:
        st.download_button(
            label="⬇️ Download Semua Log CSV",
            data=csv_data,
            file_name=f"brankas_logs_{int(time.time())}.csv",
            mime="text/csv"
        )
    else:
        st.info("Tidak ada data log untuk diunduh.")


with log_col:
    st.header("Log Status Brankas")
    st.dataframe(st.session_state.df_brankas.tail(10)) 
    st.markdown("### Kontrol Cepat")
    
    # Pindahkan tombol kontrol cepat ke sini
    control_col1, control_col2, control_col3 = st.columns(3)
    
    with control_col1:
        if st.button("📷 Ambil Foto", use_container_width=True):
            mqtt_client.publish(TOPIC_CAM_TRIGGER, "capture")
            
    with control_col2:
        if st.button("🔇 Matikan Alarm", use_container_width=True):
            mqtt_client.publish(TOPIC_ALARM_CONTROL, "OFF")
            
    with control_col3:
        if st.button("🔄 Refresh Foto", use_container_width=True):
            st.session_state.photo_url = f"{st.session_state.photo_url.split('?')[0]}?t={int(time.time())}"

    st.subheader("Foto Terbaru")
    st.image(st.session_state.photo_url, use_column_width=True)

# --- GARIS PEMISAH (opsional) ---
st.markdown("---")

# --- BAGIAN BAWAH (TAB UNTUK DETAIL ML) ---
tab1, tab2, tab3 = st.tabs(["🏠 Detail Brankas", "🖼️ Log Prediksi Wajah", "🔊 Log Prediksi Suara"])

# Tab Detail Brankas (sekarang bisa berisi log lama/lainnya)
with tab1:
    st.subheader("Log Data Brankas Lengkap")
    st.dataframe(st.session_state.df_brankas)


# Tab Log Prediksi Wajah
with tab2:
    st.subheader("Log Prediksi Wajah")
    st.dataframe(st.session_state.df_face.tail(10))

# Tab Log Prediksi Suara
with tab3:
    st.subheader("Audio Terbaru untuk Analisis")
    if st.session_state.audio_url:
        st.audio(st.session_state.audio_url, format='audio/wav')
    else:
        st.info("Menunggu rekaman audio terbaru...")

    st.subheader("Log Prediksi Suara")
    st.dataframe(st.session_state.df_voice.tail(10))
    
# Refresh otomatis
time.sleep(2)
st.rerun()
