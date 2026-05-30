import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import json
import cv2
import io
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import Image as MpImage
from PIL import Image
import os
import time
import tensorflow as tf
import gdown
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import threading

st.set_page_config(
    page_title="BISINDO Translator",
    page_icon="🤟",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #f5f0e8; color: #1a1a1a; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; max-width: 720px; }
.hero { text-align: center; padding: 2.5rem 0 1.5rem; }
.hero-logo { font-family: 'Syne', sans-serif; font-size: 3rem; font-weight: 800; letter-spacing: -2px; color: #f0ede6; line-height: 1; }
.hero-logo span { color: #7dd3fc; }
.hero-tag { font-size: 0.82rem; font-weight: 300; color: #888; letter-spacing: 3px; text-transform: uppercase; margin-top: 0.5rem; }
.result-box { background: #111; border: 1px solid #222; border-radius: 20px; padding: 2rem; margin: 1rem 0; text-align: center; }
.sign-label { font-family: 'Syne', sans-serif; font-size: 6rem; font-weight: 800; color: #7dd3fc; line-height: 1; letter-spacing: -4px; }
.conf-badge { display: inline-block; background: #7dd3fc; color: #0a0a0a; font-size: 0.75rem; font-weight: 600; letter-spacing: 1px; padding: 4px 14px; border-radius: 100px; margin-top: 0.8rem; }
.warn-badge { display: inline-block; background: #374151; color: #9ca3af; font-size: 0.75rem; padding: 4px 14px; border-radius: 100px; margin-top: 0.8rem; }
.top3-card { background: #111; border: 1px solid #1e1e1e; border-radius: 14px; padding: 1rem 1.5rem; margin-top: 1rem; }
.top3-row { display: flex; justify-content: space-between; align-items: center; padding: 0.6rem 0; border-bottom: 1px solid #1a1a1a; }
.top3-label { font-size: 0.9rem; color: #aaa; font-weight: 500; }
.top3-conf { font-size: 0.85rem; color: #555; font-family: 'Syne', sans-serif; }
.bar-bg { height: 3px; background: #1e1e1e; border-radius: 2px; margin-top: 3px; }
.bar-fill { height: 100%; background: #7dd3fc; border-radius: 2px; }
.lm-card { background: #0d1a24; border: 1px solid #1e3a5f; border-left: 3px solid #7dd3fc; border-radius: 12px; padding: 0.8rem 1.2rem; margin-top: 0.8rem; font-size: 0.82rem; color: #7dd3fc; }
.warn-card { background: #1a0d0d; border: 1px solid #ef4444; border-left: 3px solid #ef4444; border-radius: 12px; padding: 1rem 1.5rem; margin: 1rem 0; color: #ef4444; font-size: 0.88rem; line-height: 1.6; }
.timer-overlay { background: rgba(10,10,10,0.95); border: 1px solid #f59e0b; border-radius: 20px; padding: 1.5rem; margin: 0.5rem 0; text-align: center; }
.timer-count { font-family: 'Syne', sans-serif; font-size: 5rem; font-weight: 800; color: #f59e0b; line-height: 1; }
.timer-label { font-size: 0.85rem; color: #888; margin-top: 0.3rem; }
.history-strip { display: flex; gap: 0.5rem; flex-wrap: wrap; padding: 1rem; background: #111; border: 1px solid #1e1e1e; border-radius: 12px; margin-top: 1rem; min-height: 56px; align-items: center; }
.history-char { font-family: 'Syne', sans-serif; font-size: 1.5rem; font-weight: 700; color: #7dd3fc; background: #0d1a24; padding: 4px 14px; border-radius: 8px; }
.tip-card { background: #0d1a24; border: 1px solid #1e3a5f; border-radius: 12px; padding: 1rem 1.5rem; margin-bottom: 1rem; font-size: 0.85rem; color: #7dd3fc; line-height: 1.6; }
.download-card { background: #0d1a24; border: 1px solid #1e3a5f; border-radius: 16px; padding: 1.5rem; margin: 1rem 0; text-align: center; }
.download-title { font-family: 'Syne', sans-serif; font-size: 1.2rem; font-weight: 700; color: #7dd3fc; margin-bottom: 0.5rem; }
.download-sub { font-size: 0.82rem; color: #888; }
.disclaimer { text-align: center; font-size: 0.73rem; color: #444; margin-top: 2.5rem; padding-bottom: 2rem; line-height: 1.6; }
</style>
""", unsafe_allow_html=True)

# ── Konstanta ─────────────────────────────────────────────────
SEQUENCE_LENGTH      = 10
CONFIDENCE_THRESHOLD = 0.60
NOISE_STD            = 0.01
DRIVE_FOLDER_ID      = "1cMUKR6orOm8VeRS0CVpf4ZgpVLqNaV_r"
REQUIRED_FILES       = ["bisindo_model.keras", "hand_landmarker.task", "bisindo_assets.json"]

# ── Download dari Google Drive ────────────────────────────────
@st.cache_resource
def download_models():
    missing = [f for f in REQUIRED_FILES if not os.path.exists(f)]
    if not missing:
        return True
    placeholder = st.empty()
    placeholder.markdown("""
    <div class="download-card">
        <div class="download-title">⬇️ Mengunduh model dari Google Drive...</div>
        <div class="download-sub">Hanya sekali saat pertama deploy · Mohon tunggu sebentar</div>
    </div>
    """, unsafe_allow_html=True)
    try:
        gdown.download_folder(
            f"https://drive.google.com/drive/folders/{DRIVE_FOLDER_ID}",
            output=".", quiet=False, use_cookies=False
        )
        placeholder.empty()
        return True
    except Exception as e:
        placeholder.error(f"❌ Gagal download dari Google Drive: {e}")
        return False

@st.cache_resource
def load_hand_detector():
    base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
    options      = vision.HandLandmarkerOptions(
        base_options=base_options, num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
    return vision.HandLandmarker.create_from_options(options)

@st.cache_resource
def load_bisindo_model():
    return tf.keras.models.load_model("bisindo_model.keras")

@st.cache_data
def load_assets():
    with open("bisindo_assets.json", "r") as f:
        return json.load(f)

# ── Image helper ──────────────────────────────────────────────
def show_image(arr: np.ndarray, caption: str = ""):
    pil = Image.fromarray(np.ascontiguousarray(arr, dtype=np.uint8))
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=92)
    buf.seek(0)
    st.image(buf, caption=caption, use_container_width=True)

# ── Core functions ────────────────────────────────────────────
def extract_landmarks(img_rgb: np.ndarray, detector):
    arr      = np.ascontiguousarray(img_rgb, dtype=np.uint8)
    mp_image = MpImage(image_format=mp.ImageFormat.SRGB, data=arr)
    result   = detector.detect(mp_image)
    if result.hand_landmarks:
        lm_list = []
        for lm in result.hand_landmarks[0]:
            lm_list.extend([lm.x, lm.y, lm.z])
        return np.array(lm_list, dtype=np.float32), result.hand_landmarks[0]
    return None, None

def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    reshaped   = landmarks.reshape(21, 3)
    normalized = reshaped - reshaped[0]
    scale      = np.max(np.abs(normalized)) + 1e-8
    return (normalized / scale).flatten()

def make_sequence(lm_norm: np.ndarray) -> np.ndarray:
    seq = []
    for _ in range(SEQUENCE_LENGTH):
        n = np.random.normal(0, NOISE_STD, lm_norm.shape).astype(np.float32)
        seq.append(np.clip(lm_norm + n, -1, 1))
    return np.array(seq, dtype=np.float32)

def predict_sign(sequence: np.ndarray, model, class_names: list) -> dict:
    probs    = model.predict(np.expand_dims(sequence, 0), verbose=0)[0]
    top_idx  = int(np.argmax(probs))
    top_conf = float(probs[top_idx])
    top3_idx = np.argsort(probs)[::-1][:3]
    return {
        'label':        class_names[top_idx],
        'confidence':   top_conf,
        'is_confident': top_conf >= CONFIDENCE_THRESHOLD,
        'top3': [{'label': class_names[i], 'confidence': float(probs[i])} for i in top3_idx]
    }

def draw_landmarks(img_rgb: np.ndarray, hand_landmarks) -> np.ndarray:
    h, w        = img_rgb.shape[:2]
    bgr         = cv2.cvtColor(np.ascontiguousarray(img_rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR)
    connections = [
        (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
        (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
        (0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17)
    ]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
    for a, b in connections:
        cv2.line(bgr, pts[a], pts[b], (255, 200, 100), 2)
    for x, y in pts:
        cv2.circle(bgr, (x, y), 5, (150, 255, 0), -1)
        cv2.circle(bgr, (x, y), 5, (255, 255, 255), 1)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def render_result(result: dict):
    if result['is_confident']:
        st.markdown(f"""
        <div class="result-box">
            <div class="sign-label">{result['label']}</div>
            <div style="color:#888; font-size:0.85rem; margin-top:0.4rem">Huruf BISINDO</div>
            <span class="conf-badge">KEYAKINAN {result['confidence']:.0%}</span>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-box">
            <div class="sign-label" style="color:#374151">?</div>
            <div style="color:#666; font-size:0.85rem; margin-top:0.4rem">Gestur tidak dikenali</div>
            <span class="warn-badge">CONFIDENCE {result['confidence']:.0%} — DI BAWAH THRESHOLD</span>
            <div style="color:#555; font-size:0.8rem; margin-top:0.8rem">
                Tebakan: <strong style="color:#9ca3af">{result['top3'][0]['label']}</strong>
            </div>
        </div>""", unsafe_allow_html=True)
    top3_html = ""
    for r in result['top3']:
        bar_w = int(r['confidence'] * 100)
        top3_html += f"""
        <div class="top3-row">
            <div>
                <div class="top3-label">{r['label']}</div>
                <div class="bar-bg"><div class="bar-fill" style="width:{bar_w}%"></div></div>
            </div>
            <div class="top3-conf">{r['confidence']:.1%}</div>
        </div>"""
    st.markdown(f'<div class="top3-card">{top3_html}</div>', unsafe_allow_html=True)

def process_image(img_rgb: np.ndarray, model, class_names: list, hand_detector):
    img_rgb = np.ascontiguousarray(img_rgb, dtype=np.uint8)
    with st.spinner("Mendeteksi tangan..."):
        try:
            landmarks, hand_lm = extract_landmarks(img_rgb, hand_detector)
        except Exception as e:
            st.error(f"Error deteksi: {e}")
            show_image(img_rgb)
            return
    if hand_lm is not None:
        try:
            annotated = draw_landmarks(img_rgb, hand_lm)
            show_image(annotated, caption="Landmark terdeteksi ✅")
        except Exception as e:
            st.error(f"Error draw: {e}")
            show_image(img_rgb)
        st.markdown("""
        <div class="lm-card">
            ✅ <strong>21 landmark terdeteksi</strong> ·
            63 fitur diekstrak via MediaPipe Hand Landmarker
        </div>""", unsafe_allow_html=True)
    else:
        show_image(img_rgb)
        st.markdown("""
        <div class="warn-card">
            ✋ <strong>Tangan tidak terdeteksi.</strong><br>
            <span style="color:#888">
            • Latar polos / kontras<br>
            • Tangan penuh di frame<br>
            • Pencahayaan cukup dari depan
            </span>
        </div>""", unsafe_allow_html=True)
    if landmarks is not None:
        try:
            lm_norm  = normalize_landmarks(landmarks)
            sequence = make_sequence(lm_norm)
            result   = predict_sign(sequence, model, class_names)
            render_result(result)
            if result['is_confident']:
                if st.button("➕ Tambah ke kalimat", use_container_width=True):
                    st.session_state.history.append(result['label'])
                    st.rerun()
        except Exception as e:
            st.error(f"Error prediksi: {e}")

# ── Session state ─────────────────────────────────────────────
if 'history' not in st.session_state:
    st.session_state.history = []

# ═════════════════════════════════════════════════════════════
# UI
# ═════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
    <div class="hero-logo">BISINDO<span>.</span>AI</div>
    <div class="hero-tag">Real-time · Indonesian Sign Language · Accessibility</div>
</div>""", unsafe_allow_html=True)

ok = download_models()
if not ok:
    st.stop()

try:
    hand_detector = load_hand_detector()
    model         = load_bisindo_model()
    assets        = load_assets()
    CLASS_NAMES   = assets['class_names']
except Exception as e:
    st.error(f"⚠️ Gagal memuat model: {e}")
    st.stop()

mode = st.radio("Mode:", ["📸 Upload Foto", "🎥 Webcam Real-time"],
                horizontal=True, label_visibility="collapsed")
st.markdown("<hr style='border-color:#1e1e1e; margin:1rem 0'>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# MODE 1 — Upload Foto
# ══════════════════════════════════════════════════════════════
if mode == "📸 Upload Foto":
    uploaded = st.file_uploader("Upload foto", type=["jpg","jpeg","png"],
                                label_visibility="collapsed")
    if uploaded is None:
        st.markdown("""
        <div style="border:1.5px dashed #333; border-radius:16px; padding:2.5rem;
        text-align:center; background:#111; margin:1rem 0">
            <div style="font-size:2.5rem">🤟</div>
            <div style="color:#666; margin-top:0.5rem">Upload foto gestur tangan BISINDO</div>
            <div style="font-size:0.75rem; color:#444; margin-top:0.3rem">
                Pastikan tangan terlihat jelas · Latar polos lebih baik
            </div>
        </div>""", unsafe_allow_html=True)
    else:
        img_pil = Image.open(uploaded).convert('RGB')
        img_rgb = np.ascontiguousarray(np.array(img_pil), dtype=np.uint8)
        process_image(img_rgb, model, CLASS_NAMES, hand_detector)

# Di app.py — Mode 2 final
else:
    st.markdown("""
    <div class="tip-card">
        💡 <strong>Tips:</strong>
        <span style="color:#888">
        Latar polos · Cahaya dari depan · Tangan penuh di frame · Jarak 30–60cm
        </span>
    </div>
    """, unsafe_allow_html=True)

    timer_sec = st.selectbox(
        "⏱️ Timer sebelum jepret otomatis:",
        [0, 3, 5, 10],
        format_func=lambda x: "Langsung jepret (tanpa timer)" if x == 0
                              else f"{x} detik — siapkan pose dulu",
        key="timer_select"
    )

    # ── Inject JS countdown yang trigger st.camera_input ─────
    # st.camera_input render tombol dengan aria-label "Take Photo"
    # JS kita klik tombol itu setelah countdown selesai
    countdown_js = f"""
    <script>
    const TIMER = {timer_sec};

    function waitForCamButton(callback) {{
      // Cari tombol "Take Photo" di seluruh DOM parent Streamlit
      const check = () => {{
        // st.camera_input berada di iframe parent
        const frames = window.parent.document.querySelectorAll('button[data-testid="stCameraInputButton"]');
        if (frames.length > 0) {{
          callback(frames[0]);
        }} else {{
          setTimeout(check, 200);
        }}
      }};
      check();
    }}

    // Buat UI countdown terpisah
    const ui = document.getElementById('countdown-ui');

    function runCountdown() {{
      if (TIMER === 0) {{
        waitForCamButton(btn => btn.click());
        return;
      }}

      let rem = TIMER;
      ui.innerHTML = `
        <div style="
          background:rgba(245,158,11,0.15);
          border:1px solid #f59e0b;
          border-radius:16px;
          padding:1.5rem;
          text-align:center;
          font-family:'Syne',sans-serif;
        ">
          <div style="font-size:5rem;font-weight:800;color:#f59e0b;line-height:1" id="cd-num">${{rem}}</div>
          <div style="color:#9ca3af;margin-top:0.3rem;font-size:0.85rem">
            Siapkan gestur — jepret otomatis dalam ${{rem}} detik
          </div>
        </div>`;

      const iv = setInterval(() => {{
        rem--;
        const el = document.getElementById('cd-num');
        if (el) el.textContent = rem <= 0 ? '📸' : rem;
        if (rem <= 0) {{
          clearInterval(iv);
          setTimeout(() => {{
            waitForCamButton(btn => btn.click());
            ui.innerHTML = '';
          }}, 300);
        }}
      }}, 1000);
    }}

    // Tunggu tombol start diklik (kita buat tombolnya)
    document.getElementById('start-btn').addEventListener('click', () => {{
      document.getElementById('start-btn').style.display = 'none';
      runCountdown();
    }});
    </script>
    """

    # Tombol mulai timer (di atas kamera)
    label = "📸 Jepret Sekarang" if timer_sec == 0 else f"⏱️ Mulai Timer {timer_sec} Detik"
    components.html(f"""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Syne:wght@800&display=swap');
      body {{ background: transparent; margin: 0; }}
      #start-btn {{
        width: 100%; padding: 14px;
        background: #7dd3fc; color: #0a0a0a;
        font-family: 'Syne', sans-serif; font-weight: 800;
        font-size: 1rem; letter-spacing: 1px;
        border: none; border-radius: 12px;
        cursor: pointer; transition: opacity 0.2s;
      }}
      #start-btn:hover {{ opacity: 0.85; }}
    </style>
    <button id="start-btn">{label}</button>
    <div id="countdown-ui" style="margin-top:10px"></div>
    {countdown_js}
    """, height=120 if timer_sec == 0 else 180)

    # ── st.camera_input native (yang di-trigger JS) ───────────
    img_file = st.camera_input(
        "Arahkan tangan ke kamera",
        label_visibility="collapsed",
        key="cam_input"
    )

    if img_file is not None:
        img_pil = Image.open(img_file).convert('RGB')
        img_arr = np.array(img_pil, dtype=np.uint8)
        img_rgb = np.ascontiguousarray(cv2.flip(img_arr, 1), dtype=np.uint8)
        process_image(img_rgb, model, CLASS_NAMES, hand_detector)

# ══════════════════════════════════════════════════════════════
# HISTORY
# ══════════════════════════════════════════════════════════════
if st.session_state.history:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**📝 Kalimat yang dibentuk:**")
    chars_html = ''.join([
        f'<span class="history-char">{c}</span>' for c in st.session_state.history
    ])
    st.markdown(f'<div class="history-strip">{chars_html}</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⌫ Hapus terakhir", use_container_width=True):
            st.session_state.history.pop()
            st.rerun()
    with col2:
        if st.button("🗑️ Reset semua", use_container_width=True):
            st.session_state.history = []
            st.rerun()

with st.expander("📋 26 huruf BISINDO yang didukung"):
    cols = st.columns(6)
    for i, cls in enumerate(sorted(CLASS_NAMES)):
        cols[i % 6].markdown(
            f"<div style='text-align:center; background:#111; border-radius:8px;"
            f"padding:8px 4px; margin:2px; font-family:Syne,sans-serif;"
            f"font-weight:800; color:#7dd3fc; font-size:1.1rem'>{cls}</div>",
            unsafe_allow_html=True
        )

st.markdown("""
<div class="disclaimer">
    BISINDO.AI · MediaPipe Hand Landmarker + Bidirectional LSTM<br>
    Mendukung aksesibilitas komunitas tuli/bisu Indonesia<br>
    Alat bantu komunikasi — bukan pengganti interpreter profesional
</div>""", unsafe_allow_html=True)
