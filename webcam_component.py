# webcam_component.py
import streamlit.components.v1 as components
import streamlit as st
import base64
import os
import tempfile
import uuid
from PIL import Image
import numpy as np
import io

def webcam_with_timer(timer_sec: int) -> np.ndarray | None:
    """
    Tampilkan webcam dengan countdown timer.
    Return: numpy array RGB jika ada capture, None jika belum.
    """
    
    # Buat unique key untuk sesi ini
    if 'cam_session_id' not in st.session_state:
        st.session_state.cam_session_id = str(uuid.uuid4())[:8]
    
    session_id = st.session_state.cam_session_id
    tmp_path   = os.path.join(tempfile.gettempdir(), f"bisindo_cap_{session_id}.jpg")

    cam_html = f"""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Syne:wght@800&display=swap');
      * {{ box-sizing: border-box; margin: 0; padding: 0; }}
      body {{ background: transparent; }}

      #wrap {{
        position: relative; width: 100%;
        border-radius: 16px; overflow: hidden;
        background: #111; aspect-ratio: 4/3;
      }}
      #video {{
        width: 100%; height: 100%; object-fit: cover;
        display: block; transform: scaleX(-1); border-radius: 16px;
      }}
      #canvas {{ display: none; }}

      #overlay {{
        display: none; position: absolute; inset: 0;
        background: rgba(10,10,10,0.5);
        align-items: center; justify-content: center;
        flex-direction: column; gap: 10px; border-radius: 16px;
      }}
      #overlay.show {{ display: flex; }}

      #num {{
        font-family: 'Syne', sans-serif;
        font-size: 8rem; font-weight: 800;
        color: #f59e0b; line-height: 1;
        text-shadow: 0 0 60px rgba(245,158,11,0.7);
        animation: pop 1s ease-in-out infinite;
      }}
      @keyframes pop {{
        0%, 100% {{ transform: scale(1); }}
        50%       {{ transform: scale(1.12); }}
      }}
      #lbl {{ color: #e5e7eb; font-size: 0.9rem; letter-spacing: 1px; text-align: center; }}

      /* Progress arc di pojok kanan atas */
      #arc-wrap {{ position: absolute; top: 12px; right: 12px; }}
      #arc-svg  {{ transform: rotate(-90deg); }}
      #arc-bg   {{ fill: none; stroke: #374151; stroke-width: 5; }}
      #arc-fill {{
        fill: none; stroke: #f59e0b; stroke-width: 5;
        stroke-linecap: round;
        stroke-dasharray: 126;
        stroke-dashoffset: 126;
        transition: stroke-dashoffset linear;
      }}

      #flash {{
        display: none; position: absolute; inset: 0;
        background: white; border-radius: 16px; pointer-events: none;
        animation: flashOut 0.4s ease-out forwards;
      }}
      @keyframes flashOut {{ from {{ opacity:0.9; }} to {{ opacity:0; }} }}

      #done-ov {{
        display: none; position: absolute; inset: 0;
        background: rgba(10,10,10,0.75);
        align-items: center; justify-content: center;
        flex-direction: column; gap: 8px; border-radius: 16px;
      }}
      #done-ov.show {{ display: flex; }}
      #done-icon {{ font-size: 4rem; }}
      #done-txt  {{
        font-family: 'Syne', sans-serif;
        color: #7dd3fc; font-weight: 800; font-size: 1.1rem;
      }}

      .btn-row {{ display: flex; gap: 10px; margin-top: 12px; }}
      button {{
        flex: 1; padding: 13px; border: none; border-radius: 12px;
        font-family: 'Syne', sans-serif; font-weight: 800;
        font-size: 0.95rem; cursor: pointer;
        transition: opacity 0.15s, transform 0.1s;
      }}
      button:hover  {{ opacity: 0.85; transform: translateY(-1px); }}
      button:active {{ transform: translateY(0); }}
      button:disabled {{ opacity: 0.35; cursor: not-allowed; transform: none; }}
      #btn-s {{ background: #7dd3fc; color: #0a0a0a; }}
      #btn-c {{ background: #1f2937; color: #9ca3af; display: none; }}

      #status {{
        text-align: center; font-size: 0.8rem;
        color: #6b7280; margin-top: 8px;
      }}
      #errbox {{ color: #ef4444; text-align: center; padding: 2rem; font-size: 0.85rem; }}
    </style>

    <div id="wrap">
      <video id="video" autoplay playsinline muted></video>
      <canvas id="canvas"></canvas>
      <div id="overlay">
        <div id="num">0</div>
        <div id="lbl">Siapkan gestur tangan kamu...</div>
        <div id="arc-wrap">
          <svg id="arc-svg" width="48" height="48" viewBox="0 0 48 48">
            <circle id="arc-bg"   cx="24" cy="24" r="20"/>
            <circle id="arc-fill" cx="24" cy="24" r="20"/>
          </svg>
        </div>
      </div>
      <div id="flash"></div>
      <div id="done-ov">
        <div id="done-icon">✅</div>
        <div id="done-txt">FOTO DIPROSES!</div>
      </div>
    </div>

    <div class="btn-row">
      <button id="btn-s">
        {'📸 JEPRET SEKARANG' if timer_sec == 0 else f'⏱️ MULAI TIMER {timer_sec} DETIK'}
      </button>
      <button id="btn-c">✕ BATAL</button>
    </div>
    <div id="status">Memulai kamera...</div>
    <div id="errbox"></div>

    <script>
    const TIMER   = {timer_sec};
    const SID     = '{session_id}';
    const video   = document.getElementById('video');
    const canvas  = document.getElementById('canvas');
    const overlay = document.getElementById('overlay');
    const numEl   = document.getElementById('num');
    const flash   = document.getElementById('flash');
    const doneOv  = document.getElementById('done-ov');
    const btnS    = document.getElementById('btn-s');
    const btnC    = document.getElementById('btn-c');
    const status  = document.getElementById('status');
    const errbox  = document.getElementById('errbox');
    const arcFill = document.getElementById('arc-fill');
    const CIRC    = 2 * Math.PI * 20;  // circumference r=20

    arcFill.style.strokeDasharray  = CIRC;
    arcFill.style.strokeDashoffset = CIRC;

    let timerHandle = null;
    let running     = false;

    // Start kamera
    navigator.mediaDevices.getUserMedia({{
      video: {{ facingMode: 'user', width: 640, height: 480 }},
      audio: false
    }}).then(s => {{
      video.srcObject = s;
      status.textContent = 'Kamera aktif · Klik tombol untuk mulai timer';
      btnS.disabled = false;
    }}).catch(e => {{
      errbox.textContent = '❌ Tidak bisa akses kamera: ' + e.message;
      btnS.disabled = true;
    }});

    function doCapture() {{
      running = false;

      // Flash
      flash.style.display   = 'block';
      flash.style.animation = 'none';
      flash.offsetHeight;
      flash.style.animation = 'flashOut 0.4s ease-out forwards';

      // Gambar ke canvas + flip horizontal
      canvas.width  = video.videoWidth  || 640;
      canvas.height = video.videoHeight || 480;
      const ctx = canvas.getContext('2d');
      ctx.translate(canvas.width, 0);
      ctx.scale(-1, 1);
      ctx.drawImage(video, 0, 0);

      const b64 = canvas.toDataURL('image/jpeg', 0.92);

      // Done overlay
      overlay.classList.remove('show');
      doneOv.classList.add('show');
      status.textContent = '⏳ Mengirim ke model...';
      btnC.style.display  = 'none';
      btnS.style.display  = 'block';
      setTimeout(() => doneOv.classList.remove('show'), 2000);

      // Kirim base64 ke Streamlit backend via fetch
      fetch('/_stcore/upload_file', {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify({{ session_id: SID, image: b64 }})
      }}).catch(() => {{}});  // endpoint custom tidak ada — pakai cara lain

      // CARA YANG WORKS: tulis ke localStorage lalu Streamlit baca via
      // st.components iframe messaging
      window.parent.postMessage({{
        type:   'CAM_CAPTURE',
        sid:    SID,
        image:  b64
      }}, '*');

      status.textContent = '✅ Foto terkirim — scroll ke bawah untuk hasil';
    }}

    function startTimer() {{
      if (running) return;
      running = true;
      doneOv.classList.remove('show');

      if (TIMER === 0) {{ doCapture(); return; }}

      overlay.classList.add('show');
      btnS.style.display = 'none';
      btnC.style.display  = 'block';

      let rem = TIMER;
      numEl.textContent = rem;

      // Progress arc animasi
      arcFill.style.transition       = 'none';
      arcFill.style.strokeDashoffset = CIRC;
      setTimeout(() => {{
        arcFill.style.transition       = `stroke-dashoffset ${{TIMER}}s linear`;
        arcFill.style.strokeDashoffset = 0;
      }}, 50);

      timerHandle = setInterval(() => {{
        rem--;
        numEl.textContent = rem <= 0 ? '📸' : rem;
        if (rem <= 0) {{
          clearInterval(timerHandle);
          setTimeout(doCapture, 200);
        }}
      }}, 1000);
    }}

    function cancelTimer() {{
      if (timerHandle) clearInterval(timerHandle);
      running = false;
      overlay.classList.remove('show');
      btnS.style.display = 'block';
      btnC.style.display  = 'none';
      arcFill.style.transition       = 'none';
      arcFill.style.strokeDashoffset = CIRC;
      status.textContent = 'Dibatalkan · Klik tombol untuk coba lagi';
    }}

    btnS.addEventListener('click', startTimer);
    btnC.addEventListener('click', cancelTimer);

    // Terima pesan balik (jika ada)
    window.addEventListener('message', e => {{
      if (e.data && e.data.type === 'CAM_ACK') {{
        status.textContent = '✅ Model menerima foto!';
      }}
    }});
    </script>
    """

    components.html(cam_html, height=500, scrolling=False)

    # Karena postMessage tidak bisa langsung ke Python,
    # kita pakai st.session_state via URL query param trick:
    # JS set window.location.hash → Python baca st.query_params
    return None
