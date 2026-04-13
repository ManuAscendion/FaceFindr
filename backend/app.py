import io
import shutil
import tempfile
import sys
from pathlib import Path

import streamlit as st
from PIL import Image

from face_utils import find_matching_images, clear_embedding_cache

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Face Findr ✦",
    page_icon="📸",
    layout="wide",
)

# ── Constants ──────────────────────────────────────────────────────────────────
EVENTS_ROOT_DIR  = Path("event_images")   # each sub-folder = one event
TMP_DIR          = Path(tempfile.gettempdir())
APP_NAME         = "Face Findr"

ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "admin123"   # ← change before going live

EVENTS_ROOT_DIR.mkdir(exist_ok=True)

# ── GLOBAL CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800;900&family=Caveat:wght@600;700&display=swap" rel="stylesheet">

<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

.stApp {
    background: #FDFAF4;
    font-family: 'Nunito', sans-serif;
}
section[data-testid="stSidebar"] { display: none !important; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 2rem 4rem !important; max-width: 1200px; }

h1,h2,h3,p,label,span,div { font-family: 'Nunito', sans-serif; }
.stMarkdown p { color: #3d3929 !important; }

/* ── Text inputs ── */
.stTextInput > div > div > input {
    background: #fff !important;
    border: 2.5px solid #e8e0cc !important;
    border-radius: 14px !important;
    color: #2d2a1e !important;
    font-family: 'Nunito', sans-serif !important;
    font-size: 15px !important;
    padding: 10px 16px !important;
    transition: border-color 0.2s, box-shadow 0.2s;
}
.stTextInput > div > div > input:focus {
    border-color: #FF6B6B !important;
    box-shadow: 0 0 0 4px rgba(255,107,107,0.15) !important;
}
.stTextInput label {
    color: #7a6f55 !important;
    font-size: 13px !important;
    font-weight: 700 !important;
}

/* ── Selectbox (all) ── */
.stSelectbox label {
    color: #7a6f55 !important;
    font-size: 13px !important;
    font-weight: 700 !important;
}
/* Make ALL selectboxes look like proper dropdowns */
.stSelectbox [data-baseweb="select"] > div:first-child {
    background: #fff !important;
    border: 2.5px solid #e8e0cc !important;
    border-radius: 14px !important;
    font-family: 'Nunito', sans-serif !important;
    font-size: 14px !important;
    font-weight: 700 !important;
    color: #2d2a1e !important;
    padding: 10px 16px !important;
    box-shadow: 0 2px 6px rgba(0,0,0,0.06) !important;
    cursor: pointer !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
.stSelectbox [data-baseweb="select"] > div:first-child:hover {
    border-color: #FF6B6B !important;
    box-shadow: 0 3px 10px rgba(255,107,107,0.15) !important;
}
/* Red arrow icon */
.stSelectbox [data-baseweb="select"] svg {
    fill: #FF6B6B !important;
    width: 18px !important;
    height: 18px !important;
}

/* ── Buttons (global) ── */
.stButton > button {
    font-family: 'Nunito', sans-serif !important;
    font-weight: 800 !important;
    border-radius: 14px !important;
    border: 2.5px solid transparent !important;
    padding: 11px 22px !important;
    font-size: 15px !important;
    transition: all 0.18s ease !important;
    width: 100%;
    background: #FF6B6B !important;
    color: #fff !important;
    box-shadow: 0 4px 0 #d94f4f !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 0 #d94f4f !important;
    background: #ff5252 !important;
}
.stButton > button:active {
    transform: translateY(1px) !important;
    box-shadow: 0 2px 0 #d94f4f !important;
}
.stButton > button:disabled {
    background: #e8e0cc !important;
    color: #b5a98a !important;
    box-shadow: 0 4px 0 #d4cbb5 !important;
}

/* ── Consent toggle ── */
.consent-btn-wrapper .stButton > button {
    background: #f5f0e8 !important;
    color: #7a6f55 !important;
    border: 2px solid #d4cbb5 !important;
    box-shadow: 0 2px 0 #c4bba5 !important;
    border-radius: 10px !important;
    font-size: 13px !important;
    font-weight: 700 !important;
    padding: 7px 16px !important;
    width: auto !important;
}
.consent-btn-wrapper .stButton > button:hover {
    background: #ede8de !important;
    color: #3d3929 !important;
    border-color: #b5a98a !important;
    box-shadow: 0 3px 0 #b5a98a !important;
    transform: translateY(-1px) !important;
}
.consent-btn-agreed .stButton > button {
    background: #f0fdf4 !important;
    color: #1e7a3a !important;
    border: 2px solid #9de0af !important;
    box-shadow: 0 2px 0 #7acc92 !important;
    border-radius: 10px !important;
    font-size: 13px !important;
    font-weight: 700 !important;
    padding: 7px 16px !important;
    width: auto !important;
}
.consent-btn-agreed .stButton > button:hover {
    background: #e0f9ea !important;
    color: #155a2a !important;
    border-color: #6dc88a !important;
    box-shadow: 0 3px 0 #6dc88a !important;
    transform: translateY(-1px) !important;
}

/* ── Danger button (delete) ── */
.danger-btn .stButton > button {
    background: #fff0f0 !important;
    color: #c0392b !important;
    border: 2px solid #f5a6a6 !important;
    box-shadow: 0 2px 0 #e08080 !important;
    font-size: 13px !important;
}
.danger-btn .stButton > button:hover {
    background: #fde8e8 !important;
    border-color: #e07070 !important;
    transform: translateY(-1px) !important;
}

/* ── Secondary button ── */
.secondary-btn .stButton > button {
    background: #f5f0e8 !important;
    color: #7a6f55 !important;
    border: 2px solid #d4cbb5 !important;
    box-shadow: 0 2px 0 #c4bba5 !important;
    font-size: 13px !important;
    font-weight: 700 !important;
}
.secondary-btn .stButton > button:hover {
    background: #ede8de !important;
    color: #3d3929 !important;
    border-color: #b5a98a !important;
    transform: translateY(-1px) !important;
}

/* ── Admin compact buttons ── */
.admin-action-btn .stButton > button {
    background: #fff !important;
    color: #2d2a1e !important;
    border: 2px solid #e8e0cc !important;
    box-shadow: 0 2px 0 #d4cbb5 !important;
    font-size: 12px !important;
    font-weight: 700 !important;
    padding: 6px 12px !important;
    border-radius: 10px !important;
    width: auto !important;
}
.admin-action-btn .stButton > button:hover {
    border-color: #FF6B6B !important;
    color: #FF6B6B !important;
    box-shadow: 0 2px 0 #d94f4f !important;
    transform: translateY(-1px) !important;
}
.admin-action-btn-active .stButton > button {
    background: #f0fdf4 !important;
    color: #1e7a3a !important;
    border: 2px solid #9de0af !important;
    box-shadow: 0 2px 0 #7acc92 !important;
    font-size: 12px !important;
    font-weight: 700 !important;
    padding: 6px 12px !important;
    border-radius: 10px !important;
    width: auto !important;
}
.admin-del-btn .stButton > button {
    background: #fff0f0 !important;
    color: #c0392b !important;
    border: 2px solid #f5a6a6 !important;
    box-shadow: 0 2px 0 #e08080 !important;
    font-size: 12px !important;
    font-weight: 700 !important;
    padding: 6px 12px !important;
    border-radius: 10px !important;
    width: auto !important;
}
.admin-del-btn .stButton > button:hover {
    background: #fde8e8 !important;
    border-color: #e07070 !important;
    transform: translateY(-1px) !important;
}

/* ── Upload action row buttons ── */
.upload-btn .stButton > button {
    background: #FF6B6B !important;
    color: #fff !important;
    border: none !important;
    box-shadow: 0 3px 0 #d94f4f !important;
    font-size: 13px !important;
    font-weight: 800 !important;
    padding: 9px 18px !important;
    border-radius: 12px !important;
    width: 100% !important;
}
.upload-btn .stButton > button:hover {
    background: #ff5252 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 5px 0 #d94f4f !important;
}
.upload-btn .stButton > button:disabled {
    background: #e8e0cc !important;
    color: #b5a98a !important;
    box-shadow: 0 3px 0 #d4cbb5 !important;
}
.clear-photos-btn .stButton > button {
    background: #fff0f0 !important;
    color: #c0392b !important;
    border: 2px solid #f5a6a6 !important;
    box-shadow: 0 2px 0 #e08080 !important;
    font-size: 13px !important;
    font-weight: 700 !important;
    padding: 9px 18px !important;
    border-radius: 12px !important;
    width: 100% !important;
}
.clear-photos-btn .stButton > button:hover {
    background: #fde8e8 !important;
    border-color: #e07070 !important;
    transform: translateY(-1px) !important;
}
.clear-photos-btn .stButton > button:disabled {
    background: #f5f0e8 !important;
    color: #c4bba5 !important;
    border-color: #e0dbd0 !important;
    box-shadow: none !important;
}
.cache-btn .stButton > button {
    background: #f5f0e8 !important;
    color: #7a6f55 !important;
    border: 2px solid #d4cbb5 !important;
    box-shadow: 0 2px 0 #c4bba5 !important;
    font-size: 13px !important;
    font-weight: 700 !important;
    padding: 9px 18px !important;
    border-radius: 12px !important;
    width: 100% !important;
}
.cache-btn .stButton > button:hover {
    background: #ede8de !important;
    color: #3d3929 !important;
    border-color: #b5a98a !important;
    transform: translateY(-1px) !important;
}

/* ── Slider ── */
.stSlider > div > div > div > div { background: #FF6B6B !important; }
.stSlider label { color: #7a6f55 !important; font-size: 13px !important; font-weight: 700 !important; }

/* ── Checkbox ── */
.stCheckbox { margin-top: 6px !important; }
.stCheckbox [data-baseweb="checkbox"] > div:first-child { display: none !important; }
.stCheckbox [data-baseweb="checkbox"] svg { display: none !important; }
.stCheckbox input[type="checkbox"] { display: none !important; }
.stCheckbox label { color: #2d2a1e !important; font-size: 15px !important; font-weight: 700 !important; cursor: pointer !important; }

/* ── File uploader ── */
.stFileUploader > div {
    background: #fff !important;
    border: 2.5px dashed #d4cbb5 !important;
    border-radius: 16px !important;
}
.stFileUploader > div:hover { border-color: #FF6B6B !important; }
.stFileUploader label { color: #7a6f55 !important; font-weight: 700 !important; }

/* ── Progress bar ── */
.stProgress > div > div > div { background: #FF6B6B !important; border-radius: 100px; }

/* ── Status pills ── */
.pill-ok   { background:#e8f8ec; color:#1e7a3a; border:2px solid #9de0af; border-radius:12px; padding:10px 16px; font-size:13px; font-weight:700; margin-top:10px; line-height:1.5; }
.pill-err  { background:#fdeaea; color:#c0392b; border:2px solid #f5a6a6; border-radius:12px; padding:10px 16px; font-size:13px; font-weight:700; margin-top:10px; line-height:1.5; }
.pill-info { background:#fef9e7; color:#9a6f00; border:2px solid #f5dfa0; border-radius:12px; padding:10px 16px; font-size:13px; font-weight:700; margin-top:10px; line-height:1.5; }
.pill-warn { background:#fff4e0; color:#b35a00; border:2px solid #ffc87a; border-radius:12px; padding:10px 16px; font-size:13px; font-weight:700; margin-top:10px; line-height:1.5; }

/* ── Cards ── */
.fun-card {
    background: #fff;
    border: 2.5px solid #e8e0cc;
    border-radius: 24px;
    padding: 28px;
    margin-bottom: 20px;
    position: relative;
}
.fun-card-yellow { background: #FFFBEC; border-color: #FFE066; }
.fun-card-blue   { background: #EEF4FF; border-color: #B3CFFF; }
.fun-card-green  { background: #F0FDF4; border-color: #9DE0AF; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #fff !important;
    border: 2.5px solid #e8e0cc !important;
    border-radius: 16px !important;
    padding: 6px !important;
    gap: 6px !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border-radius: 12px !important;
    font-family: 'Nunito', sans-serif !important;
    font-weight: 800 !important;
    font-size: 15px !important;
    color: #7a6f55 !important;
    padding: 10px 24px !important;
    border: none !important;
    transition: all 0.18s ease !important;
}
.stTabs [aria-selected="true"] {
    background: #FF6B6B !important;
    color: #fff !important;
    border-radius: 12px !important;
    box-shadow: 0 3px 0 #d94f4f !important;
}
.stTabs [data-baseweb="tab-panel"] { padding-top: 28px !important; }
.stTabs [data-baseweb="tab-border"] { display: none !important; }

/* ── Confidence bar ── */
.cbar-bg   { background:#e8e0cc; border-radius:100px; height:6px; margin-top:6px; overflow:hidden; }
.cbar-fill { background:linear-gradient(90deg,#FF6B6B,#FFD166); height:6px; border-radius:100px; }

/* ── Consent rows ── */
.consent-row { display:flex; gap:12px; margin-bottom:12px; align-items:flex-start; font-size:14px; color:#5a5240; line-height:1.6; }
.consent-icon { font-size:18px; flex-shrink:0; margin-top:1px; }

/* ── Sticker badges ── */
.sticker { display:inline-flex; align-items:center; gap:6px; border-radius:100px; padding:6px 14px; font-size:12px; font-weight:800; letter-spacing:0.3px; }
.sticker-red    { background:#FF6B6B; color:#fff; }
.sticker-yellow { background:#F5A623; color:#fff; }
.sticker-green  { background:#27AE60; color:#fff; }
.sticker-blue   { background:#3B82F6; color:#fff; }
.sticker-purple { background:#8B5CF6; color:#fff; }
.sticker-gray   { background:#9a9080; color:#fff; }

/* ── Event row (compact list) ── */
.event-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: #fff;
    border: 2px solid #e8e0cc;
    border-radius: 14px;
    padding: 10px 14px;
    margin-bottom: 8px;
    gap: 10px;
    transition: border-color 0.2s;
}
.event-row:hover { border-color: #FF6B6B; }
.event-row-active { border-color: #27AE60 !important; background: #f0fdf4 !important; }
.event-row-name { font-size: 14px; font-weight: 800; color: #2d2a1e; flex: 1; min-width: 0;
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.event-row-count { font-size: 11px; color: #a09070; font-weight: 600; white-space: nowrap; }
.event-row-actions { display: flex; gap: 6px; align-items: center; flex-shrink: 0; }

/* ── Divider ── */
.fancy-divider {
    height: 3px;
    background: linear-gradient(90deg,#FF6B6B 0%,#FFD166 33%,#6BCB77 66%,#4D96FF 100%);
    border-radius: 100px;
    margin-bottom: 28px;
    opacity: 0.5;
}

/* ── Upload panel section label ── */
.section-label {
    font-family: 'Caveat', cursive;
    font-size: 20px;
    font-weight: 700;
    color: #2d2a1e;
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* ── Action row ── */
.action-row {
    display: flex;
    gap: 10px;
    align-items: stretch;
    margin-top: 12px;
    flex-wrap: wrap;
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────────

def get_event_folders() -> list[Path]:
    """Return sorted list of event sub-folders inside EVENTS_ROOT_DIR."""
    if not EVENTS_ROOT_DIR.exists():
        return []
    return sorted(
        [p for p in EVENTS_ROOT_DIR.iterdir() if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def count_images_in_folder(folder: Path) -> int:
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    return sum(1 for f in folder.iterdir() if f.suffix.lower() in exts)


def sanitize_folder_name(name: str) -> str:
    """Strip unsafe characters for folder names."""
    safe = "".join(c for c in name if c.isalnum() or c in " _-()").strip()
    return safe[:60] if safe else "Untitled Event"


# ── Session state ──────────────────────────────────────────────────────────────
_defaults = {
    "admin_logged_in":    False,
    "results":            None,
    "consent_given":      False,
    "admin_active_event": None,   # path string of event being managed
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ══════════════════════════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════════════════════════
def render_header():
    admin_badge = ""
    if st.session_state.admin_logged_in:
        admin_badge = (
            '<span style="background:#FFE066;color:#7a5c00;font-size:11px;font-weight:900;'
            'letter-spacing:1px;padding:3px 10px;border-radius:100px;'
            'border:1.5px solid #FFD000;margin-left:10px;vertical-align:middle">ADMIN</span>'
        )

    events = get_event_folders()
    total_events = len(events)
    total_photos = sum(count_images_in_folder(e) for e in events)

    if total_events:
        photos_badge = (
            f'<span class="sticker sticker-green">'
            f'{total_events} event{"s" if total_events != 1 else ""} · {total_photos} photos</span>'
        )
    else:
        photos_badge = '<span class="sticker sticker-red">No events yet</span>'

    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:14px;padding:28px 0 10px;flex-wrap:wrap">
        <div style="position:relative;width:52px;height:52px;flex-shrink:0">
            <div style="width:52px;height:52px;background:#FF6B6B;border-radius:16px;
                        display:flex;align-items:center;justify-content:center;
                        font-size:26px;box-shadow:0 4px 0 #d94f4f">📸</div>
            <div style="position:absolute;bottom:-4px;right:-6px;width:20px;height:20px;
                        background:#FFD166;border-radius:50%;border:2px solid #FDFAF4;
                        display:flex;align-items:center;justify-content:center;font-size:10px">✦</div>
        </div>
        <div>
            <div style="font-family:'Caveat',cursive;font-size:34px;font-weight:700;
                        color:#2d2a1e;line-height:1;letter-spacing:-0.5px">
                Face Findr {admin_badge}
            </div>
            <div style="font-size:12px;color:#a09070;font-weight:700;letter-spacing:0.8px;margin-top:2px">
                Find yourself in every memory
            </div>
        </div>
        <div style="margin-left:auto;display:flex;gap:8px;flex-wrap:wrap;align-items:center">
            <span class="sticker sticker-yellow">Multi-Event</span>
            <span class="sticker sticker-purple">AI-Powered</span>
            {photos_badge}
        </div>
    </div>
    <div class="fancy-divider"></div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  USER TAB  (unchanged)
# ══════════════════════════════════════════════════════════════════════════════
def render_user_tab():

    # ── Hero ──────────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="display:flex;align-items:flex-start;gap:24px;margin-bottom:32px;flex-wrap:wrap">
        <div style="flex:1;min-width:280px">
            <div style="font-family:'Caveat',cursive;font-size:46px;font-weight:700;
                        color:#2d2a1e;line-height:1.05;margin-bottom:12px">
                Find yourself<br>
                <span style="color:#FF6B6B">in every shot</span>
            </div>
            <div style="font-size:15px;color:#7a6f55;line-height:1.7;max-width:460px">
                Pick your event, take a quick selfie, and we'll scan every photo in that
                event to find the ones you're in. Live camera only — no file uploads,
                no stored data, no fuss.
            </div>
        </div>
        <div style="display:flex;flex-direction:column;gap:8px;padding-top:8px">
            <div style="display:flex;align-items:center;gap:10px;background:#fff;
                        border:2px solid #FFD4D4;border-radius:100px;padding:8px 18px 8px 8px">
                <div style="width:28px;height:28px;background:#FF6B6B;border-radius:50%;
                            display:flex;align-items:center;justify-content:center;
                            font-size:13px;font-weight:800;color:#fff;flex-shrink:0">1</div>
                <span style="font-size:14px;font-weight:700;color:#3d3929">Read and agree to the consent</span>
            </div>
            <div style="display:flex;align-items:center;gap:10px;background:#fff;
                        border:2px solid #B3CFFF;border-radius:100px;padding:8px 18px 8px 8px">
                <div style="width:28px;height:28px;background:#3B82F6;border-radius:50%;
                            display:flex;align-items:center;justify-content:center;
                            font-size:13px;font-weight:800;color:#fff;flex-shrink:0">2</div>
                <span style="font-size:14px;font-weight:700;color:#3d3929">Select your event from the list</span>
            </div>
            <div style="display:flex;align-items:center;gap:10px;background:#fff;
                        border:2px solid #FFE5A0;border-radius:100px;padding:8px 18px 8px 8px">
                <div style="width:28px;height:28px;background:#F5A623;border-radius:50%;
                            display:flex;align-items:center;justify-content:center;
                            font-size:13px;font-weight:800;color:#fff;flex-shrink:0">3</div>
                <span style="font-size:14px;font-weight:700;color:#3d3929">Take your selfie live via camera</span>
            </div>
            <div style="display:flex;align-items:center;gap:10px;background:#fff;
                        border:2px solid #A8E6BC;border-radius:100px;padding:8px 18px 8px 8px">
                <div style="width:28px;height:28px;background:#27AE60;border-radius:50%;
                            display:flex;align-items:center;justify-content:center;
                            font-size:13px;font-weight:800;color:#fff;flex-shrink:0">4</div>
                <span style="font-size:14px;font-weight:700;color:#3d3929">Download your photos!</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    left, right = st.columns([1.05, 1], gap="large")

    # ── LEFT: Consent ─────────────────────────────────────────────────────────
    with left:
        st.markdown("""
        <div class="fun-card fun-card-yellow">
            <div style="display:flex;align-items:center;gap:12px;margin-bottom:18px">
                <div style="width:42px;height:42px;background:#FF6B6B;border-radius:12px;
                            display:flex;align-items:center;justify-content:center;flex-shrink:0">
                    <svg width="22" height="22" viewBox="0 0 24 24" fill="none">
                        <rect x="5" y="11" width="14" height="10" rx="2.5" fill="white"/>
                        <path d="M8 11V7.5a4 4 0 0 1 8 0V11" stroke="white" stroke-width="2.2"
                              stroke-linecap="round" fill="none"/>
                        <circle cx="12" cy="16" r="1.5" fill="#FF6B6B"/>
                    </svg>
                </div>
                <div>
                    <div style="font-family:'Caveat',cursive;font-size:22px;font-weight:700;color:#2d2a1e">
                        Privacy &amp; Consent Notice
                    </div>
                    <div style="font-size:12px;color:#9a7a20;font-weight:700;margin-top:1px">
                        Your data stays yours — always
                    </div>
                </div>
            </div>
            <div style="font-size:13px;color:#7a6a3a;margin-bottom:16px;font-weight:600">
                By proceeding, you agree to all of the following:
            </div>
            <div class="consent-row">
                <span class="consent-icon">&#128247;</span>
                <span>I <strong>voluntarily consent</strong> to the live camera capture of my selfie
                by <strong>Face Findr</strong>. I understand this is a live capture, not a file upload.</span>
            </div>
            <div class="consent-row">
                <span class="consent-icon">&#127919;</span>
                <span>My image will be used <strong>only</strong> for real-time identification
                and retrieval of my event photos — nothing else.</span>
            </div>
            <div class="consent-row">
                <span class="consent-icon">&#128465;</span>
                <span>My photo is <strong>processed temporarily</strong> and will <strong>not</strong>
                be stored, saved, or retained after processing is complete.</span>
            </div>
            <div class="consent-row">
                <span class="consent-icon">&#128683;</span>
                <span>My data will <strong>not</strong> be used for profiling, tracking,
                or sharing with any third parties.</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        is_checked = st.session_state.consent_given

        check_svg = """<svg width="14" height="14" viewBox="0 0 14 14" fill="none">
            <path d="M2 7l3.5 3.5L12 3" stroke="white" stroke-width="2.2"
                  stroke-linecap="round" stroke-linejoin="round"/>
        </svg>"""

        if is_checked:
            box_html = f"""
            <div style="display:flex;align-items:center;gap:12px;
                        padding:10px 16px;background:#f0fdf4;
                        border:2px solid #27AE60;border-radius:12px;
                        cursor:pointer;user-select:none;margin-bottom:10px">
                <div style="width:20px;height:20px;border-radius:5px;
                            background:#27AE60;border:2px solid #27AE60;
                            display:flex;align-items:center;justify-content:center;
                            flex-shrink:0">{check_svg}</div>
                <span style="font-size:14px;font-weight:700;color:#1e7a3a">
                    I give my explicit consent for this one-time photo processing.
                </span>
            </div>"""
        else:
            box_html = """
            <div style="display:flex;align-items:center;gap:12px;
                        padding:10px 16px;background:#fff;
                        border:2px solid #e0dbd0;border-radius:12px;
                        cursor:pointer;user-select:none;margin-bottom:10px">
                <div style="width:20px;height:20px;border-radius:5px;
                            background:#fff;border:2.5px solid #aaa89a;
                            flex-shrink:0"></div>
                <span style="font-size:14px;font-weight:700;color:#7a6f55">
                    I give my explicit consent for this one-time photo processing.
                </span>
            </div>"""

        st.markdown(box_html, unsafe_allow_html=True)

        if is_checked:
            st.markdown('<div class="consent-btn-agreed">', unsafe_allow_html=True)
            if st.button("✓ Agreed — click to undo", key="consent_toggle"):
                st.session_state.consent_given = False
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="consent-btn-wrapper">', unsafe_allow_html=True)
            if st.button("Tap to give consent →", key="consent_toggle"):
                st.session_state.consent_given = True
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        if not st.session_state.consent_given:
            st.markdown(
                '<div class="pill-info" style="margin-top:12px">'
                'Accept the consent above to unlock the event picker and camera.</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="pill-ok" style="margin-top:12px">'
                'Consent given — pick your event and take your selfie!</div>',
                unsafe_allow_html=True,
            )

    # ── RIGHT: Event picker + Camera + Search ─────────────────────────────────
    with right:
        st.markdown("""
        <div style="font-family:'Caveat',cursive;font-size:22px;font-weight:700;
                    color:#2d2a1e;margin-bottom:6px">
            Choose your event
        </div>
        <div style="font-size:13px;color:#a09070;font-weight:600;margin-bottom:14px">
            Select the event you attended to search only those photos.
        </div>
        """, unsafe_allow_html=True)

        events = get_event_folders()

        if not events:
            st.markdown("""
            <div style="background:#fff4e0;border:2px solid #ffc87a;border-radius:14px;
                        padding:18px 20px;margin-bottom:18px;text-align:center">
                <div style="font-size:28px;margin-bottom:6px">📭</div>
                <div style="font-size:14px;color:#b35a00;font-weight:700">
                    No events available yet. Check back soon!
                </div>
            </div>
            """, unsafe_allow_html=True)
            selected_event_path = None
        else:
            event_options = {f.name: f for f in events}
            event_labels = list(event_options.keys())

            if not st.session_state.consent_given:
                st.markdown("""
                <div style="background:#f5f0e8;border:2.5px dashed #d4cbb5;border-radius:14px;
                            padding:18px 20px;margin-bottom:16px;text-align:center">
                    <div style="font-size:13px;color:#b5a98a;font-weight:700">
                        🔒 Accept consent to unlock event picker
                    </div>
                </div>
                """, unsafe_allow_html=True)
                selected_event_path = None
            else:
                chosen_name = st.selectbox(
                    "Select an event",
                    options=event_labels,
                    index=0,
                    key="user_event_select",
                    label_visibility="collapsed",
                    placeholder="— Choose an event —",
                )
                selected_event_path = event_options.get(chosen_name)

                if selected_event_path:
                    img_count = count_images_in_folder(selected_event_path)
                    st.markdown(f"""
                    <div style="display:flex;align-items:center;gap:10px;
                                background:#f0fdf4;border:2px solid #9de0af;
                                border-radius:12px;padding:10px 16px;margin-bottom:14px">
                        <span style="font-size:18px">📁</span>
                        <div>
                            <div style="font-size:13px;font-weight:800;color:#1e7a3a">{chosen_name}</div>
                            <div style="font-size:12px;color:#4a9a5a;font-weight:600">
                                {img_count} photo{"s" if img_count != 1 else ""} in this event
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        st.markdown("""
        <div style="font-family:'Caveat',cursive;font-size:22px;font-weight:700;
                    color:#2d2a1e;margin-bottom:6px">
            Take your selfie
        </div>
        <div style="font-size:13px;color:#a09070;font-weight:600;margin-bottom:14px">
            Live camera only — no file uploads allowed for your privacy.
        </div>
        """, unsafe_allow_html=True)

        camera_disabled = not st.session_state.consent_given or selected_event_path is None

        if camera_disabled:
            st.markdown("""
            <div style="background:#f5f0e8;border:2.5px dashed #d4cbb5;border-radius:16px;
                        padding:40px 20px;text-align:center;margin-bottom:16px">
                <div style="font-size:36px;margin-bottom:8px">📷</div>
                <div style="font-size:14px;color:#b5a98a;font-weight:700">
                    Accept consent &amp; pick an event to unlock camera
                </div>
            </div>
            """, unsafe_allow_html=True)
            camera_photo = None
        else:
            camera_photo = st.camera_input(
                "Point your face at the camera",
                key="camera_selfie",
                label_visibility="collapsed",
            )

        if not camera_disabled and camera_photo is not None:
            st.markdown("""
            <div style="display:flex;align-items:center;gap:8px;background:#e8f8ec;
                        border:2px solid #9de0af;border-radius:10px;padding:8px 14px;margin-bottom:12px">
                <span style="font-size:16px">🔒</span>
                <span style="font-size:13px;font-weight:700;color:#1e7a3a">
                    Live capture confirmed — not a file upload
                </span>
            </div>
            """, unsafe_allow_html=True)

        if not camera_disabled:
            threshold = st.slider(
                "Match sensitivity",
                min_value=0.55, max_value=0.85, value=0.65, step=0.01,
                help="Lower = more matches. Higher = fewer but more precise.",
            )
            label = "Strict" if threshold > 0.72 else "Lenient" if threshold < 0.60 else "Balanced"
            st.caption(f"Threshold: {threshold:.2f} — {label}")
        else:
            threshold = 0.65

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        can_search = (
            st.session_state.consent_given
            and camera_photo is not None
            and selected_event_path is not None
            and count_images_in_folder(selected_event_path) > 0
        )

        search_clicked = st.button("🔍 Find My Photos", key="search_btn", disabled=not can_search)

        if search_clicked and can_search:
            selfie_bytes = camera_photo.getvalue()
            with st.spinner("Scanning event photos — hang tight..."):
                try:
                    results = find_matching_images(
                        selfie_bytes=selfie_bytes,
                        event_folder=str(selected_event_path),
                        threshold=threshold,
                        tmp_selfie_path=str(TMP_DIR / "facefindr_selfie.jpg"),
                    )
                    st.session_state.results = results
                except ValueError as e:
                    st.markdown(f'<div class="pill-err">{e}</div>', unsafe_allow_html=True)
                    st.session_state.results = None
                except Exception as e:
                    st.markdown(f'<div class="pill-err">Something went wrong: {e}</div>', unsafe_allow_html=True)
                    st.session_state.results = None

    # ── Results ───────────────────────────────────────────────────────────────
    if st.session_state.results is not None:
        results = st.session_state.results
        st.markdown(
            "<hr style='border:none;border-top:2.5px dashed #e8e0cc;margin:32px 0'>",
            unsafe_allow_html=True,
        )

        if not results:
            st.markdown("""
            <div class="fun-card" style="text-align:center;padding:40px">
                <div style="font-size:48px;margin-bottom:12px">🙈</div>
                <div style="font-family:'Caveat',cursive;font-size:26px;font-weight:700;
                            color:#2d2a1e;margin-bottom:8px">No matches found!</div>
                <div style="font-size:15px;color:#7a6f55;max-width:400px;margin:0 auto">
                    Try sliding the sensitivity lower and searching again.
                    Or maybe you're just camera shy?
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            count = len(results)
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:14px;margin-bottom:24px;flex-wrap:wrap">
                <div style="font-family:'Caveat',cursive;font-size:34px;font-weight:700;color:#2d2a1e">
                    Found you in {count} photo{'s' if count != 1 else ''}!
                </div>
                <span class="sticker sticker-green">Sorted by confidence</span>
            </div>
            """, unsafe_allow_html=True)

            grid = st.columns(4)
            for i, match in enumerate(results):
                img_path = Path(match["filepath"])
                if not img_path.exists():
                    continue
                with grid[i % 4]:
                    st.image(Image.open(img_path), use_container_width=True)
                    confidence = match["confidence"]
                    color = "#1e7a3a" if confidence >= 80 else "#8a6d00" if confidence >= 70 else "#c0392b"
                    st.markdown(f"""
                    <div style="font-size:11px;color:#a09070;margin-top:4px;overflow:hidden;
                                text-overflow:ellipsis;white-space:nowrap"
                         title="{match['filename']}">{match['filename']}</div>
                    <div style="display:flex;justify-content:space-between;font-size:12px;
                                color:#7a6f55;margin-top:4px;font-weight:700">
                        <span>Match</span>
                        <span style="color:{color}">{confidence}%</span>
                    </div>
                    <div class="cbar-bg">
                        <div class="cbar-fill" style="width:{int(confidence)}%"></div>
                    </div>
                    """, unsafe_allow_html=True)
                    with open(img_path, "rb") as f:
                        st.download_button(
                            "⬇ Download",
                            data=f.read(),
                            file_name=match["filename"],
                            mime="image/jpeg",
                            key=f"dl_{i}",
                            use_container_width=True,
                        )

        st.markdown("""
        <p style="text-align:center;font-size:12px;color:#c0b090;margin-top:28px">
            Your selfie has been discarded. No personal data was stored.
        </p>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  ADMIN TAB  (redesigned)
# ══════════════════════════════════════════════════════════════════════════════
def render_admin_tab():
    if not st.session_state.admin_logged_in:
        _, mid, _ = st.columns([1, 1.2, 1])
        with mid:
            st.markdown("""
            <div class="fun-card fun-card-blue" style="text-align:center;padding:36px 32px 24px">
                <div style="font-size:48px;margin-bottom:10px">🔐</div>
                <div style="font-family:'Caveat',cursive;font-size:28px;font-weight:700;
                            color:#2d2a1e;margin-bottom:6px">Admin Login</div>
                <div style="font-size:14px;color:#7a6f55;margin-bottom:24px">
                    This area is for event admins only.
                </div>
            </div>
            """, unsafe_allow_html=True)

            username = st.text_input("Username", placeholder="admin", key="login_user")
            password = st.text_input("Password", type="password", placeholder="••••••••", key="login_pass")
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

            if st.button("Sign in", key="login_btn"):
                if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
                    st.session_state.admin_logged_in = True
                    st.rerun()
                else:
                    st.markdown(
                        '<div class="pill-err">Incorrect username or password.</div>',
                        unsafe_allow_html=True,
                    )
        return

    # ── Admin is logged in ────────────────────────────────────────────────────
    st.markdown("""
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:24px;flex-wrap:wrap">
        <div style="font-family:'Caveat',cursive;font-size:28px;font-weight:700;color:#2d2a1e">
            Admin Dashboard
        </div>
        <span class="sticker sticker-green">Logged in</span>
    </div>
    """, unsafe_allow_html=True)

    admin_left, admin_right = st.columns([1, 1.4], gap="large")

    # ═══════════════════════════════════════════════════════════════════════════
    #  LEFT PANEL — Create event + dropdown event selector
    # ═══════════════════════════════════════════════════════════════════════════
    with admin_left:

        # ── Create new event ──────────────────────────────────────────────────
        st.markdown("""
        <div class="section-label">📁 Create New Event</div>
        """, unsafe_allow_html=True)

        if "new_event_name_val" not in st.session_state:
            st.session_state["new_event_name_val"] = ""

        def _on_event_name_change():
            st.session_state["new_event_name_val"] = st.session_state["new_event_name"]

        new_event_name = st.text_input(
            "Event name",
            placeholder="e.g. Annual Day 2025, Wedding – June 14…",
            key="new_event_name",
            label_visibility="collapsed",
            on_change=_on_event_name_change,
        )

        # Button enabled as soon as any character is typed (no Enter needed)
        create_disabled = len(st.session_state["new_event_name_val"].strip()) == 0

        if st.button(
            "✚  Create Event Folder",
            key="create_event_btn",
            disabled=create_disabled,
        ):
            safe_name = sanitize_folder_name(st.session_state.get("new_event_name", ""))
            new_folder = EVENTS_ROOT_DIR / safe_name
            if new_folder.exists():
                st.markdown(
                    '<div class="pill-warn">An event with this name already exists.</div>',
                    unsafe_allow_html=True,
                )
            else:
                new_folder.mkdir(parents=True, exist_ok=True)
                st.session_state.admin_active_event = str(new_folder)
                st.markdown(
                    f'<div class="pill-ok">Event "<strong>{safe_name}</strong>" created and selected!</div>',
                    unsafe_allow_html=True,
                )
                st.rerun()

        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

        # ── Divider ───────────────────────────────────────────────────────────
        st.markdown(
            "<hr style='border:none;border-top:2px dashed #e8e0cc;margin:4px 0 20px'>",
            unsafe_allow_html=True,
        )

        # ── Fix 3: Event selector as dropdown ─────────────────────────────────
        events = get_event_folders()

        st.markdown(f"""
        <div class="section-label">
            📂 Your Events
            <span class="sticker sticker-gray" style="font-size:11px;margin-left:4px">
                {len(events)}
            </span>
        </div>
        """, unsafe_allow_html=True)

        if not events:
            st.markdown("""
            <div style="background:#fff4e0;border:2px solid #ffc87a;border-radius:14px;
                        padding:20px;text-align:center;margin-bottom:16px">
                <div style="font-size:28px;margin-bottom:6px">🗂️</div>
                <div style="font-size:13px;color:#b35a00;font-weight:700">
                    No events yet. Create one above!
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Dropdown to pick which event to manage
            event_names = [f.name for f in events]
            current_active = st.session_state.get("admin_active_event")
            current_active_name = Path(current_active).name if current_active else None

            # Determine default index for dropdown
            default_idx = 0
            if current_active_name and current_active_name in event_names:
                default_idx = event_names.index(current_active_name)

            st.markdown("""
            <div style="font-size:12px;font-weight:700;color:#7a6f55;margin-bottom:4px;
                        display:flex;align-items:center;gap:6px">
                <span>📋</span> Select event to manage
            </div>
            """, unsafe_allow_html=True)

            chosen_event_name = st.selectbox(
                "Select event to manage",
                options=event_names,
                index=default_idx,
                key="admin_event_dropdown",
                label_visibility="collapsed",
            )

            # Sync active event from dropdown (no rerun — avoids swallowing button clicks)
            chosen_folder = EVENTS_ROOT_DIR / chosen_event_name
            st.session_state.admin_active_event = str(chosen_folder)

            # Info pill for selected event
            img_count = count_images_in_folder(chosen_folder)
            is_active = str(chosen_folder) == st.session_state.get("admin_active_event", "")
            st.markdown(f"""
            <div style="display:flex;align-items:center;justify-content:space-between;
                        background:#f0fdf4;border:2px solid #9de0af;border-radius:12px;
                        padding:10px 14px;margin:10px 0 14px">
                <div style="display:flex;align-items:center;gap:8px">
                    <span style="font-size:16px">{"✅" if is_active else "📁"}</span>
                    <div>
                        <div style="font-size:13px;font-weight:800;color:#1e7a3a;
                                    overflow:hidden;text-overflow:ellipsis;white-space:nowrap;
                                    max-width:160px" title="{chosen_event_name}">
                            {chosen_event_name}
                        </div>
                        <div style="font-size:11px;color:#4a9a5a;font-weight:600">
                            {img_count} photo{"s" if img_count != 1 else ""}
                        </div>
                    </div>
                </div>
                <span class="sticker sticker-green" style="font-size:10px">Managing</span>
            </div>
            """, unsafe_allow_html=True)

            # Delete button for selected event — compact, right-aligned
            st.markdown('<div class="admin-del-btn">', unsafe_allow_html=True)
            if st.button(f'🗑  Delete  "{chosen_event_name}"', key=f"del_{chosen_event_name}"):
                shutil.rmtree(chosen_folder, ignore_errors=True)
                clear_embedding_cache()
                st.session_state.admin_active_event = None
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

        # Logout
        st.markdown('<div class="secondary-btn">', unsafe_allow_html=True)
        if st.button("🚪  Log out", key="logout_btn"):
            st.session_state.admin_logged_in = False
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        # Footer stats
        events_all = get_event_folders()
        total_photos_all = sum(count_images_in_folder(e) for e in events_all)
        st.markdown(f"""
        <div style="margin-top:20px;padding:12px 16px;background:#fff;border:2px solid #e8e0cc;
                    border-radius:12px;font-size:12px;color:#a09070;line-height:1.9">
            <div><strong>Platform:</strong> {sys.platform}</div>
            <div><strong>Total events:</strong> {len(events_all)}</div>
            <div><strong>Total photos:</strong> {total_photos_all}</div>
            <div><strong>Cache:</strong> embeddings_cache/</div>
        </div>
        """, unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════════════════
    #  RIGHT PANEL — Upload & manage photos for selected event
    # ═══════════════════════════════════════════════════════════════════════════
    with admin_right:
        active_event_path = st.session_state.get("admin_active_event")

        if not active_event_path or not Path(active_event_path).exists():
            st.markdown("""
            <div class="fun-card" style="text-align:center;padding:56px 32px">
                <div style="font-size:56px;margin-bottom:14px">👈</div>
                <div style="font-family:'Caveat',cursive;font-size:26px;font-weight:700;
                            color:#2d2a1e;margin-bottom:10px">Select an event to manage</div>
                <div style="font-size:14px;color:#7a6f55;max-width:340px;margin:0 auto">
                    Create a new event or pick one from the dropdown on the left
                    to upload and manage photos here.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            active_folder = Path(active_event_path)
            img_count = count_images_in_folder(active_folder)

            # Header
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:10px;margin-bottom:18px;flex-wrap:wrap">
                <span style="font-size:20px">📤</span>
                <div style="font-family:'Caveat',cursive;font-size:22px;font-weight:700;color:#2d2a1e">
                    Uploading to:
                </div>
                <span class="sticker sticker-blue" style="font-size:13px;max-width:260px;
                    overflow:hidden;text-overflow:ellipsis;white-space:nowrap">
                    {active_folder.name}
                </span>
                <span class="sticker sticker-gray">{img_count} photo{"s" if img_count != 1 else ""}</span>
            </div>
            """, unsafe_allow_html=True)

            # ── File uploader ─────────────────────────────────────────────────
            event_files = st.file_uploader(
                "Select photos to upload",
                type=["jpg", "jpeg", "png", "webp", "bmp"],
                accept_multiple_files=True,
                key=f"uploader_{active_folder.name}",
                label_visibility="collapsed",
            )

            if event_files:
                st.caption(f"{len(event_files)} file{'s' if len(event_files) != 1 else ''} selected — ready to upload")

            st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

            # ── Fix 2: Compact, aligned action buttons in one clean row ───────
            col_up, col_clr, col_cache = st.columns([1.2, 1, 1], gap="small")

            with col_up:
                st.markdown('<div class="upload-btn">', unsafe_allow_html=True)
                upload_clicked = st.button(
                    "⬆  Upload Photos",
                    key=f"upload_btn_{active_folder.name}",
                    disabled=not event_files,
                )
                st.markdown('</div>', unsafe_allow_html=True)

            with col_clr:
                st.markdown('<div class="clear-photos-btn">', unsafe_allow_html=True)
                clear_clicked = st.button(
                    "🗑  Clear All",
                    key=f"clear_photos_{active_folder.name}",
                    disabled=img_count == 0,
                )
                st.markdown('</div>', unsafe_allow_html=True)

            with col_cache:
                st.markdown('<div class="cache-btn">', unsafe_allow_html=True)
                cache_clicked = st.button(
                    "🔄  Clear Cache",
                    key=f"clear_cache_{active_folder.name}",
                )
                st.markdown('</div>', unsafe_allow_html=True)

            # ── Handle button actions ─────────────────────────────────────────
            if upload_clicked and event_files:
                prog = st.progress(0, text="Saving…")
                for i, f in enumerate(event_files):
                    dest = active_folder / f.name
                    dest.write_bytes(f.read())
                    prog.progress(
                        (i + 1) / len(event_files),
                        text=f"Saved {i + 1}/{len(event_files)}",
                    )
                prog.empty()
                clear_embedding_cache()
                st.session_state.results = None
                st.rerun()

            if clear_clicked:
                for img in active_folder.iterdir():
                    if img.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
                        img.unlink(missing_ok=True)
                clear_embedding_cache()
                st.rerun()

            if cache_clicked:
                clear_embedding_cache()
                st.markdown('<div class="pill-ok" style="margin-top:6px">Cache cleared successfully.</div>', unsafe_allow_html=True)

            # ── Status pill ───────────────────────────────────────────────────
            refreshed_count = count_images_in_folder(active_folder)
            pill_class = "pill-ok" if refreshed_count > 0 else "pill-info"
            pill_msg = (
                f"{refreshed_count} photo{'s' if refreshed_count != 1 else ''} in this event — users can search!"
                if refreshed_count > 0
                else "No photos yet — upload some above."
            )
            st.markdown(f'<div class="{pill_class}" style="margin-top:10px">{pill_msg}</div>', unsafe_allow_html=True)

            st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

            # ── Photo preview grid ────────────────────────────────────────────
            st.markdown(f"""
            <div class="section-label">
                📸 Photos in "{active_folder.name}"
            </div>
            """, unsafe_allow_html=True)

            if refreshed_count > 0:
                imgs = sorted([
                    p for p in active_folder.iterdir()
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
                ])
                pcols = st.columns(4)
                for i, img_path in enumerate(imgs[:20]):
                    with pcols[i % 4]:
                        st.image(
                            Image.open(img_path),
                            use_container_width=True,
                            caption=img_path.name[:14],
                        )
                if len(imgs) > 20:
                    st.caption(f"… and {len(imgs) - 20} more photos")
            else:
                st.markdown("""
                <div class="fun-card" style="text-align:center;padding:36px">
                    <div style="font-size:40px;margin-bottom:8px">🙏</div>
                    <div style="font-size:15px;color:#7a6f55">No photos uploaded yet.</div>
                </div>
                """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  APP ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
render_header()

tab_user, tab_admin = st.tabs(["  Find My Photos  ", "  Admin  "])

with tab_user:
    render_user_tab()

with tab_admin:
    render_admin_tab()