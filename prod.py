"""
app.py — SehaBot Professional Testing Dashboard.

A modern, multi-module Streamlit interface for testing the full SehaBot suite:
  • SehaBot Chat        — Stateful, conversational, multimodal (/chat)
  • Prescription Analyzer — Stateless OCR, structured JSON (/analyze/prescription)
  • Medicine Box Analyzer  — Stateless packaging parser, structured JSON (/analyze/medicine-box)
"""

import uuid
import json
import requests
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# PRODUCTION CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
AUTH_TOKEN = "d1414a76-13c4-4267-9b96-12b0b62425f5"
AUTH_HEADERS = {"Authorization": f"Bearer {AUTH_TOKEN}"}
PROD_BASE_URL = "https://8080-dep-01kp3cxxfzs0chgwd41w54zmem-d.cloudspaces.litng.ai"

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be the very first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SehaBot Dashboard",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS — Medical-grade dark-blue theme
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Core palette ─────────────────────────────────────────── */
:root {
    --blue-primary:   #1A73E8;
    --blue-dark:      #0D47A1;
    --blue-light:     #E8F0FE;
    --teal-accent:    #00ACC1;
    --green-ok:       #1E8B4F;
    --red-error:      #C62828;
    --bg-dark:        #0F1B2D;
    --bg-card:        #162032;
    --bg-card-hover:  #1E2D44;
    --text-primary:   #E8EDF5;
    --text-muted:     #8A9BB5;
    --border:         #253650;
}

/* ── App background ──────────────────────────────────────── */
.stApp { background-color: var(--bg-dark); color: var(--text-primary); }

/* ── Hide Streamlit chrome ───────────────────────────────── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Sidebar ─────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #0D1B2E 0%, #122540 100%);
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: var(--text-primary) !important; }

/* ── Nav radio buttons → pill style ─────────────────────── */
div[role="radiogroup"] label {
    display: block;
    padding: 10px 16px;
    margin: 4px 0;
    border-radius: 10px;
    cursor: pointer;
    font-weight: 500;
    font-size: 15px;
    transition: background 0.18s, color 0.18s;
    border: 1px solid transparent;
}
div[role="radiogroup"] label:hover {
    background: #1E3A5F;
    border-color: var(--blue-primary);
}
div[role="radiogroup"] label[data-testid~="selected"],
div[role="radiogroup"] label:has(input:checked) {
    background: linear-gradient(90deg, #1A73E8 0%, #00ACC1 100%);
    color: #fff !important;
    border-color: transparent;
}

/* ── Section heading strip ───────────────────────────────── */
.section-header {
    background: linear-gradient(90deg, var(--blue-dark) 0%, #0D2A4F 100%);
    border-left: 4px solid var(--teal-accent);
    padding: 12px 20px;
    border-radius: 0 10px 10px 0;
    margin-bottom: 24px;
    font-size: 22px;
    font-weight: 700;
    letter-spacing: .3px;
}

/* ── Result cards ────────────────────────────────────────── */
.result-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 20px 24px;
    margin-bottom: 12px;
    transition: border-color .2s;
}
.result-card:hover { border-color: var(--blue-primary); }
.result-card h4 { color: var(--teal-accent); margin: 0 0 10px; font-size: 13px; text-transform: uppercase; letter-spacing: 1px; }
.result-card p  { margin: 0; font-size: 16px; color: var(--text-primary); }

/* ── Medication table rows ───────────────────────────────── */
.med-row {
    display: grid;
    grid-template-columns: 2fr 1fr 1.5fr 1fr 2fr;
    gap: 12px;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 8px;
    align-items: center;
    font-size: 14px;
}
.med-row .label { font-size: 11px; color: var(--text-muted); margin-bottom: 3px; }
.med-row .value { color: var(--text-primary); font-weight: 500; }

/* ── Confidence badge ────────────────────────────────────── */
.badge-high   { background:#1E8B4F22; color:#34C77B; border:1px solid #1E8B4F; border-radius:6px; padding:2px 10px; font-size:13px; font-weight:700; }
.badge-medium { background:#F9A82522; color:#FFC107; border:1px solid #F9A825; border-radius:6px; padding:2px 10px; font-size:13px; font-weight:700; }
.badge-low    { background:#C6282822; color:#EF5350; border:1px solid #C62828; border-radius:6px; padding:2px 10px; font-size:13px; font-weight:700; }

/* ── Divider ─────────────────────────────────────────────── */
hr { border-color: var(--border) !important; margin: 20px 0; }

/* ── Input fields ────────────────────────────────────────── */
.stTextInput input, .stTextArea textarea {
    background: var(--bg-card) !important;
    border-color: var(--border) !important;
    color: var(--text-primary) !important;
    border-radius: 8px !important;
}

/* ── Buttons ─────────────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(90deg, var(--blue-primary) 0%, var(--teal-accent) 100%);
    color: #fff;
    border: none;
    border-radius: 10px;
    font-weight: 600;
    font-size: 15px;
    padding: 10px 0;
    transition: opacity .2s, transform .1s;
}
.stButton > button:hover  { opacity: .88; transform: translateY(-1px); }
.stButton > button:active { transform: translateY(0); }

/* ── Upload zone ─────────────────────────────────────────── */
[data-testid="stFileUploader"] {
    border: 2px dashed var(--border) !important;
    border-radius: 14px !important;
    background: var(--bg-card) !important;
}

/* ── Chat messages ───────────────────────────────────────── */
[data-testid="stChatMessage"] {
    border-radius: 14px;
    padding: 4px 8px;
    margin-bottom: 4px;
}

/* ── Status box ──────────────────────────────────────────── */
[data-testid="stStatus"] {
    background: var(--bg-card) !important;
    border-color: var(--border) !important;
    border-radius: 12px !important;
}

/* ── Tabs ────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-card);
    border-radius: 10px 10px 0 0;
    padding: 4px 8px 0;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0;
    padding: 8px 20px;
    font-weight: 600;
    color: var(--text-muted) !important;
}
.stTabs [aria-selected="true"] {
    background: var(--blue-primary) !important;
    color: #fff !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE INITIALISATION
# ─────────────────────────────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "messages":   [],
        "summary":    "",
        "thread_id":  str(uuid.uuid4()),
        "active_page": "🏥 SehaBot Chat",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ─────────────────────────────────────────────────────────────────────────────
# BACKEND CLIENT
# ─────────────────────────────────────────────────────────────────────────────
class SehaBackendClient:
    """
    Single HTTP client for all three SehaBot endpoints.
    Handles multipart payloads, SSE streaming, and one-shot JSON calls.
    """

    def __init__(self, base_url: str, user_id: str):
        self.base_url = base_url.rstrip("/")
        self.user_id  = user_id

    # ── /chat — streaming SSE ─────────────────────────────────────────────────
    def stream_chat(self, query: str, summary: str, thread_id: str, image_file=None):
        """
        Yields parsed JSON dicts from the /chat SSE stream.
        Caller is responsible for updating UI from each yielded event.
        """
        data_payload = {
            "query":     query,
            "user_id":   self.user_id,
            "summary":   summary,
            "thread_id": thread_id,
        }
        files = {}
        if image_file is not None:
            image_file.seek(0)
            files["image"] = (image_file.name, image_file, image_file.type)

        with requests.post(
            f"{self.base_url}/chat",
            data=data_payload,
            files=files if files else None,
            headers=AUTH_HEADERS,
            stream=True,
            timeout=180,
        ) as resp:
            resp.raise_for_status()
            for raw_line in resp.iter_lines():
                if raw_line:
                    try:
                        yield json.loads(raw_line.decode("utf-8"))
                    except json.JSONDecodeError:
                        pass

    # ── /analyze/* — one-shot JSON ────────────────────────────────────────────
    def analyze_image(self, endpoint: str, image_file) -> dict:
        """
        Posts an image to a stateless analysis endpoint and returns the JSON dict.
        `endpoint` should be 'prescription' or 'medicine-box'.
        """
        image_file.seek(0)
        resp = requests.post(
            f"{self.base_url}/analyze/{endpoint}",
            files={"image": (image_file.name, image_file, image_file.type)},
            headers=AUTH_HEADERS,
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 16px 0 8px;'>
        <span style='font-size:48px;'>🩺</span>
        <h2 style='margin:4px 0 2px; font-size:20px; color:#E8F0FE;'>SehaBot</h2>
        <p style='color:#8A9BB5; font-size:12px; margin:0;'>Medical AI Testing Dashboard</p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    page = st.radio(
        "Navigation",
        options=["🏥 SehaBot Chat", "📜 Prescription Analyzer", "📦 Medicine Box Analyzer"],
        label_visibility="collapsed",
    )
    st.session_state.active_page = page

    st.divider()

    st.markdown("**⚙️ Global Settings**")
    backend_url = st.text_input("Backend URL", value=PROD_BASE_URL, key="backend_url")
    user_id     = st.text_input("User ID",     value="demo_patient_001",       key="user_id")

    st.divider()

    # Chat controls — only visible on chat page
    if page == "🏥 SehaBot Chat":
        st.markdown("**💬 Chat Controls**")
        if st.button("🗑️ Clear Conversation", use_container_width=True):
            st.session_state.messages  = []
            st.session_state.summary   = ""
            st.session_state.thread_id = str(uuid.uuid4())
            st.rerun()

        if st.session_state.summary:
            with st.expander("📋 Session Summary", expanded=False):
                st.caption(st.session_state.summary)

    st.markdown(
        "<p style='color:#4A6080; font-size:11px; text-align:center; margin-top:20px;'>"
        "SehaBot Testing Suite v2.0</p>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# SHARED HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _make_client() -> SehaBackendClient:
    return SehaBackendClient(
        base_url=st.session_state.get("backend_url", PROD_BASE_URL),
        user_id=st.session_state.get("user_id", "demo_patient_001"),
    )


def _error_message(e: Exception) -> str:
    if isinstance(e, requests.ConnectionError):
        return "❌ **Cannot connect to the backend.** Make sure the FastAPI server is running:\n\n`uvicorn server:app --host 0.0.0.0 --port 8000`"
    if isinstance(e, requests.Timeout):
        return "⏱️ **Request timed out.** The model is taking too long — try a simpler query."
    if isinstance(e, requests.HTTPError):
        code = e.response.status_code if e.response is not None else "?"
        if code == 429:
            return "⏳ **Rate limit exceeded.** Please wait a minute before trying again."
        if code == 503:
            return "🔧 **Service unavailable.** The server graph is still compiling — retry in a moment."
        if code == 422:
            detail = ""
            try: detail = e.response.json().get("detail", "")
            except Exception: pass
            return f"⚠️ **Validation error:** {detail}"
        if code == 400:
            detail = ""
            try: detail = e.response.json().get("detail", "")
            except Exception: pass
            return f"🖼️ **Bad image:** {detail}"
        return f"❌ **HTTP {code} error** from server."
    return f"❌ **Unexpected error:** {e}"


def _confidence_badge(level: str) -> str:
    level = (level or "").upper()
    cls = {"HIGH": "badge-high", "MEDIUM": "badge-medium", "LOW": "badge-low"}.get(level, "badge-medium")
    return f'<span class="{cls}">{level}</span>'


def _tag_list(items: list[str], color="#00ACC1") -> str:
    if not items:
        return "<em style='color:#4A6080'>None listed</em>"
    tags = "".join(
        f'<span style="background:{color}22;color:{color};border:1px solid {color};'
        f'border-radius:6px;padding:2px 10px;margin:2px;display:inline-block;font-size:13px;">'
        f'{item}</span>'
        for item in items
    )
    return f'<div style="display:flex;flex-wrap:wrap;gap:4px;margin-top:4px;">{tags}</div>'


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 1 — SEHABOT CHAT
# ─────────────────────────────────────────────────────────────────────────────

def render_chat():
    st.markdown(
        '<div class="section-header">🏥 SehaBot — Intelligent Medical Chat</div>',
        unsafe_allow_html=True,
    )

    # ── Replay history ───────────────────────────────────────────────────────
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="🧑‍⚕️" if msg["role"] == "assistant" else "👤"):
            st.markdown(msg["content"])
            if msg.get("image_caption"):
                st.caption(f"🖼️ {msg['image_caption']}")

    # ── Image attachment (above chat input) ──────────────────────────────────
    with st.expander("📎 Attach an Image (X-ray, Prescription, Drug Box…)", expanded=False):
        chat_image = st.file_uploader(
            "Upload image",
            type=["jpg", "jpeg", "png"],
            key="chat_image_uploader",
            label_visibility="collapsed",
        )
        if chat_image:
            col_prev, col_info = st.columns([1, 2])
            with col_prev:
                st.image(chat_image, width=None)
            with col_info:
                st.success(f"✅ **{chat_image.name}** ready to send with your next message.")
                st.caption(f"Size: {len(chat_image.getvalue()) / 1024:.1f} KB · Type: {chat_image.type}")

    # ── Chat input ───────────────────────────────────────────────────────────
    prompt = st.chat_input("Describe your symptoms, ask a question, or type a command…")

    if not prompt:
        return

    # --- Persist user message ---
    caption = f"Image attached: {chat_image.name}" if chat_image else None
    st.session_state.messages.append({
        "role": "user",
        "content": prompt,
        "image_caption": caption,
    })

    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)
        if chat_image:
            st.caption(f"🖼️ {caption}")

    # --- Stream assistant response ---
    with st.chat_message("assistant", avatar="🧑‍⚕️"):
        status_box   = st.status("🔄 Connecting to SehaBot…", expanded=True)
        resp_holder  = st.empty()
        full_response = ""

        try:
            client = _make_client()
            for event in client.stream_chat(
                query=prompt,
                summary=st.session_state.summary,
                thread_id=st.session_state.thread_id,
                image_file=chat_image,
            ):
                etype = event.get("type")

                if etype == "status":
                    node_msg = event.get("content", "Processing…")
                    status_box.update(label=f"⚙️ {node_msg}")
                    status_box.write(f"› {node_msg}")

                elif etype == "token":
                    full_response += event.get("content", "")
                    resp_holder.markdown(full_response + " ▌")

                elif etype == "final":
                    new_summary = event.get("summary", "")
                    if new_summary:
                        st.session_state.summary = new_summary

                elif etype == "error":
                    status_box.update(label="❌ Agent error", state="error")
                    st.error(event.get("content", "Unknown error from agent."))
                    return

            status_box.update(label="✅ Response complete", state="complete", expanded=False)
            resp_holder.markdown(full_response)

            if not full_response.strip():
                st.warning("⚠️ The agent returned an empty response. Check the backend logs.")
            else:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response,
                })

        except Exception as e:
            status_box.update(label="❌ Error", state="error")
            st.error(_error_message(e))


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 2 — PRESCRIPTION ANALYZER
# ─────────────────────────────────────────────────────────────────────────────

def _render_prescription_results(data: dict):
    """Renders structured PrescriptionResponse data as formatted UI cards."""

    # ── Header row ────────────────────────────────────────────────────────────
    conf_badge = _confidence_badge(data.get("confidence", ""))
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:12px;margin-bottom:20px;">'
        f'<span style="font-size:26px;font-weight:700;">📜 Prescription Results</span>'
        f'{conf_badge}'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Patient / Doctor meta ─────────────────────────────────────────────────
    meta_col1, meta_col2, meta_col3 = st.columns(3)
    with meta_col1:
        st.markdown(
            f'<div class="result-card"><h4>👤 Patient Name</h4>'
            f'<p>{data.get("patient_name") or "Not found"}</p></div>',
            unsafe_allow_html=True,
        )
    with meta_col2:
        st.markdown(
            f'<div class="result-card"><h4>👨‍⚕️ Doctor Name</h4>'
            f'<p>{data.get("doctor_name") or "Not found"}</p></div>',
            unsafe_allow_html=True,
        )
    with meta_col3:
        st.markdown(
            f'<div class="result-card"><h4>📅 Date</h4>'
            f'<p>{data.get("patient_date") or "Not found"}</p></div>',
            unsafe_allow_html=True,
        )

    # ── Medications table ─────────────────────────────────────────────────────
    meds = data.get("medications", [])
    st.markdown(f"### 💊 Medications  •  `{len(meds)} found`")

    if not meds:
        st.info("No medications were extracted from this prescription.")
    else:
        # Column headers
        h1, h2, h3, h4, h5 = st.columns([2, 1, 1.5, 1, 2])
        for col, label in zip([h1, h2, h3, h4, h5],
                               ["💊 Medication", "📏 Dosage", "🔄 Frequency", "📆 Duration", "📝 Notes"]):
            col.markdown(f"**{label}**")
        st.divider()

        for med in meds:
            c1, c2, c3, c4, c5 = st.columns([2, 1, 1.5, 1, 2])
            c1.markdown(f"**{med.get('name', '—')}**")
            c2.markdown(med.get("dosage", "—"))
            c3.markdown(med.get("frequency", "—"))
            c4.markdown(med.get("duration") or "—")
            c5.markdown(med.get("notes") or "—")
            st.divider()

    # ── General notes ─────────────────────────────────────────────────────────
    if data.get("general_notes"):
        st.markdown(
            f'<div class="result-card"><h4>📋 General Doctor Notes</h4>'
            f'<p>{data["general_notes"]}</p></div>',
            unsafe_allow_html=True,
        )

    # ── Unreadable parts ──────────────────────────────────────────────────────
    if data.get("unreadable_parts"):
        st.warning(f"⚠️ **Unreadable sections:** {data['unreadable_parts']}")

    # ── Raw JSON (collapsible) ────────────────────────────────────────────────
    with st.expander("🔍 Raw JSON Response", expanded=False):
        st.json(data)


def render_prescription_analyzer():
    st.markdown(
        '<div class="section-header">📜 Prescription Analyzer — OCR + Structured Extraction</div>',
        unsafe_allow_html=True,
    )

    left_col, right_col = st.columns([1, 1.4], gap="large")

    with left_col:
        st.markdown("#### 📤 Upload Prescription Image")
        rx_image = st.file_uploader(
            "Drag & drop or click to browse",
            type=["jpg", "jpeg", "png"],
            key="rx_uploader",
            label_visibility="collapsed",
        )

        if rx_image:
            st.image(rx_image, caption=f"📎 {rx_image.name}", width=None)
            st.caption(f"Size: {len(rx_image.getvalue()) / 1024:.1f} KB")
            st.markdown("")  # spacer

        analyze_btn = st.button(
            "🔬 Analyze Prescription",
            disabled=(rx_image is None),
            use_container_width=True,
            key="rx_analyze_btn",
        )

        if rx_image is None:
            st.markdown(
                '<div style="text-align:center;color:#4A6080;font-size:13px;margin-top:24px;">'
                '📋 Upload a prescription image to begin analysis</div>',
                unsafe_allow_html=True,
            )

    with right_col:
        st.markdown("#### 📊 Analysis Results")

        if analyze_btn and rx_image:
            with st.spinner("🔬 Extracting prescription data… this may take a few seconds."):
                try:
                    client = _make_client()
                    result = client.analyze_image("prescription", rx_image)
                    _render_prescription_results(result)
                except Exception as e:
                    st.error(_error_message(e))
        else:
            st.markdown(
                '<div style="background:#162032;border:1px dashed #253650;border-radius:14px;'
                'padding:60px 20px;text-align:center;color:#4A6080;">'
                '<span style="font-size:40px;">📋</span><br><br>'
                'Results will appear here after analysis.'
                '</div>',
                unsafe_allow_html=True,
            )


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 3 — MEDICINE BOX ANALYZER
# ─────────────────────────────────────────────────────────────────────────────

def _render_medicine_box_results(data: dict):
    """Renders structured MedicineBoxResponse data as formatted UI cards."""

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown(
        f'<div style="margin-bottom:20px;">'
        f'<span style="font-size:26px;font-weight:700;">📦 {data.get("trade_name", "Medicine Box")}</span>'
        f'<span style="color:#8A9BB5;font-size:16px;margin-left:12px;">'
        f'{data.get("generic_name") or ""}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Key facts row ─────────────────────────────────────────────────────────
    kf1, kf2, kf3 = st.columns(3)
    with kf1:
        st.markdown(
            f'<div class="result-card"><h4>💊 Dosage Form</h4>'
            f'<p>{data.get("dosage_form") or "—"}</p></div>',
            unsafe_allow_html=True,
        )
    with kf2:
        st.markdown(
            f'<div class="result-card"><h4>⚗️ Concentration</h4>'
            f'<p>{data.get("concentration") or "—"}</p></div>',
            unsafe_allow_html=True,
        )
    with kf3:
        st.markdown(
            f'<div class="result-card"><h4>🏭 Manufacturer</h4>'
            f'<p>{data.get("manufacturer") or "—"}</p></div>',
            unsafe_allow_html=True,
        )

    # ── Active ingredients ────────────────────────────────────────────────────
    st.markdown("#### ⚗️ Active Ingredients")
    st.markdown(_tag_list(data.get("active_ingredients", []), color="#00ACC1"), unsafe_allow_html=True)
    st.markdown("")

    # ── Indications ───────────────────────────────────────────────────────────
    st.markdown("#### ✅ Indications (Uses)")
    inds = data.get("indications", [])
    if inds:
        for item in inds:
            st.markdown(f"- {item}")
    else:
        st.markdown("_None listed_")

    # ── Contraindications ─────────────────────────────────────────────────────
    st.markdown("#### ⛔ Contraindications")
    contra = data.get("contraindications", [])
    if contra:
        for item in contra:
            st.markdown(f"- {item}")
    else:
        st.markdown("_None listed_")

    # ── Storage / Expiry ──────────────────────────────────────────────────────
    sc1, sc2 = st.columns(2)
    with sc1:
        if data.get("storage_conditions"):
            st.markdown(
                f'<div class="result-card"><h4>🌡️ Storage Conditions</h4>'
                f'<p>{data["storage_conditions"]}</p></div>',
                unsafe_allow_html=True,
            )
    with sc2:
        if data.get("expiry_date"):
            st.markdown(
                f'<div class="result-card"><h4>⏰ Expiry Date</h4>'
                f'<p>{data["expiry_date"]}</p></div>',
                unsafe_allow_html=True,
            )

    # ── Raw JSON (collapsible) ────────────────────────────────────────────────
    with st.expander("🔍 Raw JSON Response", expanded=False):
        st.json(data)


def render_medicine_box_analyzer():
    st.markdown(
        '<div class="section-header">📦 Medicine Box Analyzer — Packaging Intelligence</div>',
        unsafe_allow_html=True,
    )

    left_col, right_col = st.columns([1, 1.4], gap="large")

    with left_col:
        st.markdown("#### 📤 Upload Medicine Box Image")
        box_image = st.file_uploader(
            "Drag & drop or click to browse",
            type=["jpg", "jpeg", "png"],
            key="box_uploader",
            label_visibility="collapsed",
        )

        if box_image:
            st.image(box_image, caption=f"📎 {box_image.name}", width=None)
            st.caption(f"Size: {len(box_image.getvalue()) / 1024:.1f} KB")
            st.markdown("")

        analyze_btn = st.button(
            "🔬 Analyze Medicine Box",
            disabled=(box_image is None),
            use_container_width=True,
            key="box_analyze_btn",
        )

        if box_image is None:
            st.markdown(
                '<div style="text-align:center;color:#4A6080;font-size:13px;margin-top:24px;">'
                '📦 Upload a medicine box photo to begin analysis</div>',
                unsafe_allow_html=True,
            )

    with right_col:
        st.markdown("#### 📊 Analysis Results")

        if analyze_btn and box_image:
            with st.spinner("📦 Reading medicine box packaging… this may take a few seconds."):
                try:
                    client = _make_client()
                    result = client.analyze_image("medicine-box", box_image)
                    _render_medicine_box_results(result)
                except Exception as e:
                    st.error(_error_message(e))
        else:
            st.markdown(
                '<div style="background:#162032;border:1px dashed #253650;border-radius:14px;'
                'padding:60px 20px;text-align:center;color:#4A6080;">'
                '<span style="font-size:40px;">📦</span><br><br>'
                'Results will appear here after analysis.'
                '</div>',
                unsafe_allow_html=True,
            )


# ─────────────────────────────────────────────────────────────────────────────
# ROUTER — Render the active page
# ─────────────────────────────────────────────────────────────────────────────
active = st.session_state.active_page

if active == "🏥 SehaBot Chat":
    render_chat()
elif active == "📜 Prescription Analyzer":
    render_prescription_analyzer()
elif active == "📦 Medicine Box Analyzer":
    render_medicine_box_analyzer()