"""Interactive WISDM HAR dashboard for course PT1.
Run: streamlit run dashboard.py
"""
from __future__ import annotations

from pathlib import Path
import hashlib
import os
import tempfile

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from pt1_utils import ACTIVITY_ORDER, AXES, load_wisdm_data, make_windows, outlier_audit

st.set_page_config(
    page_title="WISDM | Human Activity Recognition",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------
# Visual theme: Minimal Academic (white + navy + blue/teal accents)
# ---------------------------------------------------------------------
st.markdown(
    """
    <style>
    :root {
        --wisdm-navy: #142B4A;
        --wisdm-blue: #2878B8;
        --wisdm-teal: #22A6A1;
        --wisdm-ink: #243447;
        --wisdm-muted: #66788A;
        --wisdm-border: #E4EAF1;
        --wisdm-surface: #FFFFFF;
        --wisdm-page: #F5F8FC;
    }
    [data-testid="stAppViewContainer"] {
        background: var(--wisdm-page);
        color: var(--wisdm-ink);
    }
    [data-testid="stHeader"] { background: rgba(245, 248, 252, 0.92); }
    .block-container {
        max-width: 1440px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    h1, h2, h3 {
        color: var(--wisdm-navy) !important;
        letter-spacing: -0.025em;
    }
    h1 { font-weight: 800 !important; padding-bottom: 0.25rem; }
    h2, h3 { font-weight: 700 !important; }
    [data-testid="stCaptionContainer"] { color: var(--wisdm-muted); }

    [data-testid="stSidebar"] {
        background: var(--wisdm-navy);
        border-right: 1px solid rgba(255,255,255,0.08);
    }
    [data-testid="stSidebar"] * { color: #F4F8FC; }
    [data-testid="stSidebar"] [data-testid="stFileUploader"] section {
        background: rgba(255,255,255,0.09);
        border: 1px dashed rgba(255,255,255,0.38);
        border-radius: 12px;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploader"] button {
        color: var(--wisdm-navy) !important;
    }

    [data-testid="stMetric"] {
        background: var(--wisdm-surface);
        border: 1px solid var(--wisdm-border);
        border-radius: 16px;
        padding: 1rem 1.15rem;
        box-shadow: 0 5px 18px rgba(20, 43, 74, 0.055);
    }
    [data-testid="stMetricLabel"] {
        color: var(--wisdm-muted) !important;
        font-weight: 600;
    }
    [data-testid="stMetricValue"] {
        color: var(--wisdm-navy) !important;
        font-weight: 800;
    }
    [data-testid="stExpander"] {
        background: var(--wisdm-surface);
        border: 1px solid var(--wisdm-border);
        border-radius: 14px;
    }
    [data-testid="stDataFrame"], [data-testid="stTable"] {
        border: 1px solid var(--wisdm-border);
        border-radius: 12px;
        overflow: hidden;
    }
    [data-testid="stPlotlyChart"] {
        background: var(--wisdm-surface);
        border: 1px solid var(--wisdm-border);
        border-radius: 16px;
        padding: 0.45rem;
        box-shadow: 0 4px 16px rgba(20, 43, 74, 0.045);
    }
    div.stButton > button, div[data-testid="stDownloadButton"] button {
        background: var(--wisdm-blue);
        color: white;
        border: 0;
        border-radius: 10px;
        font-weight: 650;
    }
    div.stButton > button:hover, div[data-testid="stDownloadButton"] button:hover {
        background: var(--wisdm-navy);
        color: white;
        border: 0;
    }
    hr { border-color: var(--wisdm-border); margin: 1.5rem 0; }
    @media (max-width: 800px) {
        .block-container {
            padding-top: 1.1rem;
            padding-left: 1rem;
            padding-right: 1rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# High-contrast, color-blind-conscious palette: each plotted signal gets a
# clearly distinguishable color, including all 9 traces in activity comparisons.
WISDM_COLORS = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#D55E00",  # vermilion
    "#CC79A7",  # pink
    "#009E73",  # green
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#7A3E9D",  # purple
    "#333333",  # charcoal
]

def style_chart(fig):
    fig.update_layout(
        template="plotly_white",
        colorway=WISDM_COLORS,
        font={"family": "Arial, sans-serif", "color": "#243447", "size": 12},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hoverlabel={"bgcolor": "#142B4A", "font": {"color": "#FFFFFF", "size": 12}},
        legend={"bgcolor": "rgba(255,255,255,0.75)", "borderwidth": 0},
    )
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor="#DCE4ED")
    fig.update_yaxes(gridcolor="#E8EEF5", zeroline=False, linecolor="#DCE4ED")
    return fig


st.title("داشبورد تعاملی تشخیص فعالیت انسانی | WISDM")
st.caption("PT1 — تحلیل اکتشافی سیگنال‌های شتاب‌سنج، مقایسهٔ افراد و پنجره‌های زمانی")


@st.cache_data(show_spinner="در حال بارگذاری و پاک‌سازی داده‌ها...")
def cached_load(path: str, mtime: float):
    return load_wisdm_data(path)


@st.cache_data(show_spinner="در حال ساخت پنجره‌های زمانی...")
def cached_windows(path: str, mtime: float, window_size: int = 51, step: int = 26):
    frame, audit = load_wisdm_data(path)
    windows, metadata = make_windows(frame, window_size=window_size, step=step)
    return windows, metadata


st.sidebar.header("منبع داده")
default_path = Path(os.environ.get("WISDM_DATA_PATH", "data/WISDM_ar_v1.1_raw.txt"))
uploaded = st.sidebar.file_uploader("فایل خام WISDM (اختیاری)", type=["txt", "csv"])

if uploaded is not None:
    payload = uploaded.getvalue()
    digest = hashlib.sha256(payload).hexdigest()[:16]
    temp_path = Path(tempfile.gettempdir()) / f"wisdm_upload_{digest}.txt"
    if not temp_path.exists() or temp_path.stat().st_size != len(payload):
        temp_path.write_bytes(payload)
    data_path = temp_path
else:
    data_path = default_path

if not data_path.exists():
    st.info(
        "برای شروع، فایل خام مخصوص کلاس را در مسیر `data/WISDM_ar_v1.1_raw.txt` قرار دهید "
        "یا از گزینهٔ بارگذاری در نوار کناری استفاده کنید."
    )
   

mtime = data_path.stat().st_mtime
try:
    data, audit = cached_load(str(data_path), mtime)
    X_windows, meta = cached_windows(str(data_path), mtime, 51, 26)
except Exception as exc:
    st.error(f"خواندن داده ناموفق بود: {exc}")
    st.stop()

if len(meta) == 0:
    st.error("هیچ پنجرهٔ ۵۱ نمونه‌ای ساخته نشد. فایل یا طول قطعات فعالیت را بررسی کنید.")
    st.stop()

# Top-level summary metrics
m1, m2, m3, m4 = st.columns(4)
m1.metric("رکوردهای معتبر", f"{len(data):,}")
m2.metric("شرکت‌کنندگان", f"{data.user_id.nunique():,}")
m3.metric("فعالیت‌های شناسایی‌شده", f"{data.activity.nunique():,}")
m4.metric("پنجره‌های ۵۱ نمونه‌ای", f"{len(meta):,}")


# Quality audit expander
with st.expander("کنترل کیفیت و آمار خلاصهٔ داده", expanded=False):
    st.write("گزارش پاک‌سازی")
    st.json(audit)
    st.write("نامزدهای پرت با قاعدهٔ IQR؛ هیچ‌کدام خودکار حذف نشده‌اند.")
    st.dataframe(outlier_audit(data).round(4), use_container_width=True)

# Dataset counts
left, right = st.columns(2)
with left:
    st.subheader("توزیع رکوردهای فعالیت")
    activity_counts = data.activity.value_counts().reindex(ACTIVITY_ORDER, fill_value=0)
    fig = go.Figure(go.Bar(x=activity_counts.index, y=activity_counts.values))
    fig.update_layout(xaxis_title="فعالیت", yaxis_title="تعداد رکورد", height=380, margin=dict(t=20, b=20))
    style_chart(fig)
    st.plotly_chart(fig, use_container_width=True)
with right:
    st.subheader("رکوردهای هر شرکت‌کننده")
    subject_counts = data.groupby("user_id").size().sort_index()
    fig = go.Figure(go.Bar(x=subject_counts.index.astype(str), y=subject_counts.values))
    fig.update_layout(xaxis_title="شناسهٔ شرکت‌کننده", yaxis_title="تعداد رکورد", height=380,
                      xaxis_tickangle=-45, margin=dict(t=20, b=20))
    style_chart(fig)
    st.plotly_chart(fig, use_container_width=True)

st.divider()
st.header("۱) مشاهدهٔ سیگنال یک فعالیت")
control1, control2, control3 = st.columns([1, 1, 1])
with control1:
    selected_activity = st.selectbox("فعالیت", ACTIVITY_ORDER, index=0)
with control2:
    activity_users = sorted(meta.loc[meta.activity == selected_activity, "user_id"].unique().tolist())
    selected_user = st.selectbox("شرکت‌کننده", ["همه"] + activity_users, index=0)
with control3:
    st.metric("پنجره‌های این فعالیت", f"{int((meta.activity == selected_activity).sum()):,}")

rows = meta.index[meta.activity == selected_activity].tolist()
if selected_user != "همه":
    rows = meta.index[(meta.activity == selected_activity) & (meta.user_id == int(selected_user))].tolist()
if rows:
    window_position = st.slider("شمارهٔ پنجره", min_value=0, max_value=len(rows) - 1, value=0)
    actual_idx = rows[window_position]
    signal = X_windows[actual_idx]
    signal_user = int(meta.loc[actual_idx, "user_id"])
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=["محور X", "محور Y", "محور Z"],
                        vertical_spacing=0.09)
    x = np.arange(signal.shape[0])
    for axis_i, axis_name in enumerate(AXES):
        fig.add_trace(go.Scatter(x=x, y=signal[:, axis_i], mode="lines", name=axis_name, line={"width": 2.4}), row=axis_i + 1, col=1)
        fig.update_yaxes(title_text="شتاب", row=axis_i + 1, col=1)
    fig.update_xaxes(title_text="شمارهٔ نمونه در پنجره", row=3, col=1)
    global_min = float(data[AXES].min().min())
    global_max = float(data[AXES].max().max())
    fig.update_yaxes(range=[global_min, global_max])
    fig.update_layout(height=650, title=f"{selected_activity} | شرکت‌کننده {signal_user} | پنجره {window_position}",
                      showlegend=False, margin=dict(t=70, b=30))
    style_chart(fig)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("برای این فعالیت پنجره‌ای با طول ۵۱ نمونه وجود ندارد.")

st.divider()
st.header("۲) مقایسهٔ چند فعالیت")
compare_user_options = sorted(meta.user_id.unique().tolist())
compare_user_choice = st.selectbox("شرکت‌کننده برای مقایسهٔ فعالیت‌ها", ["اولین فرد دارای فعالیت", *compare_user_options], index=0)
compare_activities = st.multiselect(
    "فعالیت‌های مورد مقایسه", ACTIVITY_ORDER,
    default=["Walking", "Jogging", "Upstairs"]
)
compare_window_position = st.number_input("شمارهٔ پنجره در هر فعالیت", min_value=0, value=0, step=1)
if compare_activities:
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        subplot_titles=["محور X", "محور Y", "محور Z"], vertical_spacing=0.08)
    notes = []
    for activity in compare_activities:
        candidates = meta.index[meta.activity == activity].tolist()
        user_used = None
        if compare_user_choice != "اولین فرد دارای فعالیت":
            user_candidates = meta.index[(meta.activity == activity) &
                                         (meta.user_id == int(compare_user_choice))].tolist()
            if user_candidates:
                candidates = user_candidates
        if candidates:
            pos = min(int(compare_window_position), len(candidates) - 1)
            idx = candidates[pos]
            user_used = int(meta.loc[idx, "user_id"])
            for axis_i, axis_name in enumerate(AXES):
                fig.add_trace(go.Scatter(
                    x=np.arange(51), y=X_windows[idx, :, axis_i],
                    mode="lines", name=f"{activity} — {axis_name}",
                    line={"width": 2.4}, legendgroup=activity,
                ), row=axis_i + 1, col=1)
            if compare_user_choice != "اولین فرد دارای فعالیت" and user_used != int(compare_user_choice):
                notes.append(f"{activity}: کاربر انتخابی داده ندارد؛ از کاربر {user_used} استفاده شد.")
            elif compare_user_choice == "اولین فرد دارای فعالیت":
                notes.append(f"{activity}: کاربر {user_used} برای این فعالیت انتخاب شد.")
        else:
            notes.append(f"{activity}: پنجره‌ای موجود نیست.")
    fig.update_xaxes(title_text="شمارهٔ نمونه در پنجره", row=3, col=1)
    global_min = float(data[AXES].min().min())
    global_max = float(data[AXES].max().max())
    fig.update_yaxes(range=[global_min, global_max])
    fig.update_layout(height=720, title="مقایسهٔ هم‌زمان سیگنال‌ها با مقیاس عمودی مشترک", margin=dict(t=65, b=30))
    style_chart(fig)
    st.plotly_chart(fig, use_container_width=True)
    for note in notes:
        st.caption(note)
else:
    st.info("حداقل یک فعالیت انتخاب کنید.")

st.divider()
st.header("۳) مقایسهٔ یک فعالیت بین شرکت‌کنندگان")
subject_activity = st.selectbox("فعالیت برای مقایسهٔ افراد", ACTIVITY_ORDER, index=0, key="subject_activity")
subject_candidates = sorted(meta.loc[meta.activity == subject_activity, "user_id"].unique().tolist())
selected_subjects = st.multiselect(
    "افراد", subject_candidates,
    default=subject_candidates[:min(3, len(subject_candidates))],
    key="selected_subjects",
)
subject_window_position = st.number_input("شمارهٔ پنجره برای هر فرد", min_value=0, value=0, step=1, key="subject_window_position")
if selected_subjects:
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        subplot_titles=["محور X", "محور Y", "محور Z"], vertical_spacing=0.08)
    for user in selected_subjects:
        candidates = meta.index[(meta.activity == subject_activity) & (meta.user_id == user)].tolist()
        if not candidates:
            continue
        idx = candidates[min(int(subject_window_position), len(candidates) - 1)]
        for axis_i, axis_name in enumerate(AXES):
            fig.add_trace(go.Scatter(
                x=np.arange(51), y=X_windows[idx, :, axis_i], mode="lines",
                name=f"User {user} — {axis_name}", line={"width": 2.4}, legendgroup=f"user{user}"
            ), row=axis_i + 1, col=1)
    fig.update_xaxes(title_text="شمارهٔ نمونه در پنجره", row=3, col=1)
    global_min = float(data[AXES].min().min())
    global_max = float(data[AXES].max().max())
    fig.update_yaxes(range=[global_min, global_max])
    fig.update_layout(height=720, title=f"{subject_activity}: مقایسهٔ افراد با مقیاس مشترک")
    style_chart(fig)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("برای مشاهده، دست‌کم یک شرکت‌کننده انتخاب کنید.")

st.divider()
st.header("۴) آمار توصیفی به تفکیک فعالیت")
st.dataframe(
    data.groupby("activity")[AXES].agg(["mean", "std", "min", "median", "max"]).round(4),
    use_container_width=True,
)
st.caption("مقیاس محورهای عمودی در نمودارهای مقایسه‌ای یکسان نگه داشته شده است. Timestamp فقط برای ترتیب نمونه‌ها استفاده می‌شود و زمان UTC فرض نشده است.")
