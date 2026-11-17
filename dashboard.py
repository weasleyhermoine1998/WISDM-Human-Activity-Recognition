"""Interactive WISDM HAR dashboard for course PT1.
Run: streamlit run dashboard.py
"""
from __future__ import annotations

from pathlib import Path
import hashlib
import os
import tempfile
import gdown

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from pt1_utils import (
    ACTIVITY_ORDER, AXES, load_wisdm_data, make_windows,
    outlier_audit, subject_wise_split,
)

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

default_path = Path(
    os.environ.get("WISDM_DATA_PATH", "data/WISDM_ar_v1.1_raw.txt")
)

uploaded = st.sidebar.file_uploader(
    "فایل خام WISDM (اختیاری)",
    type=["txt", "csv"]
)

if uploaded is not None:
    payload = uploaded.getvalue()
    digest = hashlib.sha256(payload).hexdigest()[:16]
    temp_path = Path(tempfile.gettempdir()) / f"wisdm_upload_{digest}.txt"

    if not temp_path.exists() or temp_path.stat().st_size != len(payload):
        temp_path.write_bytes(payload)

    data_path = temp_path

else:
    if default_path.exists():
        data_path = default_path
    else:
        drive_file_id = "1fbGo9y1y0kNO2qiFQYEuxxcSFpq8U0Pt"
        download_path = Path(tempfile.gettempdir()) / "WISDM_ar_v1.1_raw.txt"

        try:
            if not download_path.exists() or download_path.stat().st_size == 0:
                with st.spinner("در حال دریافت دیتاست از Google Drive..."):
                    result = gdown.download(
                        id=drive_file_id,
                        output=str(download_path),
                        quiet=False
                    )

                if result is None or not download_path.exists():
                    raise RuntimeError("دانلود فایل از Google Drive ناموفق بود.")

            if download_path.stat().st_size == 0:
                raise RuntimeError("فایل دانلودشده خالی است.")

            data_path = download_path
            st.sidebar.success("دیتاست با موفقیت دریافت شد.")

        except Exception as exc:
            st.error(f"دریافت دیتاست ناموفق بود: {exc}")
            st.info(
                "دسترسی فایل Google Drive را روی Anyone with the link و Viewer "
                "قرار دهید یا فایل را از نوار کناری بارگذاری کنید."
            )
            st.stop()

if not data_path.exists():
    st.error("فایل دیتاست پیدا نشد.")
    st.stop()


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
    default=ACTIVITY_ORDER
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



st.subheader("آمار توصیفی کلی سه محور")
descriptive_all = data[AXES].describe(percentiles=[0.25, 0.50, 0.75]).T
descriptive_all = descriptive_all[["mean", "std", "min", "25%", "50%", "75%", "max"]]
st.dataframe(descriptive_all.round(4), use_container_width=True)

st.subheader("توزیع مقادیر محورها به تفکیک فعالیت (Boxplot)")
box_cols = st.columns(3)
for axis_i, axis_name in enumerate(AXES):
    with box_cols[axis_i]:
        box_fig = go.Figure()
        for activity in ACTIVITY_ORDER:
            values = data.loc[data["activity"] == activity, axis_name].dropna()
            q1, median, q3 = values.quantile([0.25, 0.50, 0.75]).to_numpy()
            iqr = q3 - q1
            lower_candidates = values[values >= q1 - 1.5 * iqr]
            upper_candidates = values[values <= q3 + 1.5 * iqr]
            lower_whisker = float(lower_candidates.min()) if len(lower_candidates) else float(q1)
            upper_whisker = float(upper_candidates.max()) if len(upper_candidates) else float(q3)
            box_fig.add_trace(go.Box(
                name=activity,
                q1=[q1], median=[median], q3=[q3],
                lowerfence=[lower_whisker], upperfence=[upper_whisker],
                boxpoints=False,
                showlegend=False
            ))
        box_fig.update_layout(
            title=f"محور {axis_name}",
            yaxis_title="شتاب",
            height=430,
            margin=dict(t=45, b=25)
        )
        style_chart(box_fig)
        st.plotly_chart(box_fig, use_container_width=True)
st.caption("برای هم‌خوانی با نوت‌بوک، نقاط پرت در Boxplot نمایش داده نمی‌شوند؛ اما در داده حذف نشده‌اند.")



# ---------------------------------------------------------------------
# ۵) مقایسه‌های سیگنال مطابق نوت‌بوک
# ---------------------------------------------------------------------
st.divider()
st.header("۵) مقایسه سیگنال‌ها مطابق نوت‌بوک")

# Select a participant common to all six activities, as in notebook cell 17.
users_per_activity = {
    activity: set(meta.loc[meta.activity == activity, "user_id"].unique())
    for activity in ACTIVITY_ORDER
}
available_user_sets = [users_per_activity[a] for a in ACTIVITY_ORDER if users_per_activity[a]]
common_users = set.intersection(*available_user_sets) if len(available_user_sets) == len(ACTIVITY_ORDER) else set()
default_common_user = sorted(common_users)[0] if common_users else None

if default_common_user is not None:
    six_user = st.selectbox(
        "شرکت‌کننده مشترک برای مقایسه شش فعالیت",
        sorted(common_users),
        index=sorted(common_users).index(default_common_user),
        key="six_activity_user"
    )
else:
    six_user = None
    st.warning("شرکت‌کننده مشترکی برای هر شش فعالیت پیدا نشد؛ برای هر فعالیت اولین پنجره موجود انتخاب می‌شود.")

six_fig = make_subplots(
    rows=3, cols=2, shared_xaxes=True, shared_yaxes=True,
    subplot_titles=ACTIVITY_ORDER, vertical_spacing=0.10, horizontal_spacing=0.08
)
six_used = {}
global_min = float(data[AXES].min().min())
global_max = float(data[AXES].max().max())
for position, activity in enumerate(ACTIVITY_ORDER):
    row, col = position // 2 + 1, position % 2 + 1
    candidates = meta.index[meta.activity == activity].tolist()
    if six_user is not None:
        same_user = meta.index[
            (meta.activity == activity) & (meta.user_id == int(six_user))
        ].tolist()
    else:
        same_user = []
    chosen_idx = same_user[0] if same_user else (candidates[0] if candidates else None)
    if chosen_idx is None:
        six_used[activity] = None
        continue
    six_used[activity] = int(meta.loc[chosen_idx, "user_id"])
    for axis_i, axis_name in enumerate(AXES):
        six_fig.add_trace(
            go.Scatter(
                x=np.arange(51), y=X_windows[chosen_idx, :, axis_i],
                mode="lines", name=axis_name,
                legendgroup=axis_name,
                showlegend=(position == 0),
                line={"width": 1.6}
            ),
            row=row, col=col
        )
    six_fig.update_yaxes(range=[global_min, global_max], row=row, col=col)
    six_fig.update_xaxes(title_text="شماره نمونه در پنجره", row=row, col=col)
six_fig.update_layout(
    title="مقایسه سیگنال سه‌محوره شش فعالیت با مقیاس عمودی مشترک",
    height=850, margin=dict(t=75, b=35)
)
style_chart(six_fig)
st.plotly_chart(six_fig, use_container_width=True)
st.caption("شرکت‌کننده استفاده‌شده برای هر فعالیت: " + " | ".join(
    f"{a}: {u if u is not None else 'بدون پنجره'}" for a, u in six_used.items()
))

# Walking comparison for the first three participants with complete windows.
st.subheader("تفاوت‌های فردی در فعالیت Walking")
walking_users = sorted(meta.loc[meta.activity == "Walking", "user_id"].unique().tolist())
selected_walking_users = walking_users[:3]
if len(selected_walking_users) >= 3:
    walking_fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, shared_yaxes=True,
        subplot_titles=[f"Walking | شرکت‌کننده {u}" for u in selected_walking_users],
        vertical_spacing=0.08
    )
    for row, user in enumerate(selected_walking_users, start=1):
        idx = meta.index[
            (meta.activity == "Walking") & (meta.user_id == user)
        ][0]
        for axis_i, axis_name in enumerate(AXES):
            walking_fig.add_trace(
                go.Scatter(
                    x=np.arange(51), y=X_windows[idx, :, axis_i],
                    mode="lines", name=axis_name,
                    legendgroup=axis_name, showlegend=(row == 1),
                    line={"width": 1.7}
                ),
                row=row, col=1
            )
        walking_fig.update_yaxes(range=[global_min, global_max], row=row, col=1)
    walking_fig.update_xaxes(title_text="شماره نمونه در پنجره", row=3, col=1)
    walking_fig.update_layout(
        title="مقایسه سیگنال Walking بین سه شرکت‌کننده اول",
        height=760, margin=dict(t=75, b=35)
    )
    style_chart(walking_fig)
    st.plotly_chart(walking_fig, use_container_width=True)
    st.caption(f"شرکت‌کنندگان انتخاب‌شده مطابق نوت‌بوک: {selected_walking_users}")
else:
    st.info("برای مقایسه Walking حداقل سه شرکت‌کننده دارای پنجره کامل لازم است.")

# Similar activity pairs; select the first shared participant for each pair.
st.subheader("مقایسه جفت فعالیت‌های مشابه")
similar_pairs = [("Sitting", "Standing"), ("Walking", "Upstairs")]
pair_fig = make_subplots(
    rows=2, cols=2, shared_xaxes=True, shared_yaxes=True,
    subplot_titles=["Sitting", "Standing", "Walking", "Upstairs"],
    vertical_spacing=0.12, horizontal_spacing=0.08
)
pair_notes = []
for pair_row, (activity_a, activity_b) in enumerate(similar_pairs, start=1):
    users_a = set(meta.loc[meta.activity == activity_a, "user_id"].unique())
    users_b = set(meta.loc[meta.activity == activity_b, "user_id"].unique())
    shared = sorted(users_a & users_b)
    preferred_user = shared[0] if shared else None
    for pair_col, activity in enumerate((activity_a, activity_b), start=1):
        candidates = meta.index[meta.activity == activity].tolist()
        same = meta.index[
            (meta.activity == activity) & (meta.user_id == preferred_user)
        ].tolist() if preferred_user is not None else []
        idx = same[0] if same else (candidates[0] if candidates else None)
        if idx is None:
            pair_notes.append(f"{activity}: پنجره موجود نیست")
            continue
        used_user = int(meta.loc[idx, "user_id"])
        pair_notes.append(f"{activity}: شرکت‌کننده {used_user}")
        for axis_i, axis_name in enumerate(AXES):
            pair_fig.add_trace(
                go.Scatter(
                    x=np.arange(51), y=X_windows[idx, :, axis_i],
                    mode="lines", name=axis_name,
                    legendgroup=axis_name,
                    showlegend=(pair_row == 1 and pair_col == 1),
                    line={"width": 1.6}
                ),
                row=pair_row, col=pair_col
            )
        pair_fig.update_yaxes(range=[global_min, global_max], row=pair_row, col=pair_col)
        pair_fig.update_xaxes(title_text="شماره نمونه", row=pair_row, col=pair_col)
pair_fig.update_layout(
    title="Sitting / Standing و Walking / Upstairs — مقیاس عمودی مشترک",
    height=720, margin=dict(t=75, b=35)
)
style_chart(pair_fig)
st.plotly_chart(pair_fig, use_container_width=True)
st.caption(" | ".join(pair_notes))

# ---------------------------------------------------------------------
# ۶) تحلیل پنجره‌بندی مطابق نوت‌بوک
# ---------------------------------------------------------------------
st.divider()
st.header("۶) تحلیل پنجره‌بندی زمانی")
window_size = int(X_windows.shape[1])
window_step = 26  # همان مقدار استفاده‌شده در نوت‌بوک
overlap_samples = window_size - window_step
window_metrics = st.columns(4)
window_metrics[0].metric("تعداد پنجره‌ها", f"{len(meta):,}")
window_metrics[1].metric("تعداد قطعات زمانی", f"{meta.groupby(['user_id', 'segment_id']).ngroups:,}")
window_metrics[2].metric("نمونه در هر پنجره", f"{window_size}")
window_metrics[3].metric("درصد هم‌پوشانی", f"{100 * overlap_samples / window_size:.2f}%")

st.write(f"**ابعاد آرایه پنجره‌ها:** `{X_windows.shape}`")
st.write(
    f"**گام حرکت:** {window_step} نمونه | "
    f"**نمونه‌های مشترک:** {overlap_samples} | "
    f"**مدت تقریبی هر پنجره با فرض نرخ ۲۰ هرتز:** {window_size / 20:.3f} ثانیه"
)
window_counts = (
    meta["activity"].value_counts()
    .reindex(ACTIVITY_ORDER, fill_value=0)
)
window_fig = go.Figure(go.Bar(
    x=window_counts.index, y=window_counts.values,
    text=[f"{v:,}" for v in window_counts.values], textposition="outside"
))
window_fig.update_layout(
    title="توزیع پنجره‌ها به تفکیک فعالیت",
    xaxis_title="فعالیت", yaxis_title="تعداد پنجره",
    height=420, margin=dict(t=55, b=30)
)
style_chart(window_fig)
st.plotly_chart(window_fig, use_container_width=True)
window_table = pd.DataFrame({
    "تعداد پنجره": window_counts,
    "درصد": 100 * window_counts / max(len(meta), 1)
})
st.dataframe(window_table.round(2), use_container_width=True)

# ---------------------------------------------------------------------
# ۷) تقسیم داده بر اساس شرکت‌کننده و مدل پایه
# ---------------------------------------------------------------------
st.divider()
st.header("۷) تقسیم داده‌ها بر اساس شرکت‌کننده (Subject-wise Split)")
st.caption(
    "تقسیم‌بندی با همان تابع subject_wise_split در pt1_utils انجام می‌شود؛ "
    "شناسه‌های کاربران بین Train، Validation و Test مشترک نخواهند بود."
)

@st.cache_data(show_spinner="در حال تقسیم داده‌ها و آموزش مدل‌ها...")
def cached_model_analysis(X_array, metadata):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score, f1_score, classification_report, confusion_matrix
    )

    labels = metadata["activity"].to_numpy()
    groups = metadata["user_id"].to_numpy()
    train_i, val_i, test_i, split_details = subject_wise_split(labels, groups)
    split_indices = {
        "Train": train_i,
        "Validation": val_i,
        "Test": test_i
    }

    split_rows = []
    split_users = {}
    for split_name, indices in split_indices.items():
        part = metadata.iloc[indices]
        split_users[split_name] = sorted(part["user_id"].unique().astype(int).tolist())
        row = {
            "مجموعه": split_name,
            "شرکت‌کنندگان": part["user_id"].nunique(),
            "پنجره‌ها": len(part),
            "درصد": 100 * len(part) / len(metadata)
        }
        for activity in ACTIVITY_ORDER:
            row[activity] = int((part["activity"] == activity).sum())
        split_rows.append(row)
    split_frame = pd.DataFrame(split_rows).set_index("مجموعه")

    # Flatten each 51x3 window into the same 153 raw features as the notebook.
    flat = X_array.reshape(len(X_array), -1)
    X_train, y_train = flat[train_i], labels[train_i]
    X_val, y_val = flat[val_i], labels[val_i]
    X_test, y_test = flat[test_i], labels[test_i]

    majority_class = pd.Series(y_train).value_counts().idxmax()
    majority_val = np.repeat(majority_class, len(y_val))
    majority_val_metrics = {
        "مدل": "Majority Class",
        "Validation Accuracy": accuracy_score(y_val, majority_val),
        "Validation Macro-F1": f1_score(
            y_val, majority_val, labels=ACTIVITY_ORDER,
            average="macro", zero_division=0
        )
    }

    C_VALUES = [0.01, 0.1, 1.0, 10.0]
    validation_rows = []
    trained_models = {}
    for C in C_VALUES:
        candidate = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                C=C, max_iter=3000, class_weight="balanced",
                solver="lbfgs", random_state=42
            ))
        ])
        candidate.fit(X_train, y_train)
        pred = candidate.predict(X_val)
        validation_rows.append({
            "C": C,
            "Validation Accuracy": accuracy_score(y_val, pred),
            "Validation Macro-F1": f1_score(
                y_val, pred, labels=ACTIVITY_ORDER,
                average="macro", zero_division=0
            )
        })
        trained_models[C] = candidate

    tuning = pd.DataFrame(validation_rows).sort_values(
        ["Validation Macro-F1", "Validation Accuracy"],
        ascending=False
    ).reset_index(drop=True)
    best_C = float(tuning.iloc[0]["C"])
    best_model = trained_models[best_C]

    model_test = best_model.predict(X_test)
    majority_test = np.repeat(majority_class, len(y_test))
    test_comparison = pd.DataFrame([
        {
            "مدل": "Majority Class",
            "Test Accuracy": accuracy_score(y_test, majority_test),
            "Test Macro-F1": f1_score(
                y_test, majority_test, labels=ACTIVITY_ORDER,
                average="macro", zero_division=0
            )
        },
        {
            "مدل": "Logistic Regression",
            "Test Accuracy": accuracy_score(y_test, model_test),
            "Test Macro-F1": f1_score(
                y_test, model_test, labels=ACTIVITY_ORDER,
                average="macro", zero_division=0
            )
        }
    ]).set_index("مدل")

    report = classification_report(
        y_test, model_test, labels=ACTIVITY_ORDER,
        output_dict=True, zero_division=0
    )
    per_class = pd.DataFrame(report).T.loc[
        ACTIVITY_ORDER, ["precision", "recall", "f1-score", "support"]
    ]
    per_class.index.name = "فعالیت"
    cm_majority = confusion_matrix(y_test, majority_test, labels=ACTIVITY_ORDER)
    cm_model = confusion_matrix(y_test, model_test, labels=ACTIVITY_ORDER)

    train_pred = best_model.predict(X_train)
    train_accuracy = accuracy_score(y_train, train_pred)
    test_accuracy = accuracy_score(y_test, model_test)
    train_f1 = f1_score(
        y_train, train_pred, labels=ACTIVITY_ORDER,
        average="macro", zero_division=0
    )
    test_f1 = f1_score(
        y_test, model_test, labels=ACTIVITY_ORDER,
        average="macro", zero_division=0
    )
    train_test = pd.DataFrame([
        {"معیار": "Accuracy", "Train": train_accuracy, "Test": test_accuracy,
         "Train - Test": train_accuracy - test_accuracy},
        {"معیار": "Macro-F1", "Train": train_f1, "Test": test_f1,
         "Train - Test": train_f1 - test_f1}
    ]).set_index("معیار")

    # Explicitly verify the participant-level isolation.
    user_sets = {k: set(v) for k, v in split_users.items()}
    no_user_overlap = (
        user_sets["Train"].isdisjoint(user_sets["Validation"])
        and user_sets["Train"].isdisjoint(user_sets["Test"])
        and user_sets["Validation"].isdisjoint(user_sets["Test"])
    )

    return (
        split_frame, split_users, split_details, no_user_overlap,
        majority_val_metrics, tuning, best_C, test_comparison,
        per_class, cm_majority, cm_model, train_test
    )

model_results = cached_model_analysis(
    X_windows, meta[["user_id", "activity"]].reset_index(drop=True)
)
(
    split_frame, split_users, split_details, no_user_overlap,
    majority_val_metrics, tuning, best_C, test_comparison,
    per_class, cm_majority, cm_model, train_test
) = model_results

split_metrics = st.columns(3)
for col, split_name in zip(split_metrics, ["Train", "Validation", "Test"]):
    col.metric(
        split_name,
        f"{int(split_frame.loc[split_name, 'پنجره‌ها']):,} پنجره",
        f"{int(split_frame.loc[split_name, 'شرکت‌کنندگان'])} شرکت‌کننده"
    )
st.dataframe(split_frame.round(2), use_container_width=True)
with st.expander("شناسه شرکت‌کنندگان در هر مجموعه و جزئیات الگوریتم"):
    for split_name, users in split_users.items():
        st.write(f"**{split_name}:** {users}")
    st.json(split_details)
if no_user_overlap:
    st.success("کنترل عدم اشتراک شرکت‌کنندگان بین سه مجموعه با موفقیت انجام شد.")
else:
    st.error("هشدار: بین مجموعه‌ها اشتراک شرکت‌کننده وجود دارد.")

st.subheader("۸) مدل پایه و انتخاب Logistic Regression")
st.write(
    f"کلاس اکثریت در Train: **{majority_val_metrics['مدل']}** — "
    f"دقت Validation: **{majority_val_metrics['Validation Accuracy']:.4f}** — "
    f"Macro-F1: **{majority_val_metrics['Validation Macro-F1']:.4f}**"
)
st.markdown("**نتایج تنظیم پارامتر C روی مجموعه Validation**")
st.dataframe(tuning.round(6), use_container_width=True)
st.metric("بهترین مقدار C انتخاب‌شده", f"{best_C:g}")
tuning_fig = go.Figure()
tuning_fig.add_trace(go.Bar(
    x=tuning["C"].astype(str), y=tuning["Validation Accuracy"],
    name="Validation Accuracy"
))
tuning_fig.add_trace(go.Bar(
    x=tuning["C"].astype(str), y=tuning["Validation Macro-F1"],
    name="Validation Macro-F1"
))
tuning_fig.update_layout(
    barmode="group", title="مقایسه مدل‌های Logistic Regression روی Validation",
    xaxis_title="پارامتر C", yaxis_title="امتیاز", height=430
)
style_chart(tuning_fig)
st.plotly_chart(tuning_fig, use_container_width=True)
st.caption(
    "مدل‌ها با StandardScaler و LogisticRegression(class_weight='balanced', "
    "max_iter=3000, solver='lbfgs', random_state=42) آموزش داده شده‌اند. "
    "مجموعه Test در انتخاب C دخالت ندارد."
)

# ---------------------------------------------------------------------
# ۹) ارزیابی نهایی روی Test و مقایسه Train/Test
# ---------------------------------------------------------------------
st.divider()
st.header("۹) ارزیابی نهایی مدل روی مجموعه آزمون")
test_metrics_cols = st.columns(2)
lr_accuracy = float(test_comparison.loc["Logistic Regression", "Test Accuracy"])
lr_f1 = float(test_comparison.loc["Logistic Regression", "Test Macro-F1"])
test_metrics_cols[0].metric("Logistic Regression — Test Accuracy", f"{lr_accuracy:.4f}")
test_metrics_cols[1].metric("Logistic Regression — Test Macro-F1", f"{lr_f1:.4f}")
st.dataframe(test_comparison.round(4), use_container_width=True)

st.subheader("ماتریس‌های درهم‌ریختگی روی Test")
cm_cols = st.columns(2)
for col, matrix, title in [
    (cm_cols[0], cm_majority, "Majority Class — Test"),
    (cm_cols[1], cm_model, "Logistic Regression — Test")
]:
    with col:
        cm_fig = go.Figure(go.Heatmap(
            z=matrix, x=ACTIVITY_ORDER, y=ACTIVITY_ORDER,
            colorscale="Blues", text=matrix, texttemplate="%{text}",
            colorbar={"title": "تعداد"}
        ))
        cm_fig.update_layout(
            title=title, xaxis_title="برچسب پیش‌بینی‌شده",
            yaxis_title="برچسب واقعی", height=520,
            margin=dict(t=55, b=90), xaxis_tickangle=-35
        )
        style_chart(cm_fig)
        st.plotly_chart(cm_fig, use_container_width=True)

st.subheader("معیارهای عملکرد Logistic Regression به تفکیک فعالیت")
st.dataframe(per_class.round(4), use_container_width=True)
f1_fig = go.Figure(go.Bar(
    x=per_class.index, y=per_class["f1-score"],
    text=[f"{v:.3f}" for v in per_class["f1-score"]],
    textposition="outside"
))
f1_fig.update_layout(
    title="F1-score برای هر فعالیت در مجموعه Test",
    xaxis_title="فعالیت", yaxis_title="F1-score",
    yaxis=dict(range=[0, 1]), height=430, margin=dict(t=55, b=30)
)
style_chart(f1_fig)
st.plotly_chart(f1_fig, use_container_width=True)

st.subheader("مقایسه عملکرد Train و Test")
st.dataframe(train_test.round(4), use_container_width=True)
generalization_fig = go.Figure()
generalization_fig.add_trace(go.Bar(
    x=["Accuracy", "Macro-F1"], y=train_test["Train"],
    name="Train"
))
generalization_fig.add_trace(go.Bar(
    x=["Accuracy", "Macro-F1"], y=train_test["Test"],
    name="Test"
))
generalization_fig.update_layout(
    barmode="group", title="مقایسه معیارهای Train و Test",
    yaxis_title="امتیاز", yaxis=dict(range=[0, 1]), height=420
)
style_chart(generalization_fig)
st.plotly_chart(generalization_fig, use_container_width=True)
st.caption(
    "اختلاف Train - Test در جدول نمایش داده شده است. این اختلاف به‌تنهایی "
    "اثبات قطعی نبود یا وجود بیش‌برازش نیست و باید همراه با ماهیت تقسیم داده تفسیر شود."
)
