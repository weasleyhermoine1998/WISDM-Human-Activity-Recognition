"""توابع مشترک پروژه PT1 تشخیص فعالیت انسانی با داده WISDM."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple, Any
import warnings

import numpy as np
import pandas as pd

ACTIVITY_ORDER = ["Walking", "Jogging", "Upstairs", "Downstairs", "Sitting", "Standing"]
AXES = ["X", "Y", "Z"]
RAW_COLUMNS = ["user_id", "activity", "timestamp", "X", "Y", "Z"]





def load_wisdm_data(path: str | Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Load and clean WISDM data using the notebook reference method."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"فایل داده پیدا نشد: {path}")

    columns = ["user_id", "activity", "timestamp", "X", "Y", "Z"]

    df = pd.read_csv(
        path,
        header=None,
        names=columns,
        sep=",",
        on_bad_lines="skip"
    )

    raw_rows = len(df)

    # پاک‌سازی انتهای ستون Z
    df["Z"] = (
        df["Z"].astype(str).str.strip().str.rstrip(";")
    )

    # تبدیل ستون‌های عددی
    numeric_cols = ["user_id", "timestamp", "X", "Y", "Z"]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # حذف رکوردهای دارای مقادیر نامعتبر
    df = df.dropna(
        subset=["user_id", "activity", "timestamp", "X", "Y", "Z"]
    ).copy()

    df["user_id"] = df["user_id"].astype("int64")
    df["timestamp"] = df["timestamp"].astype("int64")

    # مرتب‌سازی مطابق نوت‌بوک
    df = df.sort_values(
        ["user_id", "timestamp"]
    ).reset_index(drop=True)

    # آمار کنترل کیفیت
    audit = {
        "raw_lines": None,
        "parsed_rows": raw_rows,
        "clean_rows": int(len(df)),
        "rows_removed_total": int(raw_rows - len(df)),
        "num_users": int(df["user_id"].nunique()),
        "num_activities": int(df["activity"].nunique()),
        "users": sorted(df["user_id"].unique().tolist()),
        "activities": [
            a for a in ACTIVITY_ORDER
            if a in set(df["activity"])
        ],
    }

    audit["exact_duplicate_rows"] = int(
        df.duplicated(subset=columns).sum()
    )

    if audit["num_activities"] != 6:
        warnings.warn(
            f"در داده پاک‌شده {audit['num_activities']} فعالیت شناسایی شد.",
            RuntimeWarning,
        )

    return df, audit


def outlier_audit(data: pd.DataFrame) -> pd.DataFrame:
    """IQR-based outlier candidates per axis; candidates are not removed."""
    rows = []
    for col in AXES:
        q1, q3 = data[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        mask = (data[col] < low) | (data[col] > high)
        rows.append({
            "axis": col,
            "Q1": q1,
            "Q3": q3,
            "IQR": iqr,
            "lower_fence": low,
            "upper_fence": high,
            "candidate_count": int(mask.sum()),
            "candidate_percent": float(mask.mean() * 100),
        })
    return pd.DataFrame(rows).set_index("axis")


def make_windows(
    data: pd.DataFrame,
    window_size: int = 51,
    step: int = 26,
    split_on_time_gaps: bool = True,
    gap_factor: float = 2.5,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Create pure-label windows without crossing users/activity runs.

    Rows are first sorted by participant and timestamp. A new segment starts
    whenever the activity changes. Optionally, unusually large timestamp gaps
    (relative to the participant's median positive sampling interval) also
    start a new segment, preventing windows from bridging recording breaks.
    """
    if window_size <= 0 or step <= 0:
        raise ValueError("window_size و step باید مثبت باشند.")
    required = {"user_id", "activity", "timestamp", *AXES}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"ستون‌های لازم وجود ندارند: {sorted(missing)}")

    df = data.sort_values(["user_id", "timestamp"], kind="mergesort").copy()
    previous_activity = df.groupby("user_id", sort=False)["activity"].shift()
    new_segment = previous_activity.isna() | df["activity"].ne(previous_activity)

    if split_on_time_gaps:
        dt = df.groupby("user_id", sort=False)["timestamp"].diff()
        positive_dt = dt.where(dt > 0)
        median_dt = positive_dt.groupby(df["user_id"]).median()
        threshold = df["user_id"].map(median_dt * gap_factor)
        new_segment |= (dt > threshold).fillna(False)

    df["segment_id"] = new_segment.groupby(df["user_id"], sort=False).cumsum().astype("int64")

    windows = []
    metadata = []
    for (user_id, segment_id), group in df.groupby(["user_id", "segment_id"], sort=False):
        values = group[AXES].to_numpy(dtype=np.float32, copy=True)
        if len(values) < window_size:
            continue
        label_values = group["activity"].unique()
        if len(label_values) != 1:
            raise AssertionError("یک قطعه شامل بیش از یک برچسب فعالیت است.")
        activity = str(label_values[0])
        timestamps = group["timestamp"].to_numpy()
        for start in range(0, len(values) - window_size + 1, step):
            end = start + window_size
            windows.append(values[start:end])
            metadata.append({
                "user_id": int(user_id),
                "activity": activity,
                "segment_id": int(segment_id),
                "start_index_in_segment": int(start),
                "start_timestamp": int(timestamps[start]),
                "end_timestamp": int(timestamps[end - 1]),
                "window_size": int(window_size),
            })

    if not windows:
        return np.empty((0, window_size, 3), dtype=np.float32), pd.DataFrame(
            columns=["user_id", "activity", "segment_id", "start_index_in_segment",
                     "start_timestamp", "end_timestamp", "window_size"]
        )
    X = np.stack(windows).astype(np.float32, copy=False)
    meta = pd.DataFrame(metadata)
    if len(X) != len(meta):
        raise AssertionError("تعداد پنجره‌ها و ردیف‌های metadata برابر نیست.")
    return X, meta


def subject_wise_split(
    y: np.ndarray,
    groups: np.ndarray,
    random_states=range(40),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Approximate 60/20/20 train/validation/test split by participant.

    Uses StratifiedGroupKFold (5 folds), then selects one fold for validation,
    one for test, and the remaining three for training. A small deterministic
    search chooses the assignment with good subject/window/class balance.
    """
    from sklearn.model_selection import StratifiedGroupKFold

    y = np.asarray(y)
    groups = np.asarray(groups)
    if len(y) != len(groups):
        raise ValueError("طول y و groups باید برابر باشد.")
    if len(np.unique(groups)) < 5:
        raise ValueError("برای تقسیم پنج‌فولدی دست‌کم پنج شرکت‌کننده لازم است.")

    classes, y_encoded = np.unique(y, return_inverse=True)
    n_classes = len(classes)
    global_class = np.bincount(y_encoded, minlength=n_classes).astype(float)
    global_class /= max(global_class.sum(), 1)
    target = np.array([0.60, 0.20, 0.20])
    best = None
    n = len(y)
    all_groups = np.unique(groups)

    for seed in random_states:
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=int(seed))
        fold_id = np.full(n, -1, dtype=np.int8)
        try:
            for fold, (_, held_out) in enumerate(cv.split(np.zeros(n), y_encoded, groups)):
                fold_id[held_out] = fold
        except ValueError:
            continue
        if np.any(fold_id < 0):
            continue

        fold_rows = np.bincount(fold_id, minlength=5).astype(float)
        fold_classes = np.vstack([
            np.bincount(y_encoded[fold_id == f], minlength=n_classes)
            for f in range(5)
        ]).astype(float)
        fold_subjects = np.array([
            len(np.unique(groups[fold_id == f])) for f in range(5)
        ], dtype=float)

        for val_fold in range(5):
            for test_fold in range(5):
                if val_fold == test_fold:
                    continue
                train_mask = (fold_id != val_fold) & (fold_id != test_fold)
                val_mask = fold_id == val_fold
                test_mask = fold_id == test_fold
                masks = [train_mask, val_mask, test_mask]
                row_counts = np.array([m.sum() for m in masks], dtype=float)
                subject_counts = np.array([
                    len(np.unique(groups[m])) for m in masks
                ], dtype=float)
                row_share = row_counts / n
                subject_share = subject_counts / max(len(all_groups), 1)
                class_counts = np.vstack([
                    np.bincount(y_encoded[m], minlength=n_classes) for m in masks
                ]).astype(float)
                class_share = class_counts / np.maximum(row_counts[:, None], 1)
                missing = int((class_counts == 0).sum())

                row_error = np.mean((row_share - target) ** 2)
                subject_error = np.mean((subject_share - target) ** 2)
                class_error = np.mean((class_share - global_class[None, :]) ** 2)
                score = 3.0 * row_error + 2.0 * subject_error + 2.0 * class_error + 20.0 * missing
                candidate = (score, int(seed), val_fold, test_fold, masks, class_counts)
                if best is None or candidate[0] < best[0]:
                    best = candidate

    if best is None:
        raise RuntimeError("ساخت تقسیم گروهی ممکن نشد؛ تعداد و توزیع داده‌ها را بررسی کنید.")

    score, seed, val_fold, test_fold, masks, class_counts = best
    train_idx, val_idx, test_idx = [np.flatnonzero(mask) for mask in masks]
    split_info = {
        "random_state": seed,
        "validation_fold": val_fold,
        "test_fold": test_fold,
        "score": float(score),
        "class_coverage": {
            "train": classes[class_counts[0] > 0].tolist(),
            "validation": classes[class_counts[1] > 0].tolist(),
            "test": classes[class_counts[2] > 0].tolist(),
        },
        "missing_classes_in_any_split": bool((class_counts == 0).any()),
    }
    return train_idx, val_idx, test_idx, split_info


def split_summary(meta: pd.DataFrame, indices: Dict[str, np.ndarray]) -> pd.DataFrame:
    """Return rows/users and class distribution for each data partition."""
    rows = []
    for split_name, idx in indices.items():
        part = meta.iloc[idx]
        rows.append({
            "split": split_name,
            "participants": int(part["user_id"].nunique()),
            "windows": int(len(part)),
            **{f"{activity}_windows": int((part["activity"] == activity).sum())
               for activity in ACTIVITY_ORDER},
        })
    return pd.DataFrame(rows).set_index("split")
