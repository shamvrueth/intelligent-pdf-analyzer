import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import precision_recall_fscore_support
from sklearn.pipeline import Pipeline
import joblib

# hyperparameters

TRAIN_CSV = "input.csv"                
MODEL_OUT = Path("models/rf_heading5.joblib")
N_ESTIMATORS = 300
RANDOM_STATE = 42
PROB_THRESHOLD = 0.4                      # threshold for inference

def load_and_clean(csv_path):
    df = pd.read_csv(csv_path)
    df["font_size"] = pd.to_numeric(df["font_size"], errors="coerce")
    df["is_bold"] = df["is_bold"].astype(int)
    df["page"] = pd.to_numeric(df["page"], errors="coerce").astype(int)
    for c in ["x0","y0","x1","y1"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["centeredness"] = pd.to_numeric(df["centeredness"], errors="coerce").fillna(0)
    df["is_semantic"] = pd.to_numeric(df["is_semantic"], errors="coerce").fillna(0)

    df["font_size"] = df["font_size"].fillna(df["font_size"].median()) # fill missing font sizes with median
    # heading = 1 , none = 0
    df["y"] = df["label"].astype(str).str.upper().map(lambda s: 1 if s == "HEADING" else 0)

    return df

def add_derived_features(df):
    df = df.copy()
    df["bbox_w"] = df["x1"] - df["x0"]
    df["bbox_h"] = df["y1"] - df["y0"]
    df["x_ctr"] = (df["x0"] + df["x1"]) / 2.0
    df["y_ctr"] = (df["y0"] + df["y1"]) / 2.0
    df["text_len"] = df["text"].fillna("").str.len()
    df["word_count"] = df["text"].fillna("").str.split().apply(len)
    # simple guard: replace any negative bbox dims with small positive
    df["bbox_w"] = df["bbox_w"].clip(lower=1e-3)
    df["bbox_h"] = df["bbox_h"].clip(lower=1e-3)
    df["relative_font_size"] = df["font_size"] / df.groupby("id")["font_size"].transform("median")
    df["relative_y"] = df["y0"] / df.groupby("id")["y0"].transform("max")
    g = df.groupby("id")["font_size"]
    df["font_pct"] = g.transform(lambda s: s.rank(method="average", pct=True))
    df["font_rank"] = g.transform(lambda s: s.rank(ascending=False, method="min"))  # 1 is largest
    df["ratio_to_max"] = df["font_size"] / g.transform("max")
    df["font_z"] = (df["font_size"] - g.transform("median")) / (g.transform("std").replace(0, 1))
    df["relsize_centered"] = df["relative_font_size"] * df["centeredness"]
    df["relsize_bold"] = df["relative_font_size"] * df["is_bold"]
    df["top_and_large"] = ((df["relative_y"] < 0.25) & (df["font_pct"] > 0.75)).astype(int)
    return df

def build_features_and_target(df):
    numeric_cols = [
        "font_size","bbox_w","bbox_h","x_ctr","y_ctr",
        "text_len","word_count","centeredness","is_semantic"
    ]
    bool_cols = ["is_bold"]  # already int 0/1
    page_col = ["page"]
    added_cols = ["bbox_w", "bbox_h", "x_ctr", "y_ctr", "text_len", "word_count", "bbox_w", "bbox_h", "relative_font_size", "relative_y",
                   "font_pct", "font_rank", "ratio_to_max", "font_z", "relsize_centered", "relsize_bold", "top_and_large"]

    feature_cols = bool_cols + numeric_cols + page_col + added_cols
    X = df[feature_cols].fillna(0)
    y = df["y"].values
    groups = df["id"].values
    return X, y, groups, feature_cols

def evaluate_group_cv(X, y, groups, feature_cols):
    unique_groups = np.unique(groups)
    n_splits = 5
    gkf = GroupKFold(n_splits=n_splits)

    prfs = []
    fold = 0
    for train_idx, test_idx in gkf.split(X, y, groups):
        fold += 1
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf = RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        clf.fit(X_train, y_train)
        probs = clf.predict_proba(X_test)[:, 1]
        y_pred = (probs >= PROB_THRESHOLD).astype(int)

        p, r, f, _ = precision_recall_fscore_support(y_test, y_pred, labels=[1], average="binary", zero_division=0)
        prfs.append((p, r, f))
        print(f"Fold {fold}: groups held-out = {np.unique(groups[test_idx]).tolist()}")
        print(f"HEADING precision={p:.3f}, recall={r:.3f}, f1={f:.3f}")

    p_avg = np.mean([x[0] for x in prfs])
    r_avg = np.mean([x[1] for x in prfs])
    f_avg = np.mean([x[2] for x in prfs])
    print(f"\nCross-fold avg — precision={p_avg:.3f}, recall={r_avg:.3f}, f1={f_avg:.3f}")
    return

def train(X, y):
    clf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    clf.fit(X, y)
    return clf

def save_model(model, feature_cols, path):
    os.makedirs(path.parent, exist_ok=True)
    payload = {"model": model, "feature_cols": feature_cols}
    joblib.dump(payload, path)
    print(f"Saved model and metadata to {path}")


if __name__ == "__main__":
    print("Loading and cleaning data...")
    df = load_and_clean(TRAIN_CSV)
    df = add_derived_features(df)
    # print(df.head())
    # print(df.tail())
    print(f"Rows: {len(df)}, Heading positives: {int(df['y'].sum())}, Docs: {df['id'].nunique()}")

    X, y, groups, feature_cols = build_features_and_target(df)

    print("Running grouped cross-validation evaluation...")
    evaluate_group_cv(X, y, groups, feature_cols)

    print("Training final model on all data...")
    model = train(X, y)

    # Save model + feature list
    save_model(model, feature_cols, MODEL_OUT)

    df["pred"] = model.predict(X)
    print("Predicted headings:", int(df["pred"].sum()))


