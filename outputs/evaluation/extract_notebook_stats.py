import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"


def load_jsonl(path: Path) -> pd.DataFrame:
    with path.open(encoding="utf-8") as handle:
        return pd.DataFrame([json.loads(line) for line in handle if line.strip()])


def main() -> None:
    train = load_jsonl(DATA_DIR / "train.jsonl")
    dev = load_jsonl(DATA_DIR / "dev.jsonl")
    test = load_jsonl(DATA_DIR / "test.jsonl")
    all_data = pd.concat([train, dev, test], ignore_index=True)

    print("SIZES", {"train": len(train), "dev": len(dev), "test": len(test)})

    for name, frame in (("train", train), ("dev", dev)):
        print("BALANCE", name, frame["label"].value_counts().sort_index().to_dict())

    for name, frame in (("train", train), ("dev", dev), ("test", test)):
        print(
            "TEXT_MISSING",
            name,
            {
                "null": int(frame["text"].isna().sum()),
                "empty": int((frame["text"].fillna("").str.strip() == "").sum()),
            },
        )

    train["text_hash"] = train["text"].apply(lambda text: hashlib.md5(str(text).encode()).hexdigest())
    print("TRAIN_TEXT_DUPLICATES", int(train["text_hash"].duplicated().sum()))

    image_label_cardinality = train.groupby("img")["label"].nunique()
    print("CONFOUNDER_IMAGES", int((image_label_cardinality > 1).sum()))

    missing = 0
    corrupted = 0
    for _, row in all_data.iterrows():
        image_path = DATA_DIR / row["img"]
        if not image_path.exists():
            missing += 1
            continue
        try:
            with Image.open(image_path) as image:
                image.verify()
        except Exception:
            corrupted += 1
    print("IMAGE_CHECK", {"missing": missing, "corrupted": corrupted, "total": len(all_data)})

    widths = []
    heights = []
    for _, row in train.sample(200, random_state=42).iterrows():
        try:
            with Image.open(DATA_DIR / row["img"]) as image:
                width, height = image.size
            widths.append(width)
            heights.append(height)
        except Exception:
            continue
    print(
        "RESOLUTION_SAMPLE",
        {
            "width_min": min(widths),
            "width_max": max(widths),
            "width_mean": round(float(np.mean(widths)), 2),
            "height_min": min(heights),
            "height_max": max(heights),
            "height_mean": round(float(np.mean(heights)), 2),
            "sample_n": len(widths),
        },
    )

    train["word_count"] = train["text"].str.split().str.len()
    train["char_count"] = train["text"].str.len()
    print(
        "TEXT_STATS",
        {
            "overall_word_mean": round(float(train["word_count"].mean()), 2),
            "non_hateful_word_mean": round(float(train.loc[train.label == 0, "word_count"].mean()), 2),
            "hateful_word_mean": round(float(train.loc[train.label == 1, "word_count"].mean()), 2),
            "median_word_count": int(train["word_count"].median()),
            "p95_word_count": round(float(np.percentile(train["word_count"], 95)), 2),
            "overall_char_mean": round(float(train["char_count"].mean()), 2),
        },
    )

    groups = {
        "race": ["black", "white", "asian", "african", "hispanic", "latino", "arab"],
        "gender": ["woman", "women", "man", "men", "female", "male", "girl", "boy"],
        "religion": ["muslim", "jewish", "christian", "islam", "jew", "christ", "hindu"],
        "disability": ["disabled", "retard", "autistic", "blind", "deaf"],
        "sexuality": ["gay", "lesbian", "trans", "lgbt", "queer"],
    }
    for group_name, terms in groups.items():
        pattern = r"\\b(" + "|".join(terms) + r")\\b"
        counts = {}
        for label in (0, 1):
            subset = train[train["label"] == label]
            counts[label] = int(subset["text"].str.lower().str.contains(pattern, regex=True).sum())
        print("GROUP_MENTIONS", group_name, counts)

    train["text_len"] = train["text"].str.len()
    train["has_text"] = train["text"].fillna("").str.strip().str.len() > 0
    train["is_short"] = train["word_count"] <= 2
    train["has_numbers"] = train["text"].str.contains(r"\\d", regex=True)
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 '!?.,-")
    train["has_special"] = train["text"].fillna("").apply(lambda text: any(char not in allowed for char in str(text)))
    print(
        "OCR_AUDIT",
        {
            "has_text": int(train["has_text"].sum()),
            "very_short": int(train["is_short"].sum()),
            "has_numbers": int(train["has_numbers"].sum()),
            "has_special": int(train["has_special"].sum()),
            "avg_word_count": round(float(train["word_count"].mean()), 2),
            "avg_char_count": round(float(train["text_len"].mean()), 2),
        },
    )


if __name__ == "__main__":
    main()