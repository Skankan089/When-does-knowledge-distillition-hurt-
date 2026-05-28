"""
Prepare a balanced 5-language subset from xlsum_all_train.csv
=============================================================
Samples SAMPLES_PER_LANG rows from each of the 5 target languages,
shuffles within each language, applies an 85/15 train/val split,
and saves two JSON files ready for the teacher and student trainers.

Usage:
    python prepare_xlsum.py
"""

import os, sys, json
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_mt5 import (
    DATASET_CSV, LANGUAGES, SAMPLES_PER_LANG,
    TRAIN_JSON, VAL_JSON, TEST_JSON,
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO, SEED,
    DATASET_TEXT_KEY, DATASET_SUMMARY_KEY,
)


def main():
    print(f"Loading {DATASET_CSV}  (reading only needed columns)...")
    df = pd.read_csv(
        DATASET_CSV,
        usecols=['language', DATASET_TEXT_KEY, DATASET_SUMMARY_KEY],
    )
    print(f"  Total rows in CSV : {len(df):,}")

    rng = np.random.default_rng(SEED)
    train_records: list = []
    val_records:   list = []
    test_records:  list = []

    print(f"\nSampling {SAMPLES_PER_LANG} per language  (train/val split {TRAIN_RATIO:.0%}/{1-TRAIN_RATIO:.0%}):")
    print(f"  {'Language':<15} {'Available':>10} {'Sampled':>8} {'Train':>7} {'Val':>6} {'Test':>6}")
    print(f"  {'-'*15} {'-'*10} {'-'*8} {'-'*7} {'-'*6} {'-'*6}")

    for lang in LANGUAGES:
        subset = df[df['language'] == lang].reset_index(drop=True)
        available = len(subset)
        n = min(SAMPLES_PER_LANG, available)
        if n < SAMPLES_PER_LANG:
            print(f"  WARNING: {lang} only has {available} samples (wanted {SAMPLES_PER_LANG})")

        # Sample without replacement, then shuffle
        idx = rng.choice(available, size=n, replace=False)
        idx = idx[rng.permutation(n)]
        samples = subset.iloc[idx]

        records = []
        for _, row in samples.iterrows():
            text = row[DATASET_TEXT_KEY]
            summary = row[DATASET_SUMMARY_KEY]
            # Drop rows with empty text or summary
            if not isinstance(text, str) or not isinstance(summary, str):
                continue
            if not text.strip() or not summary.strip():
                continue
            records.append({
                DATASET_TEXT_KEY:    text.strip(),
                DATASET_SUMMARY_KEY: summary.strip(),
                'language':          lang,
            })

        n_train = int(len(records) * TRAIN_RATIO)
        n_val   = int(len(records) * VAL_RATIO)
        # test gets the remainder to avoid rounding loss
        train_records.extend(records[:n_train])
        val_records.extend(records[n_train:n_train + n_val])
        test_records.extend(records[n_train + n_val:])
        n_test = len(records) - n_train - n_val

        print(f"  {lang:<15} {available:>10,} {len(records):>8} {n_train:>7} {n_val:>6} {n_test:>6}")

    # Global shuffle of each split (mixes languages)
    rng.shuffle(train_records)
    rng.shuffle(val_records)
    rng.shuffle(test_records)

    with open(TRAIN_JSON, 'w', encoding='utf-8') as f:
        json.dump(train_records, f, ensure_ascii=False, indent=2)
    with open(VAL_JSON, 'w', encoding='utf-8') as f:
        json.dump(val_records, f, ensure_ascii=False, indent=2)
    with open(TEST_JSON, 'w', encoding='utf-8') as f:
        json.dump(test_records, f, ensure_ascii=False, indent=2)

    total = len(train_records) + len(val_records) + len(test_records)
    print(f"\n  Total sampled : {total:,}")
    print(f"  Saved train   : {TRAIN_JSON}  ({len(train_records):,} records)")
    print(f"  Saved val     : {VAL_JSON}    ({len(val_records):,} records)")
    print(f"  Saved test    : {TEST_JSON}   ({len(test_records):,} records)")

    # Language distribution check
    from collections import Counter
    t_langs  = Counter(r['language'] for r in train_records)
    v_langs  = Counter(r['language'] for r in val_records)
    te_langs = Counter(r['language'] for r in test_records)
    print(f"\nLanguage distribution:")
    print(f"  {'Language':<15} {'Train':>7} {'Val':>6} {'Test':>6}")
    for lang in LANGUAGES:
        print(f"  {lang:<15} {t_langs[lang]:>7} {v_langs[lang]:>6} {te_langs[lang]:>6}")


if __name__ == '__main__':
    main()
