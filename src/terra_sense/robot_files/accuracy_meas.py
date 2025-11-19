#!/usr/bin/env python3
from glob import glob
from os.path import basename, splitext
from sklearn.metrics import accuracy_score, classification_report

y_true, y_pred = [], []

for fn in glob("*_preds.txt"):
    # e.g. "sand_01_preds.txt" → true label "sand"
    label = splitext(basename(fn))[0].split("_")[0]
    with open(fn) as f:
        preds = [l.strip() for l in f if l.strip()]
    y_true += [label] * len(preds)
    y_pred += preds

print("Overall accuracy:", accuracy_score(y_true, y_pred))
print("\nClassification report:\n", classification_report(y_true, y_pred, zero_division=0))
