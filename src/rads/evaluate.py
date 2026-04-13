from typing import Dict, Any, List

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from transformers import Trainer

from .data import CustomDataset


def build_eval_dataset(tokenizer, texts, labels, max_length: int = 512) -> CustomDataset:
    encodings = tokenizer(
        list(texts),
        truncation=True,
        padding=True,
        max_length=max_length,
    )
    return CustomDataset(encodings, labels)


def evaluate_classifier(
    model,
    tokenizer,
    df,
    text_col: str,
    label_col: str,
    max_length: int = 512,
) -> Dict[str, float]:
    dataset = build_eval_dataset(
        tokenizer=tokenizer,
        texts=df[text_col].tolist(),
        labels=df[label_col].tolist(),
        max_length=max_length,
    )

    trainer = Trainer(model=model)
    outputs = trainer.predict(dataset)

    logits = outputs.predictions
    labels = np.array(df[label_col].tolist())
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    preds = probs.argmax(axis=1)

    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, zero_division=0),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
    }

    try:
        metrics["roc_auc"] = roc_auc_score(labels, probs[:, 1])
    except Exception:
        metrics["roc_auc"] = float("nan")

    return metrics


def summarize_selection(selected_indices, pseudo_labels=None) -> Dict[str, Any]:
    summary = {
        "num_selected": int(len(selected_indices)),
        "selected_indices": [int(x) for x in selected_indices],
    }

    if pseudo_labels is not None and len(selected_indices) > 0:
        sel_labels = np.asarray(pseudo_labels)[selected_indices]
        pos = int((sel_labels == 1).sum())
        neg = int((sel_labels == 0).sum())
        summary["pseudo_pos"] = pos
        summary["pseudo_neg"] = neg
        summary["pseudo_pos_ratio"] = float(pos / max(len(sel_labels), 1))

    return summary
