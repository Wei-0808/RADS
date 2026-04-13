"""Source-domain fine-tuning for RADS.

Fine-tunes a HuggingFace sequence-classification model on the source dataset
and saves a checkpoint that the target-selection pipeline can load.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)

from .data import CustomDataset


@dataclass
class SourceTrainConfig:
    model_name: str
    output_dir: str
    text_col: str = "order_results"
    label_col: str = "y"
    max_length: int = 512
    epochs: int = 15
    learning_rate: float = 2e-5
    batch_size: int = 8
    weight_decay: float = 0.01
    early_stopping_patience: int = 3
    seed: int = 66


def _encode(df: pd.DataFrame, tokenizer, cfg: SourceTrainConfig) -> CustomDataset:
    enc = tokenizer(
        df[cfg.text_col].astype(str).tolist(),
        truncation=True,
        padding=True,
        max_length=cfg.max_length,
    )
    return CustomDataset(enc, df[cfg.label_col].tolist())


def _compute_metrics(eval_pred):
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="macro"),
        "precision": precision_score(labels, preds, average="macro", zero_division=0),
        "recall": recall_score(labels, preds, average="macro", zero_division=0),
    }


def train_source(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    cfg: SourceTrainConfig,
) -> str:
    """Fine-tune ``cfg.model_name`` on the source dataset.

    Returns the path of the saved best checkpoint.
    """
    os.makedirs(cfg.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(cfg.model_name, num_labels=2)

    train_ds = _encode(df_train, tokenizer, cfg)
    val_ds = _encode(df_val, tokenizer, cfg)

    args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        weight_decay=cfg.weight_decay,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        seed=cfg.seed,
        report_to=[],
        logging_steps=20,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=_compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg.early_stopping_patience)],
    )

    trainer.train()

    best_dir = os.path.join(cfg.output_dir, "best")
    trainer.save_model(best_dir)
    tokenizer.save_pretrained(best_dir)
    return best_dir


def load_source_model(checkpoint_dir: str, device: str = "cpu"):
    """Load a checkpoint produced by :func:`train_source`."""
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir).to(device)
    return model, tokenizer


@dataclass
class SourceArtifacts:
    model: torch.nn.Module
    tokenizer: object
    checkpoint_dir: str


def train_source_classifier(
    train_df: pd.DataFrame,
    dev_df: pd.DataFrame,
    text_col: str,
    label_col: str,
    model_name: str,
    output_dir: str,
    max_length: int = 512,
    learning_rate: float = 2e-5,
    batch_size: int = 8,
    num_train_epochs: int = 15,
    weight_decay: float = 0.01,
    early_stopping_patience: int = 3,
    seed: int = 66,
    device: Optional[str] = None,
) -> SourceArtifacts:
    """Wrapper used by the MIMIC pipeline.

    Trains the source classifier and returns model + tokenizer ready for
    downstream MC-dropout uncertainty extraction.
    """
    cfg = SourceTrainConfig(
        model_name=model_name,
        output_dir=output_dir,
        text_col=text_col,
        label_col=label_col,
        max_length=max_length,
        epochs=num_train_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        weight_decay=weight_decay,
        early_stopping_patience=early_stopping_patience,
        seed=seed,
    )
    ckpt = train_source(train_df, dev_df, cfg)
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_source_model(ckpt, device=dev)
    return SourceArtifacts(model=model, tokenizer=tokenizer, checkpoint_dir=ckpt)
