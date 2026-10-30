"""
Parameter-Efficient Fine-Tuning (PEFT / LoRA) Module for Phishing Detection.

Course Rubric Alignment:
- Model size: Small open-source pretrained model (`prajjwal1/bert-tiny`, 4.4M params, 17MB).
- Method: Low-Rank Adaptation (LoRA) via Hugging Face `peft` library.
- Baseline: Raw / un-finetuned pretrained model evaluated zero-shot.
- Dataset: Target ~500-1,000 examples (uses curated 800-email balanced split).
- Comparison: Controlled comparison: Base Model vs. LoRA Fine-Tuned Model.
- Explainability: Logs trainable vs. frozen parameter count, loss trajectory, and test metrics.
"""

from __future__ import annotations

import csv
import datetime
import json
import os
import sys
import time
from typing import Dict, Any, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import BertTokenizer, BertForSequenceClassification
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

csv.field_size_limit(min(sys.maxsize, 2147483647))

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SUBSET_DIR = os.path.join(BASE_DIR, "data", "finetune_subset")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
LORA_MODEL_DIR = os.path.join(MODELS_DIR, "lora_bert_tiny")
FINETUNE_REPORT_PATH = os.path.join(REPORTS_DIR, "finetune_results.json")

MODEL_NAME = "prajjwal1/bert-tiny"
RANDOM_SEED = 42
MAX_LEN = 128
BATCH_SIZE = 16
LEARNING_RATE = 2e-4
EPOCHS = 3
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.1


def load_csv_data(file_path: str) -> Tuple[list[str], list[int]]:
    texts = []
    labels = []
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            texts.append(row.get("text", ""))
            labels.append(int(row.get("label", 0)))
    return texts, labels


class PhishingDataset(Dataset):
    """PyTorch Dataset for text tokenization."""

    def __init__(self, texts: list[str], labels: list[int], tokenizer: Any, max_len: int = MAX_LEN):
        self.texts = [str(t) for t in texts]
        self.labels = [int(l) for l in labels]
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }


def compute_eval_metrics(y_true: list[int], y_pred: list[int]) -> Dict[str, Any]:
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "confusion_matrix": {
            "tn": int(cm[0, 0]),
            "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]),
            "tp": int(cm[1, 1]),
        },
    }


def evaluate_model(model: torch.nn.Module, dataloader: DataLoader, device: torch.device) -> Tuple[Dict[str, Any], list[int], list[int]]:
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1).cpu().tolist()

            all_preds.extend(preds)
            all_labels.extend(labels.cpu().tolist())

    metrics = compute_eval_metrics(all_labels, all_preds)
    return metrics, all_labels, all_preds


def run_lora_finetuning() -> Dict[str, Any]:
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    device = torch.device("cpu")
    print(f"Executing PEFT/LoRA fine-tuning on device: {device}", flush=True)

    # Load 800-sample curated dataset
    train_path = os.path.join(SUBSET_DIR, "train_800.csv")
    val_path = os.path.join(SUBSET_DIR, "val_800.csv")
    test_path = os.path.join(SUBSET_DIR, "test_800.csv")

    if not os.path.exists(train_path):
        from src.dataset_subset import create_curated_subset
        create_curated_subset()

    train_texts, train_labels = load_csv_data(train_path)
    val_texts, val_labels = load_csv_data(val_path)
    test_texts, test_labels = load_csv_data(test_path)

    print(f"Loaded curated fine-tuning splits: Train={len(train_texts)}, Val={len(val_texts)}, Test={len(test_texts)}", flush=True)

    # Load Tokenizer & Base Model
    print(f"Loading base pretrained model: {MODEL_NAME}...", flush=True)
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    base_model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2, local_files_only=True)
    base_model.to(device)

    # Prepare DataLoaders
    train_dataset = PhishingDataset(train_texts, train_labels, tokenizer)
    val_dataset = PhishingDataset(val_texts, val_labels, tokenizer)
    test_dataset = PhishingDataset(test_texts, test_labels, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # ---------------------------------------------------------
    # 1. EVALUATE BASELINE (Pretrained Model Zero-Shot / Raw)
    # ---------------------------------------------------------
    print("\n--- Evaluating Baseline: Raw Pretrained Base Model ---", flush=True)
    base_test_metrics, _, _ = evaluate_model(base_model, test_loader, device)
    print(f"Baseline Test Accuracy: {base_test_metrics['accuracy']*100:.2f}% | F1: {base_test_metrics['f1']:.4f}", flush=True)

    # ---------------------------------------------------------
    # 2. CONFIGURE & APPLY LoRA (PEFT)
    # ---------------------------------------------------------
    print("\n--- Applying LoRA (Low-Rank Adaptation) ---", flush=True)
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["query", "value"],
        lora_dropout=LORA_DROPOUT,
        bias="none",
    )

    lora_model = get_peft_model(base_model, lora_config)
    trainable_params, all_params = lora_model.get_nb_trainable_parameters()
    trainable_pct = (trainable_params / all_params) * 100

    print(f"Trainable Parameters : {trainable_params:,} / {all_params:,} ({trainable_pct:.2f}%)", flush=True)
    print(f"Frozen Parameters    : {all_params - trainable_params:,} ({(100 - trainable_pct):.2f}%)", flush=True)

    # ---------------------------------------------------------
    # 3. FINE-TUNING LOOP
    # ---------------------------------------------------------
    optimizer = AdamW(lora_model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.CrossEntropyLoss()

    print(f"\nTraining for {EPOCHS} epochs (Batch size={BATCH_SIZE}, LR={LEARNING_RATE})...", flush=True)
    training_history = []
    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        lora_model.train()
        running_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = lora_model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        val_metrics, _, _ = evaluate_model(lora_model, val_loader, device)

        training_history.append({
            "epoch": epoch,
            "train_loss": round(epoch_loss, 4),
            "val_accuracy": val_metrics["accuracy"],
            "val_f1": val_metrics["f1"],
        })

        print(f"Epoch {epoch}/{EPOCHS} | Train Loss: {epoch_loss:.4f} | Val Acc: {val_metrics['accuracy']*100:.2f}% | Val F1: {val_metrics['f1']:.4f}", flush=True)

    total_training_time = time.time() - start_time
    print(f"LoRA fine-tuning completed in {total_training_time:.2f} seconds.", flush=True)

    # ---------------------------------------------------------
    # 4. EVALUATE FINE-TUNED MODEL ON TEST SET
    # ---------------------------------------------------------
    print("\n--- Evaluating LoRA Fine-Tuned Model on Test Set ---", flush=True)
    lora_test_metrics, _, _ = evaluate_model(lora_model, test_loader, device)
    print(f"LoRA Test Accuracy: {lora_test_metrics['accuracy']*100:.2f}% | F1: {lora_test_metrics['f1']:.4f}", flush=True)

    # ---------------------------------------------------------
    # 5. SAVE ARTIFACTS & COMPILE REPORT
    # ---------------------------------------------------------
    os.makedirs(LORA_MODEL_DIR, exist_ok=True)
    lora_model.save_pretrained(LORA_MODEL_DIR)
    tokenizer.save_pretrained(LORA_MODEL_DIR)
    print(f"Saved LoRA adapter weights and tokenizer to {LORA_MODEL_DIR}", flush=True)

    comparison_results = {
        "metadata": {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "base_model": MODEL_NAME,
            "peft_method": "LoRA (Low-Rank Adaptation)",
            "dataset": "Phishing-800-Curated",
            "dataset_split": {"train": len(train_texts), "val": len(val_texts), "test": len(test_texts)},
            "hyperparameters": {
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "max_length": MAX_LEN,
                "lora_r": LORA_R,
                "lora_alpha": LORA_ALPHA,
                "lora_dropout": LORA_DROPOUT,
                "target_modules": ["query", "value"],
            },
            "parameters_summary": {
                "total_parameters": all_params,
                "trainable_parameters": trainable_params,
                "trainable_percentage": round(trainable_pct, 4),
            },
            "training_time_seconds": round(total_training_time, 2),
        },
        "baseline_model": {
            "model_type": "Pretrained BERT-Tiny (Raw Base)",
            "test_metrics": base_test_metrics,
        },
        "lora_finetuned_model": {
            "model_type": "BERT-Tiny + LoRA Adapter",
            "test_metrics": lora_test_metrics,
        },
        "controlled_comparison": {
            "accuracy_improvement": round(lora_test_metrics["accuracy"] - base_test_metrics["accuracy"], 4),
            "f1_improvement": round(lora_test_metrics["f1"] - base_test_metrics["f1"], 4),
        },
        "training_history": training_history,
    }

    with open(FINETUNE_REPORT_PATH, "w") as f:
        json.dump(comparison_results, f, indent=2)
    print(f"Saved complete fine-tuning report to {FINETUNE_REPORT_PATH}", flush=True)

    return comparison_results


if __name__ == "__main__":
    run_lora_finetuning()
