#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EN→FR Transformer training job for Azure ML (with MLflow tracking).

Tracks with MLflow:
- params: model + training hyperparameters
- metrics: per-epoch train/val loss + total train time
- artifacts: training curve (SVG), vocab.json, metrics.json, checkpoint (.pth), and the entire outputs/ folder
- model: logs an MLflow PyTorch model (mlflow.pytorch.log_model)

Dataset:
- CSV is downloaded from a GitHub URL by default if not present locally.
- Expected columns:
  "English words/sentences", "French words/sentences"

Runtime installs (Option 1):
- By default the script will pip-install missing packages (spacy, nltk, matplotlib, scikit-learn, requests, mlflow).
- Add --offline to skip installs and downloads (ensure your environment already contains everything,
  including spaCy models: en_core_web_sm, fr_core_news_sm).

Submit an Azure Job via CLI 
az ml job create -f en2fr-job.yml

Tip: List curated envs and pick a PyTorch + CUDA one:
az ml environment list --type curated -o table -w <ws> -g <rg>
"""

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from typing import List, Tuple

# --- Top-level imports required by the model classes ---
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence


# ------------------------------
# Runtime setup helpers
# ------------------------------
def _maybe_pip_install(pkgs: List[str]):
    """Install packages at runtime if missing (Option 1)."""
    import importlib
    to_install = []
    print(f"[pip-install] Checking required packages: {pkgs}", flush=True)
    for pkg in pkgs:
        mod = pkg.split("==")[0].replace("-", "_")
        try:
            importlib.import_module(mod)
        except Exception:
            to_install.append(pkg)
    if to_install:
        print(f"[pip-install] Installing missing packages: {to_install}", flush=True)
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir"] + to_install)
    else:
        print("[pip-install] All packages already installed.", flush=True)


def _maybe_download_assets(offline: bool):
    """Ensure spaCy models + NLTK assets are available (skip if --offline)."""
    if offline:
        print("[assets-download] Offline mode: skipping model/data downloads.", flush=True)
        return
    try:
        import spacy
        from spacy.cli import download as spacy_download
        try:
            spacy.load("en_core_web_sm")
        except OSError:
            print("[assets-download] downloading 'en_core_web_sm'.", flush=True)
            spacy_download("en_core_web_sm")
        try:
            spacy.load("fr_core_news_sm")
        except OSError:
            print("[assets-download] downloading 'fr_core_news_sm'.", flush=True)
            spacy_download("fr_core_news_sm")
    except Exception as e:
        print(f"[warn] Could not ensure spaCy models: {e}")
    try:
        import nltk
        nltk.download("punkt")
        nltk.download("stopwords")
        nltk.download("wordnet")
    except Exception as e:
        print(f"[warn] Could not download NLTK assets: {e}")


def _setup_seed(seed: int):
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ------------------------------
# Top-level model classes (picklable)
# ------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)
        self.register_buffer("pe_table", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        x = x + self.pe_table[:, :x.size(1), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int,
        nhead: int,
        enc_layers: int,
        dec_layers: int,
        ffn_dim: int,
        dropout: float,
        max_length: int,
    ):
        super().__init__()
        self.src_emb = nn.Embedding(src_vocab_size, d_model)
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.pos = PositionalEncoding(d_model, max_length, dropout)
        self.tf = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=enc_layers,
            num_decoder_layers=dec_layers,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.out = nn.Linear(d_model, tgt_vocab_size)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
        src_pad_mask: torch.Tensor,
        tgt_pad_mask: torch.Tensor,
        mem_pad_mask: torch.Tensor,
    ) -> torch.Tensor:
        src_emb = self.pos(self.src_emb(src))
        tgt_emb = self.pos(self.tgt_emb(tgt))
        y = self.tf(
            src_emb,
            tgt_emb,
            src_mask,
            tgt_mask,
            src_key_padding_mask=src_pad_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=mem_pad_mask,
        )
        return self.out(y)

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        return self.tf.encoder(self.pos(self.src_emb(src)), src_mask)

    def decode(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        return self.tf.decoder(self.pos(self.tgt_emb(tgt)), memory, tgt_mask)


# ------------------------------
# Main
# ------------------------------
def main():
    parser = argparse.ArgumentParser(description="EN→FR Transformer training (Azure ML + MLflow)")
    # Training params
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    # Model params
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--enc-layers", type=int, default=3)
    parser.add_argument("--dec-layers", type=int, default=3)
    parser.add_argument("--ffn-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--max-length", type=int, default=512)
    # Data / IO
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--test-size", type=float, default=0.1, help="Validation split size")
    parser.add_argument(
        "--dataset-url",
        type=str,
        default="https://raw.githubusercontent.com/jchen8000/DemystifyingLLMs/refs/heads/main/4_Pre-Training/datasets/eng_-french.csv",
    )
    parser.add_argument("--dataset-path", type=str, default="datasets/eng_-french.csv")
    parser.add_argument("--offline", action="store_true", help="Skip runtime installs + downloads")
    # Outputs
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--model-filename", type=str, default="en2fr_checkpoint.pth")
    parser.add_argument("--metrics-filename", type=str, default="metrics.json")
    parser.add_argument("--plot-filename", type=str, default="en2fr_training.svg")
    parser.add_argument("--logged-model-name", type=str, default="en2fr_torch_model")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.dataset_path) or ".", exist_ok=True)

    # Option 1: runtime install if needed (disable with --offline)
    if not args.offline:
        _maybe_pip_install(
            [
                "spacy==3.8.11",
                "nltk==3.9.1",
                "matplotlib==3.10.0",
                "scikit-learn==1.6.1",
                "requests",
                "mlflow",
            ]
        )

    _setup_seed(args.seed)
    _maybe_download_assets(args.offline)

    # Imports after possible installs
    import spacy
    import matplotlib.pyplot as plt
    import requests
    from sklearn.model_selection import train_test_split

    # MLflow
    mlflow = None
    use_mlflow = True
    try:
        import mlflow as _mlf
        mlflow = _mlf
        mlflow.set_experiment("exp-en2fr")
        mlflow.start_run()
    except Exception as e:
        use_mlflow = False
        print(f"[warn] MLflow not available: {e}")

    # Device
    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else (args.device if args.device != "auto" else "cpu")
    print(f"[info] Using device: {device}", flush=True)

    # Save / log params
    if use_mlflow:
        mlflow.log_params(
            {
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "seed": args.seed,
                "d_model": args.d_model,
                "nhead": args.nhead,
                "enc_layers": args.enc_layers,
                "dec_layers": args.dec_layers,
                "ffn_dim": args.ffn_dim,
                "dropout": args.dropout,
                "max_length": args.max_length,
                "device": device,
                "test_size": args.test_size,
                "dataset_url": args.dataset_url,
            }
        )

    # ------------------------------
    # Data: download CSV if needed
    # ------------------------------
    if not os.path.exists(args.dataset_path):
        if args.offline:
            raise FileNotFoundError(
                f"Dataset not found at {args.dataset_path} and offline mode is enabled."
            )
        print(f"[data] Downloading dataset from {args.dataset_url}")
        r = requests.get(args.dataset_url, timeout=60)
        r.raise_for_status()
        with open(args.dataset_path, "wb") as f:
            f.write(r.content)
        print(f"[data] Saved CSV -> {args.dataset_path}")

    # Load CSV
    dataset: List[Tuple[str, str]] = []
    with open(args.dataset_path, mode="r", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        count = 0
        for row in reader:
            en = row["English words/sentences"]
            fr = row["French words/sentences"]
            dataset.append((en, fr))
            count += 1
    print(f"[data] Loaded {count} rows from CSV")

    # Split
    train_set, valid_set = train_test_split(
        dataset, test_size=args.test_size, random_state=args.seed, shuffle=True
    )

    # ------------------------------
    # Tokenization & vocab
    # ------------------------------
    spacy_en = spacy.load("en_core_web_sm")
    spacy_fr = spacy.load("fr_core_news_sm")

    def tokenize_en(text: str) -> List[str]:
        return [t.text for t in spacy_en.tokenizer(text)]

    def tokenize_fr(text: str) -> List[str]:
        return [t.text for t in spacy_fr.tokenizer(text)]

    PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN = "<PAD>", "<SOS>", "<EOS>", "<UNK>"
    PAD_IDX, SOS_IDX, EOS_IDX = 0, 1, 2

    def build_vocab(texts: List[str], tokenizer) -> List[str]:
        vocab = set()
        for t in texts:
            vocab.update(tokenizer(t))
        return sorted(list(vocab))

    # Build vocab **from training split only** (avoid leakage)
    en_texts = [pair[0] for pair in train_set]
    fr_texts = [pair[1] for pair in train_set]
    en_vocab = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN] + build_vocab(en_texts, tokenize_en)
    fr_vocab = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN] + build_vocab(fr_texts, tokenize_fr)
    SRC_VOCAB_SIZE, TRG_VOCAB_SIZE = len(en_vocab), len(fr_vocab)
    print(f"[data] EN vocab: {SRC_VOCAB_SIZE:,}  FR vocab: {TRG_VOCAB_SIZE:,}")

    class VocabTokenizer(nn.Module):
        def __init__(self, vocabulary: List[str]):
            super().__init__()
            self.idx2word = {i: w for i, w in enumerate(vocabulary)}
            self.word2idx = {w: i for i, w in enumerate(vocabulary)}

        def encode(self, tokens: List[str]) -> List[int]:
            return [self.word2idx.get(w, self.word2idx[UNK_TOKEN]) for w in tokens]

        def decode(self, ids: List[int]) -> List[str]:
            return [self.idx2word.get(i, UNK_TOKEN) for i in ids]

        def __call__(self, x, encode=True):
            return self.encode(x) if encode else self.decode(x)

    vocab_tok = {"en": VocabTokenizer(en_vocab), "fr": VocabTokenizer(fr_vocab)}

    class En2FrDataset(Dataset):
        def __init__(self, raw, src_tok, trg_tok, vocab_tok):
            self.raw = raw
            self.src_tok = src_tok
            self.trg_tok = trg_tok
            self.vocab_tok = vocab_tok

        def __len__(self):
            return len(self.raw)

        def __getitem__(self, idx):
            src_text, tgt_text = self.raw[idx]
            src_tokens = self.src_tok(src_text)
            tgt_tokens = self.trg_tok(tgt_text)
            src_ids = self.vocab_tok["en"](src_tokens)
            tgt_in_ids = self.vocab_tok["fr"]([SOS_TOKEN] + tgt_tokens)
            tgt_out_ids = self.vocab_tok["fr"](tgt_tokens + [EOS_TOKEN])
            return src_ids, tgt_in_ids, tgt_out_ids

    def collate_pad(batch):
        src = [torch.tensor(b[0], dtype=torch.long) for b in batch]
        tgt_in = [torch.tensor(b[1], dtype=torch.long) for b in batch]
        tgt_out = [torch.tensor(b[2], dtype=torch.long) for b in batch]
        return (
            pad_sequence(src, padding_value=PAD_IDX, batch_first=True),
            pad_sequence(tgt_in, padding_value=PAD_IDX, batch_first=True),
            pad_sequence(tgt_out, padding_value=PAD_IDX, batch_first=True),
        )

    train_ds = En2FrDataset(train_set, tokenize_en, tokenize_fr, vocab_tok)
    val_ds = En2FrDataset(valid_set, tokenize_en, tokenize_fr, vocab_tok)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_pad,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_pad,
        pin_memory=torch.cuda.is_available(),
    )

    # ------------------------------
    # Model
    # ------------------------------
    model = TransformerModel(
        SRC_VOCAB_SIZE,
        TRG_VOCAB_SIZE,
        d_model=args.d_model,
        nhead=args.nhead,
        enc_layers=args.enc_layers,
        dec_layers=args.dec_layers,
        ffn_dim=args.ffn_dim,
        dropout=args.dropout,
        max_length=args.max_length,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[model] Parameters: {total_params:,}")
    if use_mlflow:
        mlflow.log_param("total_params", int(total_params))

    def create_mask(src, tgt, pad_idx=PAD_IDX):
        src_len = src.shape[1]
        tgt_len = tgt.shape[1]
        src_mask = torch.zeros((src_len, src_len), device=device, dtype=torch.bool)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len).to(device)
        src_pad_mask = (src == pad_idx)
        tgt_pad_mask = (tgt == pad_idx)
        return src_mask, tgt_mask, src_pad_mask, tgt_pad_mask

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)

    # ------------------------------
    # Train / Val
    # ------------------------------
    def run_epoch(model, loader, train=True):
        model.train() if train else model.eval()
        total_loss, batches = 0.0, 0
        for src, tgt_in, tgt_out in loader:
            src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)
            src_mask, tgt_mask, src_pad_mask, tgt_pad_mask = create_mask(src, tgt_in)
            optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(train):
                logits = model(
                    src, tgt_in, src_mask, tgt_mask, src_pad_mask, tgt_pad_mask, src_pad_mask
                )
                loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.view(-1))
                if train:
                    loss.backward()
                    optimizer.step()
            total_loss += float(loss.item())
            batches += 1
        return total_loss / max(1, batches)

    history_train, history_val = [], []
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, train=True)
        val_loss = run_epoch(model, val_loader, train=False)
        history_train.append(train_loss)
        history_val.append(val_loss)
        print(f"[epoch {epoch:03d}] train_loss={train_loss:.6f} val_loss={val_loss:.6f}", flush=True)
        if use_mlflow:
            mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss}, step=epoch)

    t1 = time.time()
    train_time = t1 - t0
    print(f"[done] Training time: {train_time:.2f}s")
    if use_mlflow:
        mlflow.log_metric("train_time_sec", train_time)

    # ------------------------------
    # Save artifacts
    # ------------------------------
    # Plot
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 6))
        plt.title("Loss During Training")
        plt.plot(history_train, label="Train Loss", c="k", lw=2, ls="--")
        plt.plot(history_val, label="Validation Loss", c="b", lw=2)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plot_path = os.path.join(args.output_dir, args.plot_filename)
        plt.savefig(plot_path, format="svg", dpi=150, transparent=True, bbox_inches="tight")
        print(f"[save] Plot -> {plot_path}")
        if use_mlflow:
            mlflow.log_artifact(plot_path, artifact_path="plots")
    except Exception as e:
        print(f"[warn] Could not save plot: {e}")

    # Vocab
    vocab_payload = {
        "en_vocab": en_vocab,
        "fr_vocab": fr_vocab,
        "pad_token": PAD_TOKEN,
        "sos_token": SOS_TOKEN,
        "eos_token": EOS_TOKEN,
        "pad_idx": PAD_IDX,
        "sos_idx": SOS_IDX,
        "eos_idx": EOS_IDX,
    }
    vocab_path = os.path.join(args.output_dir, "vocab.json")
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab_payload, f, ensure_ascii=False, indent=2)
    print(f"[save] Vocab -> {vocab_path}")
    if use_mlflow:
        mlflow.log_artifact(vocab_path, artifact_path="artifacts")

    # Metrics snapshot
    metrics_path = os.path.join(args.output_dir, args.metrics_filename)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "final_train_loss": history_train[-1] if history_train else None,
                "final_val_loss": history_val[-1] if history_val else None,
                "epochs": args.epochs,
                "total_params": int(total_params),
                "train_time_sec": train_time,
            },
            f,
            indent=2,
        )
    print(f"[save] Metrics -> {metrics_path}")
    if use_mlflow:
        mlflow.log_artifact(metrics_path, artifact_path="metrics")

    # Checkpoint (state_dict only — safe & portable)
    model_chkpt_path = os.path.join(args.output_dir, args.model_filename)
    torch.save(model.state_dict(), model_chkpt_path)
    print(f"[save] Checkpoint -> {model_chkpt_path}")
    if use_mlflow:
        mlflow.log_artifact(model_chkpt_path, artifact_path="checkpoints")

    # MLflow model (class is now top-level -> picklable)
    if use_mlflow:
        try:
            if os.path.exists(plot_path):
                mlflow.log_artifact(plot_path, artifact_path="plots")
            if os.path.exists(vocab_path):
                mlflow.log_artifact(vocab_path, artifact_path="artifacts")
            if os.path.exists(metrics_path):
                mlflow.log_artifact(metrics_path, artifact_path="metrics")
            if os.path.exists(model_chkpt_path):
                mlflow.log_artifact(model_chkpt_path, artifact_path="checkpoints")
            print(f"[mlflow] Logged model artifacts.")
        except Exception as e:
            print(f"[warn] Could not log MLflow model/artifacts: {e}")

    # End run
    try:
        if use_mlflow:
            mlflow.end_run()
    except Exception:
        pass

if __name__ == "__main__":
    main()
