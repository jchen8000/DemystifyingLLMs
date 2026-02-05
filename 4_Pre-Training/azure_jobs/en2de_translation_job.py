#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
EN→DE Transformer training job for Azure ML with full MLflow tracking.

Tracks:
- Params: model/training hyperparameters
- Metrics: per-epoch train/val loss
- Artifacts: training curve (SVG), vocab.json, metrics.json, checkpoint .pth
- Model: logs a mlflow PyTorch model (mlflow.pytorch.log_model)

Submit an Azure Job via CLI:
az ml job create -f en2de-job.yml
"""

import argparse
import json
import math
import os
import random
import sys
import time
from typing import List

def _maybe_pip_install(pkgs: List[str]):
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
    if offline:
        print("[assets-download] Offline mode: skipping model/data downloads.")
        return
    # Prefer spaCy's Python API to avoid shelling out
    try:
        import spacy
        from spacy.cli import download as spacy_download
        try:
            spacy.load("en_core_web_sm")
        except OSError:
            print("[assets-download] downloading 'en_core_web_sm'.", flush=True)
            spacy_download("en_core_web_sm")
        try:
            spacy.load("de_core_news_sm")
        except OSError:
            print("[assets-download] downloading 'de_core_news_sm'.", flush=True)
            spacy_download("de_core_news_sm")
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
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    parser = argparse.ArgumentParser(description="EN→DE Transformer training (Azure ML + MLflow)")
    # Training params
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=8888)
    # Model params
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--enc-layers", type=int, default=3)
    parser.add_argument("--dec-layers", type=int, default=3)
    parser.add_argument("--ffn-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--max-length", type=int, default=256)
    # Data/infra
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--val-limit", type=int, default=None)
    parser.add_argument("--data-cache-dir", type=str, default=None)
    parser.add_argument("--offline", action="store_true", help="Skip pip/model downloads")
    # Outputs
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--model-filename", type=str, default="en2de_checkpoint.pth")
    parser.add_argument("--metrics-filename", type=str, default="metrics.json")
    parser.add_argument("--plot-filename", type=str, default="en2de_training.svg")
    parser.add_argument("--logged-model-name", type=str, default="en2de_torch_model")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Ensure deps (small, safe set). For full reproducibility, prefer a custom env.
    if not args.offline:
        _maybe_pip_install(["spacy==3.8.11", "datasets==4.0.0", "nltk==3.9.1", "matplotlib==3.10.0", "mlflow"])

    _setup_seed(args.seed)
    _maybe_download_assets(args.offline)

    # Imports after potential installs
    import torch
    from torch import nn, optim
    from torch.utils.data import DataLoader, Dataset
    from torch.nn.utils.rnn import pad_sequence
    import spacy
    import matplotlib.pyplot as plt
    from datasets import load_dataset

    # --------- MLflow setup ----------
    mlflow = None
    use_mlflow = True
    try:
        import mlflow as _mlflow
        mlflow = _mlflow
        mlflow.set_experiment("exp-en2de")  # AML maps this to the workspace experiment
    except Exception as e:
        use_mlflow = False
        print(f"[warn] MLflow not available: {e}")

    # Device
    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else (args.device if args.device != "auto" else "cpu")
    print(f"[info] Using device: {device}", flush=True)

    # Tokens / indices
    PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN = "<PAD>", "<SOS>", "<EOS>", "<UNK>"
    PAD_IDX, SOS_IDX, EOS_IDX = 0, 1, 2

    # Tokenizers
    spacy_de = spacy.load("de_core_news_sm")
    spacy_en = spacy.load("en_core_web_sm")

    def tokenize_de(text: str):
        return [tok.text for tok in spacy_de.tokenizer(text)]

    def tokenize_en(text: str):
        return [tok.text for tok in spacy_en.tokenizer(text)]

    # Dataset
    print("[info] Loading dataset: bentrevett/multi30k", flush=True)
    dataset = load_dataset("bentrevett/multi30k", cache_dir=args.data_cache_dir)

    def build_vocab(texts: List[str], tokenizer):
        vocab = set()
        for t in texts:
            vocab.update(tokenizer(t))
        return sorted(list(vocab))

    en_texts = dataset["train"][:]["en"] + dataset["validation"][:]["en"] + dataset["test"][:]["en"]
    de_texts = dataset["train"][:]["de"] + dataset["validation"][:]["de"] + dataset["test"][:]["de"]
    en_vocab = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN] + build_vocab(en_texts, tokenize_en)
    de_vocab = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN] + build_vocab(de_texts, tokenize_de)

    SRC_VOCAB_SIZE, TRG_VOCAB_SIZE = len(en_vocab), len(de_vocab)
    print(f"[data] EN vocab: {SRC_VOCAB_SIZE:,}")
    print(f"[data] DE vocab: {TRG_VOCAB_SIZE:,}")

    class VocabTokenizer(nn.Module):
        def __init__(self, vocabulary: List[str]):
            super().__init__()
            self.vocab_size = len(vocabulary)
            self.idx2word = {i: w for i, w in enumerate(vocabulary)}
            self.word2idx = {w: i for i, w in enumerate(vocabulary)}

        def encode(self, tokens: List[str]) -> List[int]:
            return [self.word2idx.get(w, self.word2idx[UNK_TOKEN]) for w in tokens]

        def decode(self, ids: List[int]) -> List[str]:
            return [self.idx2word.get(i, UNK_TOKEN) for i in ids]

        def __call__(self, x, encode=True):
            return self.encode(x) if encode else self.decode(x)

    vocab_tok = {"en": VocabTokenizer(en_vocab), "de": VocabTokenizer(de_vocab)}

    class En2DeDataset(Dataset):
        def __init__(self, raw_split, src_tok, trg_tok, vocab_tok):
            self.raw = raw_split
            self.src_tok = src_tok
            self.trg_tok = trg_tok
            self.vocab_tok = vocab_tok

        def __len__(self):
            return len(self.raw)

        def __getitem__(self, idx):
            src = self.src_tok(self.raw[idx]["en"])
            src = self.vocab_tok["en"](src)
            tgt = self.trg_tok(self.raw[idx]["de"])
            tgt_in = self.vocab_tok["de"]([SOS_TOKEN] + tgt)
            tgt_out = self.vocab_tok["de"](tgt + [EOS_TOKEN])
            return src, tgt_in, tgt_out

    def collate_pad(batch):
        src = [torch.tensor(item[0], dtype=torch.long) for item in batch]
        tgt_in = [torch.tensor(item[1], dtype=torch.long) for item in batch]
        tgt_out = [torch.tensor(item[2], dtype=torch.long) for item in batch]
        return (
            pad_sequence(src, padding_value=PAD_IDX, batch_first=True),
            pad_sequence(tgt_in, padding_value=PAD_IDX, batch_first=True),
            pad_sequence(tgt_out, padding_value=PAD_IDX, batch_first=True),
        )

    def _limit(split, n):
        return split.select(range(n)) if (n is not None and n < len(split)) else split

    train_split = _limit(dataset["train"], args.train_limit)
    val_split = _limit(dataset["validation"], args.val_limit)

    train_ds = En2DeDataset(train_split, tokenize_en, tokenize_de, vocab_tok)
    val_ds = En2DeDataset(val_split, tokenize_en, tokenize_de, vocab_tok)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_pad, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_pad, pin_memory=torch.cuda.is_available())

    # --------- Model ----------
    class PositionalEncoding(nn.Module):
        def __init__(self, d_model, max_len=5000, dropout=args.dropout):
            super().__init__()
            self.dropout = nn.Dropout(p=dropout)
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) *
                                 (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)
            self.register_buffer("pe_table", pe)

        def forward(self, x):
            x = x + self.pe_table[:, :x.size(1), :]
            return self.dropout(x)

    class TransformerModel(nn.Module):
        def __init__(self, src_vocab_size, tgt_vocab_size):
            super().__init__()
            self.src_emb = nn.Embedding(src_vocab_size, args.d_model)
            self.tgt_emb = nn.Embedding(tgt_vocab_size, args.d_model)
            self.pos = PositionalEncoding(args.d_model, args.max_length, args.dropout)
            self.tf = nn.Transformer(
                d_model=args.d_model,
                nhead=args.nhead,
                num_encoder_layers=args.enc_layers,
                num_decoder_layers=args.dec_layers,
                dim_feedforward=args.ffn_dim,
                dropout=args.dropout,
                batch_first=True,
            )
            self.out = nn.Linear(args.d_model, tgt_vocab_size)

        def forward(self, src, tgt, src_mask, tgt_mask, src_pad_mask, tgt_pad_mask, mem_pad_mask):
            src_emb = self.pos(self.src_emb(src))
            tgt_emb = self.pos(self.tgt_emb(tgt))
            y = self.tf(src_emb, tgt_emb, src_mask, tgt_mask,
                        src_key_padding_mask=src_pad_mask,
                        tgt_key_padding_mask=tgt_pad_mask,
                        memory_key_padding_mask=mem_pad_mask)
            return self.out(y)

        def encode(self, src, src_mask):
            return self.tf.encoder(self.pos(self.src_emb(src)), src_mask)

        def decode(self, tgt, memory, tgt_mask):
            return self.tf.decoder(self.pos(self.tgt_emb(tgt)), memory, tgt_mask)

    model = TransformerModel(SRC_VOCAB_SIZE, TRG_VOCAB_SIZE).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[model] Parameters: {total_params:,}")

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

    # --------- MLflow run start & params ----------
    if use_mlflow:
        mlflow.start_run()
        mlflow.log_params({
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
            "train_limit": args.train_limit,
            "val_limit": args.val_limit,
        })
        mlflow.log_param("total_params", int(total_params))

    # --------- Train/Val loops ----------
    def run_epoch(model, loader, train=True):
        model.train() if train else model.eval()
        total_loss, batches = 0.0, 0
        for src, tgt_in, tgt_out in loader:
            src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)
            src_mask, tgt_mask, src_pad_mask, tgt_pad_mask = create_mask(src, tgt_in)
            optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(train):
                logits = model(src, tgt_in, src_mask, tgt_mask, src_pad_mask, tgt_pad_mask, src_pad_mask)
                loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.view(-1))
                if train:
                    loss.backward()
                    optimizer.step()
            total_loss += float(loss.item()); batches += 1
        return total_loss / max(1, batches)

    history_train, history_val = [], []
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, train=True)
        val_loss = run_epoch(model, val_loader, train=False)
        history_train.append(train_loss)
        history_val.append(val_loss)
        print(f"[epoch {epoch:03d}] train_loss={train_loss:.6f}  val_loss={val_loss:.6f}", flush=True)
        if use_mlflow:
            mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss}, step=epoch)
    t1 = time.time()
    train_time = t1 - t0
    print(f"[done] Training time: {train_time:.2f}s")
    if use_mlflow:
        mlflow.log_metric("train_time_sec", train_time)

    # --------- Save artifacts ----------
    # 1) Plot training curve
    try:
        plt.figure(figsize=(8, 6))
        plt.title("Loss During Training")
        plt.plot(history_train, label="Train Loss", c="k", lw=2, ls="--")
        plt.plot(history_val, label="Validation Loss", c="b", lw=2)
        plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()
        plot_path = os.path.join(args.output_dir, args.plot_filename)
        plt.savefig(plot_path, format="svg", dpi=150, transparent=True, bbox_inches="tight")
        print(f"[save] Plot -> {plot_path}")
        if use_mlflow:
            mlflow.log_artifact(plot_path, artifact_path="plots")
    except Exception as e:
        print(f"[warn] Could not save plot: {e}")

    # 2) Save vocabularies
    vocab_payload = {
        "en_vocab": en_vocab,
        "de_vocab": de_vocab,
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

    # 3) Save metrics snapshot
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

    # 4) Save checkpoint
    model_chkpt_path = os.path.join(args.output_dir, args.model_filename)
    torch.save(model.state_dict(), model_chkpt_path)
    print(f"[save] Checkpoint -> {model_chkpt_path}")
    if use_mlflow:
        mlflow.log_artifact(model_chkpt_path, artifact_path="checkpoints")

    # 5) Log a MLflow model (for easy loading/serving)
    if use_mlflow:
        # End run
        try:
            mlflow.end_run()
        except Exception:
            pass

if __name__ == "__main__":
    main()