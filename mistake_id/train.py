"""Fine-tuning loop for encoder classifiers.

The defaults reproduce what the submitted notebook got from Hugging Face ``Trainer``:
AdamW (betas 0.9/0.999, eps 1e-8), weight decay 0.01 on everything except biases and
LayerNorm weights, linear decay to zero with no warm-up, gradient clipping at 1.0,
batches of 16 with the last partial batch kept, and the final epoch's weights used
for prediction. A logical batch can be split into micro-batches to fit in memory; the
loss is normalised by the whole logical batch, so the update is the same as one big
batch (up to dropout noise).

With ``aux_weight > 0`` the classifier gets 12 outputs instead of 3: one 3-way head for
each of the four annotated dimensions. Only the Mistake Identification head is used
for prediction; the other three are extra supervision.
"""
from __future__ import annotations

import math
import random
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .data import DIMENSIONS
from .encode import Encoder


@dataclass
class Config:
    name: str
    model: str = "bert-base-uncased"
    input_format: str = "notebook"
    max_len: int = 512
    head_tokens: int = 96
    lr: float = 1e-5
    epochs: int = 4
    batch_size: int = 16
    micro_batch: int = 8
    warmup_ratio: float = 0.0
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    class_weights: list | None = None
    aux_weight: float = 0.0
    seed: int = 42
    precision: str = "fp32"
    notes: str = ""
    extra: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


def device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _param_groups(model, weight_decay):
    decay, no_decay = [], []
    for module_name, module in model.named_modules():
        for name, p in module.named_parameters(recurse=False):
            if not p.requires_grad:
                continue
            is_norm = isinstance(module, torch.nn.LayerNorm) or "LayerNorm" in type(module).__name__ or "layernorm" in module_name.lower()
            (no_decay if name == "bias" or is_norm else decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


def _pad(tok, feats):
    batch = tok.pad(feats, return_tensors="pt")
    return {k: v for k, v in batch.items() if k in ("input_ids", "attention_mask", "token_type_ids")}


def _autocast(cfg, dev):
    if cfg.precision == "bf16":
        return torch.autocast(device_type=dev.type, dtype=torch.bfloat16)
    return torch.autocast(device_type=dev.type, enabled=False)


class Run:
    """One fine-tuning run: build the model, train on ``train_rows``, predict on others."""

    def __init__(self, cfg: Config, log=print):
        self.cfg, self.log, self.dev = cfg, log, device()
        set_seed(cfg.seed)
        self.tok = AutoTokenizer.from_pretrained(cfg.model)
        self.n_heads = len(DIMENSIONS) if cfg.aux_weight > 0 else 1
        self.model = AutoModelForSequenceClassification.from_pretrained(cfg.model, num_labels=3 * self.n_heads)
        if self.model.config.pad_token_id is None and self.tok.pad_token_id is not None:
            self.model.config.pad_token_id = self.tok.pad_token_id
        self.model.to(self.dev)
        self.encoder = Encoder(self.tok, cfg.input_format, cfg.max_len, cfg.head_tokens)
        self.history = []

    def _loss(self, logits, y, denom_main, denom_aux):
        cfg = self.cfg
        w = torch.tensor(cfg.class_weights, device=logits.device, dtype=torch.float32) if cfg.class_weights else None
        nll = F.cross_entropy(logits[:, :3].float(), y[:, 0], reduction="none")
        weights = w[y[:, 0]] if w is not None else torch.ones_like(nll)
        loss = (weights * nll).sum() / denom_main
        if self.n_heads > 1:
            aux = sum(F.cross_entropy(logits[:, 3 * k:3 * k + 3].float(), y[:, k], reduction="sum")
                      for k in range(1, self.n_heads))
            loss = loss + cfg.aux_weight * aux / (denom_aux * (self.n_heads - 1))
        return loss

    def train(self, rows: list[dict]):
        cfg, dev, model = self.cfg, self.dev, self.model
        feats = [self.encoder(r) for r in rows]
        labels = torch.tensor([[r["labels"][d] for d in DIMENSIONS[:self.n_heads]] for r in rows])
        w = torch.tensor(cfg.class_weights, dtype=torch.float32) if cfg.class_weights else None

        steps_per_epoch = math.ceil(len(rows) / cfg.batch_size)
        total = steps_per_epoch * cfg.epochs
        warmup = int(cfg.warmup_ratio * total)
        opt = torch.optim.AdamW(_param_groups(model, cfg.weight_decay), lr=cfg.lr, betas=(0.9, 0.999), eps=1e-8)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: s / max(1, warmup) if s < warmup else max(0.0, (total - s) / max(1, total - warmup)))
        gen = torch.Generator().manual_seed(cfg.seed)
        t0 = time.time()
        step = 0
        for epoch in range(cfg.epochs):
            model.train()
            order = torch.randperm(len(rows), generator=gen).tolist()
            run_loss, n_seen = torch.zeros((), device=dev), 0
            for b in range(0, len(order), cfg.batch_size):
                batch = order[b:b + cfg.batch_size]
                yb = labels[batch]
                denom_main = float(w[yb[:, 0]].sum()) if w is not None else float(len(batch))
                for m in range(0, len(batch), cfg.micro_batch):
                    mb = batch[m:m + cfg.micro_batch]
                    inputs = {k: v.to(dev) for k, v in _pad(self.tok, [feats[i] for i in mb]).items()}
                    with _autocast(cfg, dev):
                        logits = model(**inputs).logits
                    loss = self._loss(logits, labels[mb].to(dev), denom_main, float(len(batch)))
                    loss.backward()
                    run_loss += loss.detach() * len(batch)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                n_seen += len(batch)
            epoch_loss = float(run_loss) / n_seen
            self.history.append({"epoch": epoch + 1, "train_loss": epoch_loss, "elapsed_s": time.time() - t0})
            self.log(f"  epoch {epoch + 1}/{cfg.epochs} loss {epoch_loss:.4f} ({time.time() - t0:.0f}s)")
        self.train_seconds = time.time() - t0
        self.train_stats = dict(self.encoder.stats)

    @torch.no_grad()
    def predict(self, rows: list[dict], batch_size: int = 32) -> np.ndarray:
        """Class probabilities for every head, shape (n, n_heads, 3)."""
        model, dev = self.model, self.dev
        model.eval()
        feats = [self.encoder(r) for r in rows]
        order = np.argsort([len(f["input_ids"]) for f in feats])
        out = np.zeros((len(rows), self.n_heads, 3), dtype=np.float32)
        for b in range(0, len(order), batch_size):
            idx = order[b:b + batch_size]
            inputs = {k: v.to(dev) for k, v in _pad(self.tok, [feats[i] for i in idx]).items()}
            with _autocast(self.cfg, dev):
                logits = model(**inputs).logits.float()
            probs = torch.softmax(logits.view(len(idx), self.n_heads, 3), dim=-1)
            out[idx] = probs.cpu().numpy()
        return out

    def close(self):
        del self.model
        if self.dev.type == "mps":
            torch.mps.empty_cache()
