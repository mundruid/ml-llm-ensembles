"""
Split-local ModernBERT fine-tuning for experiment 13.
=====================================================
Every fine-tune starts from the public pretrained `answerdotai/ModernBERT-base`
and is trained entirely inside one experiment-13 split: fit() receives ONLY the
training texts/labels plus the calibration texts/labels used for epoch
selection. The interface has no test parameter at all, so test rows cannot
influence checkpoint selection even by accident; the runner scores test rows
with score() strictly after fit() returns.

Only the public pretrained base model is eligible. Task-specific checkpoints are
not reused because their training-source overlap with this corpus panel is not
established.

Checkpoint identity
  A checkpoint directory name is derived from a canonical JSON of every field
  that changes the trained artifact: experiment, regime, bundle id, text
  variant, hard-negative condition, seed, annotation-policy version, the exact
  ordered SHA-256 fingerprints of the training and calibration texts, the base
  model name, and all hyperparameters. metadata.json beside the weights stores
  the full identity plus counts, per-epoch calibration AUCPR, the selected
  epoch and library versions. Resume loads a checkpoint ONLY when the stored
  identity matches the requested one exactly; a directory whose metadata
  mismatches its name raises instead of being silently reused.

Backends
  ModernBertFT  transformers/GPU implementation with per-epoch training,
                selection on calibration AUCPR only, best state restored).
  FakeFT        deterministic, dependency-light stand-in for smoke runs and
                tests: a token-overlap scorer with the identical interface,
                checkpoint files and metadata. Never downloads anything. It
                also records exactly which texts it was shown, so the tests
                can prove train/calibration/test isolation.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

BASE_MODEL_DEFAULT = "answerdotai/ModernBERT-base"

# Test hook: every backend appends {"backend", "identity", "event", "text_shas"}
# records here so isolation tests can prove what data each call saw.
FT_CALL_LOG: list[dict] = []


def _sha_texts(texts) -> str:
    h = hashlib.sha256()
    for t in texts:
        h.update(hashlib.sha256(str(t).encode("utf-8")).digest())
    return h.hexdigest()[:16]


def _text_shas(texts) -> set[str]:
    return {hashlib.sha256(str(t).encode("utf-8")).hexdigest()[:16] for t in texts}


def checkpoint_identity(*, regime: str, bundle_id: str | None, variant: str, condition: str,
                        seed: int, policy_version: str, train_fingerprint: str,
                        cal_fingerprint: str, base_model: str, hyperparams: dict) -> tuple[str, dict]:
    """(directory name, identity dict). The name embeds a hash of the canonical
    identity JSON, so ANY differing field yields a different directory."""
    ident = {"experiment": "13", "regime": regime, "bundle_id": bundle_id, "variant": variant,
             "condition": condition, "seed": int(seed), "annotation_policy_version": policy_version,
             "train_fingerprint": train_fingerprint, "cal_fingerprint": cal_fingerprint,
             "base_model": base_model,
             "hyperparams": {k: hyperparams[k] for k in sorted(hyperparams)}}
    digest = hashlib.sha256(json.dumps(ident, sort_keys=True).encode()).hexdigest()[:12]
    name = f"{regime}.{bundle_id or 'none'}.{variant}.{condition}.s{seed}.{digest}"
    return name, ident


def accum_group_size(i: int, accum: int, n_batches: int) -> int:
    """Size of the gradient-accumulation group that batch i belongs to: `accum`
    for full groups, the remainder for a trailing partial group, so the partial
    group's loss is not underweighted by dividing by the full `accum`."""
    start = (i // accum) * accum
    return min(accum, n_batches - start)


def _lib_versions() -> dict:
    out = {}
    for lib in ("torch", "transformers", "numpy"):
        try:
            import importlib.metadata as im
            out[lib] = im.version(lib)
        except Exception:
            out[lib] = None
    return out


class _FTBase:
    """Shared checkpoint/metadata/resume logic. Subclasses implement
    _train(train_texts, y_tr, cal_texts, y_cal) -> (history, selected_epoch)
    and _score_impl(texts) -> P(phishing), plus _save/_load weights."""
    backend_name = "base"

    def __init__(self, output_root: Path, identity_kwargs: dict, hyperparams: dict, resume: bool = False):
        self.hp = dict(hyperparams)
        self.name, self.identity = checkpoint_identity(**identity_kwargs, hyperparams=self.hp)
        self.dir = Path(output_root) / self.name
        self.resume = resume
        self.fitted = False
        self.resumed = False
        self.meta: dict = {}

    # -- resume ---------------------------------------------------------------
    def _try_resume(self) -> bool:
        meta_p = self.dir / "metadata.json"
        if not meta_p.exists():
            return False
        stored = json.loads(meta_p.read_text())
        if stored.get("identity") != self.identity:
            raise RuntimeError(
                f"checkpoint {self.dir} exists but its metadata does not match the requested run "
                f"(stored identity differs). Refusing to reuse a checkpoint by name alone; delete the "
                f"directory or change the run so its identity differs.")
        if not self.resume:
            print(f"  [ft] matching checkpoint exists at {self.dir.name} but --ft-resume not set: retraining")
            return False
        self._load_weights()
        self.meta = stored
        self.fitted = True
        self.resumed = True
        print(f"  [ft] resumed {self.backend_name} checkpoint {self.dir.name} "
              f"(selected epoch {stored.get('selected_epoch')})")
        return True

    # -- public API (NOTE: no test data anywhere in this interface) -----------
    def fit(self, train_texts, train_labels, cal_texts, cal_labels):
        assert self.identity["train_fingerprint"] == _sha_texts(train_texts), \
            "identity was built from different training texts than fit() received"
        assert self.identity["cal_fingerprint"] == _sha_texts(cal_texts), \
            "identity was built from different calibration texts than fit() received"
        FT_CALL_LOG.append({"backend": self.backend_name, "identity": self.name, "event": "fit_train",
                            "text_shas": _text_shas(train_texts)})
        FT_CALL_LOG.append({"backend": self.backend_name, "identity": self.name, "event": "fit_cal",
                            "text_shas": _text_shas(cal_texts)})
        if self._try_resume():
            return self
        t0 = time.time()
        history, selected = self._train(list(map(str, train_texts)), np.asarray(train_labels).astype(int),
                                        list(map(str, cal_texts)), np.asarray(cal_labels).astype(int))
        self.fitted = True
        self.meta = {
            "identity": self.identity, "backend": self.backend_name,
            "base_model": self.identity["base_model"], "hyperparams": self.hp,
            "selected_epoch": int(selected),
            "calibration_aucpr_by_epoch": [float(x) for x in history],
            "selection_metric": "calibration AUCPR (test never seen during training/selection)",
            "n_train": int(len(train_texts)), "n_cal": int(len(cal_texts)),
            "train_fingerprint": self.identity["train_fingerprint"],
            "cal_fingerprint": self.identity["cal_fingerprint"],
            "seed": self.identity["seed"], "train_seconds": round(time.time() - t0, 2),
            "library_versions": _lib_versions(), "resumed": False,
        }
        self.dir.mkdir(parents=True, exist_ok=True)
        self._save_weights()
        (self.dir / "metadata.json").write_text(json.dumps(self.meta, indent=2))
        return self

    def attach_counts(self, counts: dict):
        """Runner adds train/cal counts by subtype and source + partition fingerprints."""
        self.meta.update(counts)
        if (self.dir / "metadata.json").exists():
            (self.dir / "metadata.json").write_text(json.dumps(self.meta, indent=2))

    def score(self, texts):
        if not self.fitted:
            raise RuntimeError("score() before fit(): test scoring must happen only after "
                               "calibration-based checkpoint selection")
        FT_CALL_LOG.append({"backend": self.backend_name, "identity": self.name, "event": "score",
                            "text_shas": _text_shas(texts)})
        return self._score_impl(list(map(str, texts)))


class FakeFT(_FTBase):
    """Deterministic token-overlap scorer with the real backend's interface,
    checkpoints and metadata. For smoke runs and tests only; never downloads."""
    backend_name = "fake"

    def _train(self, train_texts, y_tr, cal_texts, y_cal):
        from sklearn.metrics import average_precision_score
        pos_tok, neg_tok = {}, {}
        for t, y in zip(train_texts, y_tr):
            for w in set(t.lower().split()):
                (pos_tok if y == 1 else neg_tok)[w] = (pos_tok if y == 1 else neg_tok).get(w, 0) + 1
        n_pos = max(int((y_tr == 1).sum()), 1); n_neg = max(int((y_tr == 0).sum()), 1)
        self._w = {w: math.log((pos_tok.get(w, 0) + 1) / n_pos) - math.log((neg_tok.get(w, 0) + 1) / n_neg)
                   for w in set(pos_tok) | set(neg_tok)}
        history = []
        for _ in range(int(self.hp.get("epochs", 2))):     # fake epochs: identical model each time
            s = self._score_impl(cal_texts)
            history.append(average_precision_score(y_cal, s) if len(set(y_cal.tolist())) > 1 else 0.0)
        return history, int(np.argmax(history)) + 1

    def _score_impl(self, texts):
        out = np.empty(len(texts))
        for i, t in enumerate(texts):
            z = sum(self._w.get(w, 0.0) for w in set(t.lower().split()))
            out[i] = 1.0 / (1.0 + math.exp(-max(min(z, 30.0), -30.0)))
        return out

    def _save_weights(self):
        (self.dir / "fake_weights.json").write_text(json.dumps(self._w))

    def _load_weights(self):
        self._w = json.loads((self.dir / "fake_weights.json").read_text())


class ModernBertFT(_FTBase):
    """Full fine-tune of answerdotai/ModernBERT-base with two labels: per-epoch training
    with AdamW + warmup/linear decay, gradient clipping, epoch selection on
    CALIBRATION AUCPR only, best state restored before scoring. AMP (bf16) and
    gradient accumulation for A100-sized batches."""
    backend_name = "modernbert_ft"

    def _build(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(self.identity["base_model"])
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.identity["base_model"], num_labels=2,
            id2label={0: "non_phishing", 1: "phishing"},
            label2id={"non_phishing": 0, "phishing": 1}).to(self.device)

    def _loader(self, texts, labels, batch, shuffle, gen=None):
        """Lazy per-item tokenization with DYNAMIC per-batch padding: each item is
        tokenized (truncated to max_length, no padding) only when the DataLoader
        asks for it, and the collate function pads each batch to its own longest
        member. This avoids materializing a whole partition padded to 512 tokens
        (gigabytes on a large e-mail corpus) and skips wasted compute on the
        padding of short messages. Output order is preserved for score()."""
        import torch
        from torch.utils.data import Dataset, DataLoader
        tok, ml = self.tokenizer, int(self.hp["max_length"])

        class DS(Dataset):
            def __len__(self): return len(texts)
            def __getitem__(self, i):
                e = tok(texts[i], truncation=True, max_length=ml)
                return {"input_ids": e["input_ids"], "attention_mask": e["attention_mask"],
                        "labels": int(labels[i])}

        def collate(batch):
            enc = tok.pad([{k: b[k] for k in ("input_ids", "attention_mask")} for b in batch],
                          return_tensors="pt")
            enc["labels"] = torch.tensor([b["labels"] for b in batch], dtype=torch.long)
            return enc
        return DataLoader(DS(), batch_size=batch, shuffle=shuffle, generator=gen, collate_fn=collate)

    def _train(self, train_texts, y_tr, cal_texts, y_cal):
        import copy
        import random
        from sklearn.metrics import average_precision_score
        self._build()
        torch = self.torch
        seed = self.identity["seed"]
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        gen = torch.Generator().manual_seed(seed)
        hp = self.hp
        tr_loader = self._loader(train_texts, y_tr.tolist(), int(hp["batch_size"]), True, gen)
        cal_loader = self._loader(cal_texts, y_cal.tolist(), int(hp["eval_batch_size"]), False)
        opt = torch.optim.AdamW(self.model.parameters(), lr=float(hp["learning_rate"]),
                                weight_decay=float(hp["weight_decay"]))
        accum = max(int(hp["gradient_accumulation"]), 1)
        steps = math.ceil(len(tr_loader) / accum) * int(hp["epochs"])
        warm = max(steps // 10, 1)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: s / warm if s < warm else max(0.0, (steps - s) / max(1, steps - warm)))
        amp = self.device == "cuda"

        @torch.no_grad()
        def cal_probs():
            self.model.eval(); out = []
            for b in cal_loader:
                b.pop("labels")
                b = {k: v.to(self.device) for k, v in b.items()}
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                    logits = self.model(**b).logits
                out.append(torch.softmax(logits.float(), -1)[:, 1].cpu().numpy())
            return np.concatenate(out)

        history, best, best_state, patience_left = [], -1.0, None, int(hp["patience"])
        selected = 1
        for epoch in range(1, int(hp["epochs"]) + 1):
            self.model.train(); opt.zero_grad()
            n_batches = len(tr_loader)
            for i, b in enumerate(tr_loader):
                b = {k: v.to(self.device) for k, v in b.items()}
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                    loss = self.model(**b).loss / accum_group_size(i, accum, n_batches)
                loss.backward()
                if (i + 1) % accum == 0 or (i + 1) == len(tr_loader):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    opt.step(); sched.step(); opt.zero_grad()
            ap = float(average_precision_score(y_cal, cal_probs())) if len(set(y_cal.tolist())) > 1 else 0.0
            history.append(ap)
            print(f"    [ft] epoch {epoch}/{hp['epochs']} calibration AUCPR {ap:.4f}")
            if ap > best:
                best, selected = ap, epoch
                best_state = copy.deepcopy(self.model.state_dict())
                patience_left = int(hp["patience"])
            else:
                patience_left -= 1
                if patience_left <= 0:
                    print(f"    [ft] early stop after epoch {epoch} (patience {hp['patience']})")
                    break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        return history, selected

    def _score_impl(self, texts):
        torch = self.torch
        loader = self._loader(texts, [0] * len(texts), int(self.hp["eval_batch_size"]), False)
        amp = self.device == "cuda"
        out = []
        self.model.eval()
        with torch.no_grad():
            for b in loader:
                b.pop("labels")
                b = {k: v.to(self.device) for k, v in b.items()}
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                    logits = self.model(**b).logits
                out.append(torch.softmax(logits.float(), -1)[:, 1].cpu().numpy())
        return np.concatenate(out)

    def _save_weights(self):
        self.model.save_pretrained(self.dir)
        self.tokenizer.save_pretrained(self.dir)

    def _load_weights(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(self.dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.dir).to(self.device)


def make_ft_backend(backend: str, output_root, identity_kwargs: dict, hyperparams: dict, resume: bool):
    cls = {"real": ModernBertFT, "fake": FakeFT}[backend]
    return cls(Path(output_root), identity_kwargs, hyperparams, resume=resume)
