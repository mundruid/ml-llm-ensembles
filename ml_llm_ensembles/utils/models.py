"""
Shared ML model utilities: XGBoost training and ModernBERT embedding extraction.
"""

import hashlib
from pathlib import Path

import numpy as np
from tqdm import tqdm
from xgboost import XGBClassifier

MODERNBERT_MODEL = "answerdotai/ModernBERT-base"
RANDOM_STATE = 42


def xgb_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        pass
    try:
        import subprocess
        return "cuda" if subprocess.run(["nvidia-smi"], capture_output=True, timeout=5).returncode == 0 else "cpu"
    except Exception:
        return "cpu"


def train_xgb(
    X_train: np.ndarray,
    y_train: np.ndarray,
    scale_pos_weight: float | None = None,
    random_state: int = RANDOM_STATE,
) -> XGBClassifier:
    kwargs = {}
    if scale_pos_weight is not None:
        kwargs["scale_pos_weight"] = scale_pos_weight
    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=random_state,
        eval_metric="logloss",
        device=xgb_device(),
        verbosity=0,
        **kwargs,
    )
    model.fit(X_train, y_train)
    return model


def make_tabpfn(
    random_state: int = RANDOM_STATE,
    device: str | None = None,
):
    """Unfitted TabPFN classifier (TabPFN-3 by default in tabpfn>=8).

    Requires accepting the PriorLabs license — set TABPFN_TOKEN in the
    environment for headless machines.
    Filters constructor kwargs by signature so minor API drift across
    tabpfn versions doesn't break us.
    """
    import inspect
    from tabpfn import TabPFNClassifier

    candidate_kwargs = {
        "device": device,
        "random_state": random_state,
        "ignore_pretraining_limits": True,
    }
    sig = inspect.signature(TabPFNClassifier.__init__)
    kwargs = {
        k: v for k, v in candidate_kwargs.items()
        if k in sig.parameters and v is not None
    }
    return TabPFNClassifier(**kwargs)


def train_tabpfn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = RANDOM_STATE,
    device: str | None = None,
):
    model = make_tabpfn(random_state=random_state, device=device)
    model.fit(X_train, y_train)
    return model


def _texts_fingerprint(texts: list[str]) -> str:
    h = hashlib.sha256(str(len(texts)).encode())
    for t in texts[:200]:
        h.update(t[:100].encode())
    return h.hexdigest()[:16]


def build_modernbert_features(
    train_texts: list[str],
    test_texts: list[str],
    cache_dir: Path,
    cache_filename: str = "modernbert_embeddings.npz",
    batch_size: int = 32,
    max_length: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Encode texts with ModernBERT (mean-pooled token embeddings).
    Embeddings are cached to disk under cache_dir/cache_filename.
    Returns (X_train_emb, X_test_emb).
    """
    import torch
    from transformers import AutoTokenizer, AutoModel

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / cache_filename

    # Content fingerprint guards against reusing embeddings from a different
    # split that happens to have the same sizes (e.g. a different seed).
    fingerprint = np.array([_texts_fingerprint(train_texts), _texts_fingerprint(test_texts)])

    if cache_file.exists():
        cached = np.load(cache_file)
        if (
            "X_train" in cached and "X_test" in cached
            and cached["X_train"].shape[0] == len(train_texts)
            and cached["X_test"].shape[0] == len(test_texts)
            and ("fingerprint" not in cached or (cached["fingerprint"] == fingerprint).all())
        ):
            print(f"  ModernBERT: loaded from cache ({cache_file})")
            return cached["X_train"], cached["X_test"]
        print("  ModernBERT: cache mismatch (size or content) — re-encoding.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  ModernBERT: loading {MODERNBERT_MODEL} on {device} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODERNBERT_MODEL)
    bert_model = AutoModel.from_pretrained(MODERNBERT_MODEL).to(device).eval()

    def encode(texts: list[str]) -> np.ndarray:
        all_embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="  Encoding", leave=False):
            batch = texts[i: i + batch_size]
            inputs = tokenizer(
                batch, padding=True, truncation=True,
                max_length=max_length, return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                outputs = bert_model(**inputs)
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            emb = (outputs.last_hidden_state * mask).sum(1) / mask.sum(1)
            all_embeddings.append(emb.cpu().float().numpy())
        return np.concatenate(all_embeddings, axis=0)

    print(f"  Encoding {len(train_texts)} train texts ...")
    X_train_emb = encode(train_texts)
    print(f"  Encoding {len(test_texts)} test texts ...")
    X_test_emb = encode(test_texts)

    np.savez(cache_file, X_train=X_train_emb, X_test=X_test_emb, fingerprint=fingerprint)
    print(f"  ModernBERT embeddings cached to {cache_file}")

    del bert_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return X_train_emb, X_test_emb
