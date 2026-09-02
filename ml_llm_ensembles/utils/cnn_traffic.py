"""
Traffic-to-image conversion + lightweight CNN base model.

Converts raw pcap bytes into 2D grayscale images and classifies them with a
small custom CNN — the "computer vision" NIDS paradigm. Only meaningful for
pcap-derived datasets (kitsune-mirai-pcap, kitsune-mirai-pcap-flows): the
image is built from the actual wire bytes, a representation none of the other
base models (XGB fields, LLM text serialisation, TabPFN table) see.

Leakage guard: src/dst IP addresses and the IP header checksum are zeroed in
every image. In a single-capture dataset the attacker/victim IPs are a
shortcut feature that would inflate results without generalising.

Image layouts
-------------
per-packet: first 1024 bytes of the packet      -> 32 x 32
per-flow:   first 16 packets x first 64 bytes   -> 16 x 64
Both scaled to [0, 1]; missing bytes zero-padded.
"""

import numpy as np

# Bump when the architecture/training recipe changes — versions the
# cnn_probs_*.npz cache files so old results stay comparable.
CNN_VERSION = 2

PACKET_IMG_BYTES = 1024     # 32 x 32
FLOW_PKTS = 16
FLOW_PKT_BYTES = 64         # 16 x 64


def _sanitize(buf: bytes) -> bytes:
    """Zero src/dst IP + header checksum for plain Ethernet+IPv4 frames."""
    b = bytearray(buf)
    # EtherType IPv4, no VLAN tag
    if len(b) >= 34 and b[12] == 0x08 and b[13] == 0x00:
        b[24:26] = b"\x00\x00"   # IP header checksum
        b[26:34] = b"\x00" * 8   # src IP (26-29) + dst IP (30-33)
    return bytes(b)


def _pad(buf: bytes, n: int) -> np.ndarray:
    arr = np.frombuffer(buf[:n], dtype=np.uint8)
    if len(arr) < n:
        arr = np.concatenate([arr, np.zeros(n - len(arr), dtype=np.uint8)])
    return arr


def build_packet_images(pcap_path, row_ids) -> np.ndarray:
    """(len(row_ids), 1, 32, 32) float32 images for the given packet indices
    (capture order — identical to load_kitsune_mirai_pcap row order)."""
    from scapy.all import PcapReader

    wanted = set(int(i) for i in row_ids)
    grabbed: dict[int, np.ndarray] = {}
    max_id = max(wanted)
    with PcapReader(str(pcap_path)) as reader:
        for idx, pkt in enumerate(reader):
            if idx > max_id:
                break
            if idx in wanted:
                grabbed[idx] = _pad(_sanitize(bytes(pkt)), PACKET_IMG_BYTES)

    out = np.zeros((len(row_ids), PACKET_IMG_BYTES), dtype=np.float32)
    for j, i in enumerate(row_ids):
        if int(i) in grabbed:
            out[j] = grabbed[int(i)]
    return (out / 255.0).reshape(len(row_ids), 1, 32, 32)


def build_flow_images(pcap_path, row_ids) -> np.ndarray:
    """(len(row_ids), 1, 16, 64) float32 images for the given flow indices.

    Flow identity and ordering replicate features.aggregate_pcap_packets_to_flows:
    flows keyed by (src_ip, dst_ip, src_port, dst_port, protocol_int), ordered
    by first occurrence in the capture (pandas groupby sort=False semantics).
    """
    from scapy.all import IP, TCP, UDP, PcapReader

    wanted = set(int(i) for i in row_ids)
    flow_order: dict[tuple, int] = {}
    flow_bytes: dict[int, list[np.ndarray]] = {}

    with PcapReader(str(pcap_path)) as reader:
        for pkt in reader:
            if IP in pkt:
                ip = pkt[IP]
                if TCP in pkt:
                    key = (ip.src, ip.dst, pkt[TCP].sport, pkt[TCP].dport, 6)
                elif UDP in pkt:
                    key = (ip.src, ip.dst, pkt[UDP].sport, pkt[UDP].dport, 17)
                else:
                    key = (ip.src, ip.dst, 0, 0, int(ip.proto) if int(ip.proto) == 1 else 0)
            else:
                key = ("", "", 0, 0, 0)

            fid = flow_order.setdefault(key, len(flow_order))
            if fid in wanted:
                pkts = flow_bytes.setdefault(fid, [])
                if len(pkts) < FLOW_PKTS:
                    pkts.append(_pad(_sanitize(bytes(pkt)), FLOW_PKT_BYTES))

    out = np.zeros((len(row_ids), FLOW_PKTS, FLOW_PKT_BYTES), dtype=np.float32)
    for j, i in enumerate(row_ids):
        for k, row in enumerate(flow_bytes.get(int(i), [])):
            out[j, k] = row
    return (out / 255.0).reshape(len(row_ids), 1, FLOW_PKTS, FLOW_PKT_BYTES)


# ── Lightweight CNN ───────────────────────────────────────────────────────────

class TrafficCNN:
    """Small conv net with an sklearn-ish fit/predict_proba interface.

    v2 recipe: pooling along the byte axis first (the packet/sequence axis is
    preserved through the early blocks), wider channels, and early stopping on
    a stratified validation split — the v1 fixed-8-epoch square-pooling recipe
    collapsed the 16-packet axis and badly underfit flow images.
    """

    def __init__(self, epochs: int = 40, batch_size: int = 256,
                 lr: float = 1e-3, val_frac: float = 0.1, patience: int = 6,
                 random_state: int = 42, device: str | None = None,
                 verbose: bool = False):
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.val_frac = val_frac
        self.patience = patience
        self.random_state = random_state
        self.device = device
        self.verbose = verbose
        self.model = None

    def _build(self):
        import torch
        import torch.nn as nn
        torch.manual_seed(self.random_state)
        return nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d((1, 2)),                       # bytes only
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d((1, 2)),                       # bytes only
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(128, 2),
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        import copy
        import torch
        import torch.nn as nn

        dev = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._build().to(dev)
        self._dev = dev

        # Stratified validation split for early stopping
        rng = np.random.default_rng(self.random_state)
        val_idx = np.concatenate([
            rng.choice(np.where(y == c)[0],
                       size=max(1, int(len(np.where(y == c)[0]) * self.val_frac)),
                       replace=False)
            for c in np.unique(y)
        ])
        val_mask = np.zeros(len(y), dtype=bool)
        val_mask[val_idx] = True
        X_tr, y_tr = X[~val_mask], y[~val_mask]
        X_val, y_val = X[val_mask], y[val_mask]

        Xt = torch.from_numpy(np.ascontiguousarray(X_tr, dtype=np.float32))
        yt = torch.from_numpy(np.ascontiguousarray(y_tr, dtype=np.int64))
        ds = torch.utils.data.TensorDataset(Xt, yt)
        g = torch.Generator().manual_seed(self.random_state)
        dl = torch.utils.data.DataLoader(ds, batch_size=self.batch_size,
                                         shuffle=True, generator=g)

        counts = np.bincount(y_tr, minlength=2).astype(np.float64)
        weights = torch.tensor((counts.sum() / np.maximum(counts, 1)),
                               dtype=torch.float32, device=dev)
        loss_fn = nn.CrossEntropyLoss(weight=weights)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        best_val, best_state, stale = float("inf"), None, 0
        for epoch in range(self.epochs):
            self.model.train()
            total = 0.0
            for xb, yb in dl:
                xb, yb = xb.to(dev), yb.to(dev)
                opt.zero_grad()
                loss = loss_fn(self.model(xb), yb)
                loss.backward()
                opt.step()
                total += float(loss) * len(yb)

            # Validation loss for early stopping
            self.model.eval()
            with torch.no_grad():
                val_losses = []
                for i in range(0, len(X_val), 1024):
                    xb = torch.from_numpy(np.ascontiguousarray(
                        X_val[i:i + 1024], dtype=np.float32)).to(dev)
                    yb = torch.from_numpy(np.ascontiguousarray(
                        y_val[i:i + 1024], dtype=np.int64)).to(dev)
                    val_losses.append(float(loss_fn(self.model(xb), yb)) * len(yb))
                val_loss = sum(val_losses) / len(X_val)

            if self.verbose:
                print(f"    CNN epoch {epoch + 1}/{self.epochs}: "
                      f"train {total / len(ds):.4f}  val {val_loss:.4f}")
            if val_loss < best_val - 1e-4:
                best_val, stale = val_loss, 0
                best_state = copy.deepcopy(self.model.state_dict())
            else:
                stale += 1
                if stale >= self.patience:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        import torch
        self.model.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(X), 1024):
                xb = torch.from_numpy(
                    np.ascontiguousarray(X[i:i + 1024], dtype=np.float32)
                ).to(self._dev)
                out.append(torch.softmax(self.model(xb), dim=1).cpu().numpy())
        return np.concatenate(out, axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)
