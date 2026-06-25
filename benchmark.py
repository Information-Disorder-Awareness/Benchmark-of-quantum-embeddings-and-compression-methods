"""
VQC benchmark — v21.

Cambiamenti rispetto a v20:
──────────────────────────────────────────────────────────────────────
Flag ENCODING_METHOD: "amplitude" (default, comportamento identico a v20)
  oppure "angle" (AngleEmbedding con StronglyEntanglingLayers).

  Angle Encoding:
    qml.AngleEmbedding applica rotazioni R_{axis}(θᵢ) su ogni qubit i.
    L'input ha dimensione n_qubits (non 2^n_qubits come AmplitudeEmbedding),
    quindi la riduzione EMBED_DIM → n_qubits avviene con uno dei due metodi:

    COMPRESSION_METHOD == "pca":
        PCA a n_qubits componenti + normalizzazione in (-π/2, π/2)
        via π/2 · tanh(·/σ_train).
        var_ratio = explained variance ratio.

    COMPRESSION_METHOD == "word2ket":
        AngleWord2KetProjector — il "local factor" di Word2Ket senza
        Kronecker product:
          1. W ∈ R^{embed_dim × n_qubits}  (top-n_qubits right SV della SVD)
          2. params = (x − μ) @ W           ∈ R^{n_qubits}
          3. θᵢ = arccos(tanh(params[i]))   ∈ (0, π)
        Output: n_qubits angoli diretti per qml.AngleEmbedding.
        var_ratio = R² reconstruction proxy (pseudoinversa lineare su train).
        Nota: WORD2KET_RANK viene ignorato in angle mode (nessun Kronecker).

  ANGLE_ROTATION: asse di rotazione per qml.AngleEmbedding.
        "X" | "Y" | "Z"  (default "X").

──────────────────────────────────────────────────────────────────────
Encoding amplitude (identico a v20):
  Word2Ket Kronecker projection: R^{embed_dim} → R^{2^n_qubits}.
  WORD2KET_RANK ≥ 1 come in v20.

──────────────────────────────────────────────────────────────────────
CSV / RUN_TAG:
  "encoding"     -> "amplitude" | "angle_X" | "angle_Y" | "angle_Z"
  "n_amplitudes" -> 2^n_qubits  (amplitude) | n_qubits (angle)
  RUN_TAG include il tag di encoding per CSV separati.
"""

import copy
import csv
import random
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import pennylane as qml

# ================================================================
#  Timing infrastructure
# ================================================================
class Timings:
    """
    Raccoglie tempi wall-clock (perf_counter) per ogni stage.
    Uso:
        tm = Timings()
        with tm.record("pca_fit"):
            pca.fit_transform(X)
        elapsed = tm.elapsed("pca_fit")
    """
    def __init__(self):
        self._starts: dict[str, float] = {}
        self._ends:   dict[str, float] = {}

    def start(self, key: str) -> None:
        self._starts[key] = time.perf_counter()

    def stop(self, key: str) -> float:
        t = time.perf_counter()
        self._ends[key] = t
        return t - self._starts[key]

    @contextmanager
    def record(self, key: str):
        self.start(key)
        try:
            yield
        finally:
            self.stop(key)

    def elapsed(self, key: str) -> float:
        s = self._starts.get(key, 0.0)
        e = self._ends.get(key, time.perf_counter())
        return max(0.0, e - s)


GLOBAL_TIMING: dict[str, float] = {}

# ================================================================
#  Device setup
# ================================================================
TORCH_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch: {torch.__version__} | CUDA available: {torch.cuda.is_available()}")
if TORCH_DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)} | "
          f"capability: {torch.cuda.get_device_capability(0)} | "
          f"CUDA build: {torch.version.cuda}")
print(f"Using PyTorch device: {TORCH_DEVICE}")

# ================================================================
#  Config benchmark
# ================================================================
SEEDS        = [42, 43, 44, 45, 46]
QUBIT_RANGE  = range(4, 11)   # 4, 5, 6, 7, 8, 9, 10
N_LAYERS     = 6
N_CLASSES    = 2
BATCH_SIZE   = 64
EPOCHS       = 50
LR           = 1e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP    = 1.0
PATIENCE     = 15

# Head MLP
HIDDEN_DIM = 32
DROPOUT    = 0.1

# SBERT
SBERT_MODEL = "sentence-transformers/all-MiniLM-L6-v2" #"sentence-transformers/all-MiniLM-L6-v2"
SBERT_TAG = "all-MiniLM-L6-v2"
SBERT_BATCH_SIZE     = 64
NORMALIZE_EMBEDDINGS = True

# ---- Encoding ----
#   "amplitude" -> AmplitudeEmbedding (comportamento identico a v20)
#                  input dim = 2^n_qubits; riduzione EMBED_DIM → 2^n_qubits
#   "angle"     -> AngleEmbedding (nuovo in v21)
#                  input dim = n_qubits;   riduzione EMBED_DIM → n_qubits
ENCODING_METHOD = "angle"
ANGLE_ROTATION  = "X"       # asse di rotazione per AngleEmbedding: "X" | "Y" | "Z"

# ---- Compression (applicato in entrambe le modalità di encoding) ----
#   "pca"      -> PCA (se amplitude: 2^n_qubits componenti; se angle: n_qubits)
#   "word2ket" -> se amplitude: Word2Ket Kronecker (→ 2^n_qubits)
#                 se angle:     AngleWord2KetProjector (local factor, → n_qubits)
COMPRESSION_METHOD = "pca"
WORD2KET_RANK      = 1       # rank Kronecker; ignorato in angle mode
USE_PCA_FALLBACK   = False   # solo per amplitude+pca: forza PCA anche se non serve

# Subsampling
TRAIN_SUBSAMPLE = 5000
VAL_SUBSAMPLE   = 1000
TEST_SUBSAMPLE  = None

VERBOSE = True

# Dataset
DATASET_HF  = "stanfordnlp/sst2"
DATASET_TAG = "sst2"
TEXT_FIELD  = "sentence"

BACKEND_NAME = "default.qubit"
DIFF_METHOD  = "backprop"
print(f"Backend: {BACKEND_NAME} | diff_method={DIFF_METHOD}")

# ================================================================
#  Output paths
# ================================================================
OUT_DIR = Path("results")
OUT_DIR.mkdir(exist_ok=True)
EMBED_CACHE_DIR = OUT_DIR / "embeddings"
EMBED_CACHE_DIR.mkdir(exist_ok=True)

q_min, q_max   = min(QUBIT_RANGE), max(QUBIT_RANGE)
_backend_tag   = BACKEND_NAME.replace(".", "_")
_comp_tag      = (f"w2k_r{WORD2KET_RANK}" if COMPRESSION_METHOD == "word2ket"
                  else "pca")
# Encoding tag: amplitude_<comp> | angle<axis>_<comp>
_enc_tag       = (f"angle{ANGLE_ROTATION.lower()}_{_comp_tag}"
                  if ENCODING_METHOD == "angle"
                  else f"amplitude_{_comp_tag}")
RUN_TAG = (
    f"{ENCODING_METHOD}_stronglyent_{DATASET_TAG}_{SBERT_TAG}_v21"
    f"_q{q_min}-{q_max}_l{N_LAYERS}_n{TRAIN_SUBSAMPLE}"
    f"_{_enc_tag}_{_backend_tag}_{DIFF_METHOD}_{TORCH_DEVICE.type}"
)
CSV_PATH = OUT_DIR / f"{RUN_TAG}.csv"

# ================================================================
#  Dataset loading
# ================================================================
print(f"\nCaricamento {DATASET_HF}...")
DS = load_dataset(DATASET_HF)

TRAIN_POOL_TEXTS  = list(DS["train"][TEXT_FIELD])
TRAIN_POOL_LABELS = np.array(DS["train"]["label"])
TEST_TEXTS        = list(DS["validation"][TEXT_FIELD])
TEST_LABELS       = np.array(DS["validation"]["label"])

print(f"Train pool: {len(TRAIN_POOL_TEXTS)} | dist: {np.bincount(TRAIN_POOL_LABELS)}")
print(f"Test (=GLUE val):   {len(TEST_TEXTS)} | dist: {np.bincount(TEST_LABELS)}")

# ================================================================
#  SBERT (cached)
# ================================================================
def cache_key(split_name: str) -> Path:
    safe     = SBERT_MODEL.replace("/", "_")
    norm_tag = "norm" if NORMALIZE_EMBEDDINGS else "raw"
    return EMBED_CACHE_DIR / f"{DATASET_TAG}_{split_name}_{safe}_{norm_tag}.npy"

def get_or_compute_embeddings(texts, split_name, embedder) -> tuple[np.ndarray, float]:
    path = cache_key(split_name)
    if path.exists():
        emb = np.load(path)
        if emb.shape[0] == len(texts):
            print(f"  [{split_name}] cache hit -> {path.name}")
            return emb, 0.0
    print(f"  [{split_name}] computing embeddings...")
    t0  = time.perf_counter()
    emb = embedder.encode(
        texts, batch_size=SBERT_BATCH_SIZE, show_progress_bar=True,
        normalize_embeddings=NORMALIZE_EMBEDDINGS, convert_to_numpy=True,
    )
    elapsed = time.perf_counter() - t0
    np.save(path, emb)
    return emb, elapsed

print(f"\nSBERT model: {SBERT_MODEL}")
embedder = SentenceTransformer(SBERT_MODEL, device=str(TORCH_DEVICE))
EMB_TRAIN_POOL, GLOBAL_TIMING["embed_train_s"] = get_or_compute_embeddings(
    TRAIN_POOL_TEXTS, "train_pool", embedder)
EMB_TEST, GLOBAL_TIMING["embed_test_s"] = get_or_compute_embeddings(
    TEST_TEXTS, "test", embedder)
EMBED_DIM = EMB_TRAIN_POOL.shape[1]
print(f"Embedding dim: {EMBED_DIM}")
print(f"SBERT timing -> train: {GLOBAL_TIMING['embed_train_s']:.2f}s | "
      f"test: {GLOBAL_TIMING['embed_test_s']:.2f}s "
      f"({'cached' if GLOBAL_TIMING['embed_train_s'] == 0 else 'computed'})")
del embedder
if TORCH_DEVICE.type == "cuda":
    torch.cuda.empty_cache()

# ================================================================
#  Helpers
# ================================================================
def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if TORCH_DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(seed)

def split_train_val_disjoint(n_total, n_train, n_val, seed):
    r = np.random.default_rng(seed)
    perm = r.permutation(n_total)
    return perm[:n_train], perm[n_train:n_train + n_val]

def subsample_indices(n_total, n_sub, seed):
    if n_sub is None or n_sub >= n_total:
        return np.arange(n_total)
    r = np.random.default_rng(seed)
    return r.choice(n_total, n_sub, replace=False)

class Word2KetCompressor:
    """
    Word2Ket compression: R^{embed_dim} → R^{2^n_qubits}.

    Per rank-1:
      1. Proiezione lineare: params = (x - mean) @ W
         dove W ∈ R^{embed_dim × n_qubits} (top-n_qubits right singular vectors).
      2. Angle encoding per qubit i:
         θᵢ = arccos(tanh(params[i]))  ∈ (0, π)
         |vᵢ⟩ = [cos(θᵢ/2), sin(θᵢ/2)]
      3. Output: |ψ⟩ = |v₁⟩ ⊗ |v₂⟩ ⊗ ... ⊗ |vₙ⟩  ∈ R^{2^n}

    Per rank-r:
      r set indipendenti di n_qubits proiezioni; output = Σ_k ⊗ᵢ |vᵢ^k⟩.
      Usa le successive n_qubits righe SVD per ogni rank.

    Nota: ogni |vᵢ⟩ è un vettore reale su S¹ per costruzione (‖vᵢ‖=1).
    L'output rank-1 è sempre su S^{2^n-1} con norma = 1.
    L'output rank-r ha norma ≤ r; AmplitudeEmbedding rinormalizza.

    Confronto con la versione precedente (W_i ∈ R^{2×embed_dim} + coppia):
      Precedente: 2 proiezioni per qubit, angolo = arctan2(p₂, p₁)
      Corretta:   1 proiezione per qubit, angolo = arccos(tanh(p))
    La formulazione corretta segue word2ket_encoder.py (Laine 2026).
    """

    def __init__(self, n_qubits: int, embed_dim: int, rank: int = 1, seed: int = 42):
        self.n_qubits     = n_qubits
        self.embed_dim    = embed_dim
        self.rank         = rank
        self.n_amplitudes = 2 ** n_qubits
        self._seed        = seed
        self.mean_: np.ndarray | None   = None
        # W_list[r] ∈ R^{embed_dim × n_qubits} — proiezione per rank r
        self.W_list: list[np.ndarray]   = []

    def fit(self, X_train: np.ndarray) -> "Word2KetCompressor":
        rng = np.random.default_rng(self._seed)
        self.mean_ = X_train.mean(axis=0, keepdims=True)
        X_c = X_train - self.mean_

        # SVD: Vt ∈ R^{min(n,D) × D}
        _, _, Vt = np.linalg.svd(X_c, full_matrices=False)

        n_rows_needed = self.n_qubits * self.rank
        n_sv = Vt.shape[0]

        if n_sv < n_rows_needed:
            extra_raw = rng.standard_normal((n_rows_needed - n_sv, self.embed_dim))
            combined  = np.vstack([Vt, extra_raw])
            Q, _      = np.linalg.qr(combined.T)
            Vt_ext    = Q.T[:n_rows_needed]
        else:
            Vt_ext = Vt[:n_rows_needed]

        # Una matrice W per rank: W[r] ∈ R^{embed_dim × n_qubits}
        self.W_list = []
        for r in range(self.rank):
            rows   = Vt_ext[r * self.n_qubits : (r + 1) * self.n_qubits]  # (n_q, D)
            self.W_list.append(rows.T)   # (D, n_q)

        # Reconstruction proxy R² (via pseudoinversa lineare su training)
        Y_train   = self._transform_centered(X_c)
        A, _, _, _ = np.linalg.lstsq(Y_train, X_c, rcond=None)
        X_hat      = Y_train @ A
        ss_res     = np.linalg.norm(X_c - X_hat) ** 2
        ss_tot     = np.linalg.norm(X_c) ** 2
        self.var_ratio_ = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

        return self

    def _transform_centered(self, X_c: np.ndarray) -> np.ndarray:
        """
        X_c: (n_samples, embed_dim) — già centrato.
        Output: (n_samples, 2^n_qubits).
        """
        n      = X_c.shape[0]
        result = np.zeros((n, self.n_amplitudes), dtype=np.float64)

        for W in self.W_list:
            # params: (n, n_qubits) — un valore per qubit
            params = X_c @ W                         # (n, n_qubits)

            # Angle encoding: θᵢ = arccos(tanh(pᵢ)) ∈ (0, π)
            thetas = np.arccos(np.clip(np.tanh(params), -1 + 1e-7, 1 - 1e-7))  # (n, n_qubits)

            # Ket per ogni qubit: [cos(θᵢ/2), sin(θᵢ/2)] ∈ R², ‖·‖=1
            cos_h = np.cos(thetas / 2)   # (n, n_qubits)
            sin_h = np.sin(thetas / 2)   # (n, n_qubits)

            # Kronecker product su tutti i qubit
            kron = np.stack([cos_h[:, 0], sin_h[:, 0]], axis=1)  # (n, 2)
            for i in range(1, self.n_qubits):
                ket_i = np.stack([cos_h[:, i], sin_h[:, i]], axis=1)  # (n, 2)
                kron  = np.einsum("ni,nj->nij", kron, ket_i).reshape(n, -1)

            result += kron

        return result

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self._transform_centered(X - self.mean_)


class AngleWord2KetProjector:
    """
    Proiezione R^{embed_dim} → R^{n_qubits} per AngleEmbedding.

    Usa il "local factor" di Word2Ket (senza Kronecker product):
      1. W ∈ R^{embed_dim × n_qubits}  inizializzata dai top-n_qubits
         right singular vectors della SVD del training set centrato.
      2. params = (x − μ) @ W           ∈ R^{n_qubits}
      3. θᵢ = arccos(tanh(params[i]))   ∈ (0, π)
         Stessa funzione angolare dei ket locali in Word2KetCompressor,
         senza il Kronecker product finale.

    Output: (n_samples, n_qubits) — angoli in (0, π) per AngleEmbedding.
    var_ratio_: R² reconstruction proxy (pseudoinversa lineare su train).

    Nota: WORD2KET_RANK è ignorato in angle mode; questa classe usa rank=1
    perché non c'è Kronecker su cui sommare rank aggiuntivi.
    """

    def __init__(self, n_qubits: int, embed_dim: int, seed: int = 42):
        self.n_qubits  = n_qubits
        self.embed_dim = embed_dim
        self._seed     = seed
        self.mean_: np.ndarray | None = None
        self.W_:    np.ndarray | None = None   # (embed_dim, n_qubits)
        self.var_ratio_: float        = 0.0

    def fit(self, X_train: np.ndarray) -> "AngleWord2KetProjector":
        rng = np.random.default_rng(self._seed)
        self.mean_ = X_train.mean(axis=0, keepdims=True)
        X_c = X_train - self.mean_

        # SVD: Vt ∈ R^{min(n_samples, embed_dim) × embed_dim}
        _, _, Vt = np.linalg.svd(X_c, full_matrices=False)
        n_sv = Vt.shape[0]

        if n_sv < self.n_qubits:
            # Pochi campioni o dim ridotta: estende con vettori ortonormali random
            extra    = rng.standard_normal((self.n_qubits - n_sv, self.embed_dim))
            combined = np.vstack([Vt, extra])
            Q, _     = np.linalg.qr(combined.T)
            Vt_ext   = Q.T[:self.n_qubits]
        else:
            Vt_ext = Vt[:self.n_qubits]        # (n_qubits, embed_dim)

        self.W_ = Vt_ext.T                     # (embed_dim, n_qubits)

        # R² reconstruction proxy sul training (pre-attivazione, spazio lineare)
        params_tr = X_c @ self.W_              # (n, n_qubits)
        A, _, _, _ = np.linalg.lstsq(params_tr, X_c, rcond=None)
        X_hat = params_tr @ A
        ss_res = np.linalg.norm(X_c - X_hat) ** 2
        ss_tot = np.linalg.norm(X_c) ** 2
        self.var_ratio_ = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Restituisce angoli θᵢ = arccos(tanh(params[i])) ∈ (0, π)."""
        X_c    = X - self.mean_
        params = X_c @ self.W_                 # (n, n_qubits)
        angles = np.arccos(np.clip(np.tanh(params), -1 + 1e-7, 1 - 1e-7))
        return angles                          # (n, n_qubits)


def prepare_angle_input(X_tr, X_va, X_te, n_qubits: int, seed: int, tm: Timings):
    """
    Riduzione dimensionale per AngleEmbedding: R^{embed_dim} → R^{n_qubits}.

    COMPRESSION_METHOD == "pca":
        PCA a min(n_qubits, embed_dim) componenti.
        Normalizzazione in (-π/2, π/2) via π/2 · tanh(·/σ_train).
        Zero-padding se embed_dim < n_qubits (raro con SBERT 768d).
        var_ratio = explained variance ratio.

    COMPRESSION_METHOD == "word2ket":
        AngleWord2KetProjector: proiezione SVD + arccos(tanh) → (0, π).
        var_ratio = R² reconstruction proxy.

    Restituisce: (X_tr, X_va, X_te, var_ratio, used_compression, comp_label)
    con shape (n_samples, n_qubits) — pronto per qml.AngleEmbedding.
    """
    embed_dim = X_tr.shape[1]
    n_comp    = min(n_qubits, embed_dim)

    if COMPRESSION_METHOD == "pca":
        with tm.record("proj_fit"):
            pca    = PCA(n_components=n_comp, random_state=seed)
            X_tr_r = pca.fit_transform(X_tr)
        var_ratio = float(pca.explained_variance_ratio_.sum())
        with tm.record("proj_transform"):
            X_va_r = pca.transform(X_va)
            X_te_r = pca.transform(X_te)

        # Normalizza in (-π/2, π/2): scala con std del training, poi tanh
        scale_global = np.percentile(np.abs(X_tr_r), 95)  # o semplicemente X_tr_r.std()
        X_tr_r = np.pi * np.tanh(X_tr_r / scale_global)  # → (−π, +π)
        X_va_r = np.pi * np.tanh(X_va_r / scale_global)
        X_te_r = np.pi * np.tanh(X_te_r / scale_global)

        # Padding (solo se embed_dim < n_qubits — raro con SBERT)
        tm.start("pad")
        if n_comp < n_qubits:
            pad    = n_qubits - n_comp
            X_tr_r = np.pad(X_tr_r, ((0, 0), (0, pad)))
            X_va_r = np.pad(X_va_r, ((0, 0), (0, pad)))
            X_te_r = np.pad(X_te_r, ((0, 0), (0, pad)))
        tm.stop("pad")
        return X_tr_r, X_va_r, X_te_r, var_ratio, True, "pca"

    if COMPRESSION_METHOD == "word2ket":
        with tm.record("proj_fit"):
            proj   = AngleWord2KetProjector(
                n_qubits=n_qubits, embed_dim=embed_dim, seed=seed)
            proj.fit(X_tr)
            X_tr_r = proj.transform(X_tr)
        var_ratio = proj.var_ratio_
        with tm.record("proj_transform"):
            X_va_r = proj.transform(X_va)
            X_te_r = proj.transform(X_te)
        # Output è esattamente (n, n_qubits) — no padding
        tm.start("pad"); tm.stop("pad")
        return X_tr_r, X_va_r, X_te_r, var_ratio, True, "w2k-angle"

    raise ValueError(
        f"Unknown COMPRESSION_METHOD for angle encoding: {COMPRESSION_METHOD!r}. "
        f"Valori validi: 'pca', 'word2ket'."
    )


def prepare_amplitude_input(X_tr, X_va, X_te, n_amplitudes, seed, tm: Timings):
    """
    Riduzione dimensionale + padding per AmplitudeEmbedding.

    COMPRESSION_METHOD == "pca":
        PCA se embed_dim > n_amplitudes (o USE_PCA_FALLBACK=True), else zero-pad.
        Restituisce (X_tr, X_va, X_te, var_ratio, used_compression, label).

    COMPRESSION_METHOD == "word2ket":
        Word2Ket Kronecker projection se embed_dim > n_amplitudes, else zero-pad.
        L'output ha sempre dimensione n_amplitudes (non serve padding dopo W2K).
    """
    embed_dim      = X_tr.shape[1]
    needs_compress = embed_dim > n_amplitudes
    n_qubits       = int(np.log2(n_amplitudes))

    # ---- Zero-pad path: solo se PCA non serve E metodo != word2ket ----
    # Word2Ket mappa sempre a esattamente n_amplitudes (espansione o compressione).
    if COMPRESSION_METHOD != "word2ket" and not needs_compress and not USE_PCA_FALLBACK:
        tm.start("proj_fit");       tm.stop("proj_fit")
        tm.start("proj_transform"); tm.stop("proj_transform")
        pad = n_amplitudes - embed_dim
        tm.start("pad")
        if pad > 0:
            X_tr = np.pad(X_tr, ((0, 0), (0, pad)))
            X_va = np.pad(X_va, ((0, 0), (0, pad)))
            X_te = np.pad(X_te, ((0, 0), (0, pad)))
        tm.stop("pad")
        return X_tr, X_va, X_te, 1.0, False, "zero-pad"

    # ---- PCA path (solo quando embed_dim > n_amplitudes o USE_PCA_FALLBACK) ----
    if COMPRESSION_METHOD == "pca":
        if not needs_compress and not USE_PCA_FALLBACK:
            # embed_dim <= n_amplitudes e nessun fallback forzato: zero-pad
            tm.start("proj_fit");       tm.stop("proj_fit")
            tm.start("proj_transform"); tm.stop("proj_transform")
            pad = n_amplitudes - embed_dim
            tm.start("pad")
            if pad > 0:
                X_tr = np.pad(X_tr, ((0, 0), (0, pad)))
                X_va = np.pad(X_va, ((0, 0), (0, pad)))
                X_te = np.pad(X_te, ((0, 0), (0, pad)))
            tm.stop("pad")
            return X_tr, X_va, X_te, 1.0, False, "zero-pad"
        n_comp = min(n_amplitudes, embed_dim)
        with tm.record("proj_fit"):
            pca    = PCA(n_components=n_comp, random_state=seed)
            X_tr_r = pca.fit_transform(X_tr)
        var_ratio = float(pca.explained_variance_ratio_.sum())
        with tm.record("proj_transform"):
            X_va_r = pca.transform(X_va)
            X_te_r = pca.transform(X_te)
        tm.start("pad")
        if n_comp < n_amplitudes:
            pad    = n_amplitudes - n_comp
            X_tr_r = np.pad(X_tr_r, ((0, 0), (0, pad)))
            X_va_r = np.pad(X_va_r, ((0, 0), (0, pad)))
            X_te_r = np.pad(X_te_r, ((0, 0), (0, pad)))
        tm.stop("pad")
        return X_tr_r, X_va_r, X_te_r, var_ratio, True, "pca"

    # ---- Word2Ket path — sempre applicato (compressione E espansione) ----
    # embed_dim > n_amplitudes  -> compressione (feature reduction)
    # embed_dim < n_amplitudes  -> espansione   (feature map polinomiale grado n_qubits)
    # embed_dim == n_amplitudes -> proiezione quadrata
    if COMPRESSION_METHOD == "word2ket":
        with tm.record("proj_fit"):
            w2k = Word2KetCompressor(
                n_qubits=n_qubits, embed_dim=embed_dim,
                rank=WORD2KET_RANK, seed=seed)
            w2k.fit(X_tr)
            X_tr_r = w2k.transform(X_tr)
        var_ratio = w2k.var_ratio_          # reconstruction R² proxy
        with tm.record("proj_transform"):
            X_va_r = w2k.transform(X_va)
            X_te_r = w2k.transform(X_te)
        # No padding needed: W2K output is exactly n_amplitudes
        tm.start("pad"); tm.stop("pad")
        return X_tr_r, X_va_r, X_te_r, var_ratio, True, "word2ket"

    raise ValueError(f"Unknown COMPRESSION_METHOD: {COMPRESSION_METHOD!r}")


# ================================================================
#  Per-(seed, n_qubits) pipeline
# ================================================================
def run_one_seed(seed: int, n_qubits: int) -> dict:
    tm = Timings()
    tm.start("seed_total")
    set_all_seeds(seed)

    # ---------- Data slicing ----------
    with tm.record("data_slice"):
        train_idx, val_idx = split_train_val_disjoint(
            len(EMB_TRAIN_POOL), TRAIN_SUBSAMPLE, VAL_SUBSAMPLE, seed)
        test_idx     = subsample_indices(len(EMB_TEST), TEST_SUBSAMPLE, seed + 200)
        X_train_emb  = EMB_TRAIN_POOL[train_idx]
        train_labels = TRAIN_POOL_LABELS[train_idx]
        X_val_emb    = EMB_TRAIN_POOL[val_idx]
        val_labels   = TRAIN_POOL_LABELS[val_idx]
        X_test_emb   = EMB_TEST[test_idx]
        test_labels  = TEST_LABELS[test_idx]

    if VERBOSE:
        print(f"\n  --- seed={seed} | n_qubits={n_qubits} | backend={BACKEND_NAME} ---")
        print(f"  train: {len(train_labels)} | dist: {np.bincount(train_labels)}")
        print(f"  val:   {len(val_labels)}   | dist: {np.bincount(val_labels)}")
        print(f"  test:  {len(test_labels)}  | dist: {np.bincount(test_labels)}")

    # ---------- Pre-processing (branch su ENCODING_METHOD) ----------
    n_amplitudes = 2 ** n_qubits   # usato solo in amplitude mode; in angle = n_qubits
    tm.start("preproc_total")
    if ENCODING_METHOD == "amplitude":
        X_train, X_val, X_test, var_ratio, used_compression, comp_label = \
            prepare_amplitude_input(
                X_train_emb, X_val_emb, X_test_emb, n_amplitudes, seed, tm)
        input_dim = n_amplitudes
    else:  # "angle"
        X_train, X_val, X_test, var_ratio, used_compression, comp_label = \
            prepare_angle_input(
                X_train_emb, X_val_emb, X_test_emb, n_qubits, seed, tm)
        input_dim = n_qubits
    tm.stop("preproc_total")

    if VERBOSE:
        # Costruisci descrizione del preprocessing
        if ENCODING_METHOD == "amplitude":
            if comp_label == "zero-pad":
                mode = "zero-pad"
            elif comp_label == "pca":
                mode = f"PCA({min(n_amplitudes, EMBED_DIM)}d)+pad"
            else:
                regime = ("compression" if EMBED_DIM > n_amplitudes else
                          "expansion"   if EMBED_DIM < n_amplitudes else "square")
                mode = f"Word2Ket(rank={WORD2KET_RANK}, {regime}: {EMBED_DIM}d→{n_amplitudes}d)"
            enc_desc = "amplitude"
        else:
            if comp_label == "pca":
                n_pca = min(n_qubits, EMBED_DIM)
                mode  = (f"PCA({n_pca}d)→tanh→(-π/2,π/2)"
                         + (f"+pad→{n_qubits}d" if n_pca < n_qubits else ""))
            else:  # w2k-angle
                regime = ("compression" if EMBED_DIM > n_qubits else
                          "expansion"   if EMBED_DIM < n_qubits else "square")
                mode = f"W2K-local({regime}: {EMBED_DIM}d→{n_qubits}d)→arccos(tanh)"
            enc_desc = f"angle_{ANGLE_ROTATION}"

        print(f"  {enc_desc} prep ({mode}): in={EMBED_DIM}d -> out={input_dim}d | "
              f"var_ratio={var_ratio:.4f}")
        print(f"  preproc -> proj_fit: {tm.elapsed('proj_fit'):.3f}s | "
              f"proj_transform: {tm.elapsed('proj_transform'):.3f}s | "
              f"pad: {tm.elapsed('pad'):.4f}s | total: {tm.elapsed('preproc_total'):.3f}s")

    # ---------- Tensor upload ----------
    with tm.record("tensor_upload"):
        X_train_t = torch.tensor(X_train, dtype=torch.float32, device=TORCH_DEVICE)
        y_train_t = torch.tensor(train_labels, dtype=torch.long,    device=TORCH_DEVICE)
        X_val_t   = torch.tensor(X_val,   dtype=torch.float32, device=TORCH_DEVICE)
        X_test_t  = torch.tensor(X_test,  dtype=torch.float32, device=TORCH_DEVICE)

    g = torch.Generator(device="cpu"); g.manual_seed(seed)
    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=BATCH_SIZE, shuffle=True, generator=g)

    cw = compute_class_weight("balanced", classes=np.array([0, 1]), y=train_labels)
    class_weights = torch.tensor(cw, dtype=torch.float32, device=TORCH_DEVICE)

    if VERBOSE:
        print(f"  tensor_upload: {tm.elapsed('tensor_upload'):.4f}s | "
              f"class_weights: {cw}")

    # ---------- Circuit ----------
    # backprop: PyTorch autograd through the NumPy statevector simulation.
    with tm.record("circuit_build"):
        dev = qml.device(BACKEND_NAME, wires=n_qubits)

        @qml.qnode(dev, interface="torch", diff_method=DIFF_METHOD)
        def circuit(inputs, weights):
            if ENCODING_METHOD == "amplitude":
                qml.AmplitudeEmbedding(
                    features=inputs, wires=range(n_qubits), normalize=True)
            else:  # "angle"
                qml.AngleEmbedding(
                    features=inputs, wires=range(n_qubits), rotation=ANGLE_ROTATION)
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return (
                [qml.expval(qml.PauliZ(w)) for w in range(n_qubits)] +
                [qml.expval(qml.PauliX(w)) for w in range(n_qubits)] +
                [qml.expval(qml.PauliY(w)) for w in range(n_qubits)]
            )

    weight_shapes = {"weights": (N_LAYERS, n_qubits, 3)}
    n_obs      = 3 * n_qubits
    n_q_params = N_LAYERS * n_qubits * 3

    if VERBOSE:
        print(f"  circuit_build: {tm.elapsed('circuit_build'):.4f}s | "
              f"n_q_params={n_q_params} | diff={DIFF_METHOD} | "
              f"state_vec_size=2^{n_qubits}={2**n_qubits}")

    # ---------- Model init ----------
    with tm.record("model_init"):
        class VQCModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.qlayer = qml.qnn.TorchLayer(circuit, weight_shapes)
                self.head   = nn.Sequential(
                    nn.Linear(n_obs, HIDDEN_DIM),
                    nn.GELU(),
                    nn.Dropout(DROPOUT),
                    nn.Linear(HIDDEN_DIM, N_CLASSES),
                )
            def forward(self, x):
                return self.head(self.qlayer(x))

        model = VQCModel().to(TORCH_DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if VERBOSE:
        print(f"  model_init: {tm.elapsed('model_init'):.4f}s | "
              f"total_params={n_params} (q={n_q_params}, head={n_params-n_q_params})")

    # ---------- Warmup (first forward, triggers JIT / cuQuantum init) ----------
    if VERBOSE:
        print(f"  warmup (first forward, triggers {BACKEND_NAME} JIT)...")
    model.eval()
    with tm.record("warmup"):
        with torch.no_grad():
            _ = model(X_train_t[:1])
    if VERBOSE:
        print(f"  warmup: {tm.elapsed('warmup'):.4f}s")

    # ---------- Training ----------
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    loss_fn   = nn.CrossEntropyLoss(weight=class_weights)

    if VERBOSE:
        print(f"  training (Adam lr={LR}, cosine, weighted CE, patience={PATIENCE})")
        print(f"  {'ep':>4s} | {'lr':>9s} | {'loss':>7s} | {'tr_acc':>6s} | "
              f"{'va_acc':>6s} | {'va_f1':>6s} | {'pat':>6s} | "
              f"{'ep_s':>6s} | {'val_s':>5s}")

    best_val_f1, best_epoch, best_state = -1.0, 0, None
    patience_counter, stopped_at = 0, EPOCHS
    epoch_train_times:   list[float] = []
    val_inference_times: list[float] = []

    tm.start("train_total")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        ep_t0 = time.perf_counter()
        running_loss, n_seen = 0.0, 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss   = loss_fn(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
            n_seen += xb.size(0)
        scheduler.step()
        train_loss = running_loss / n_seen
        epoch_train_times.append(time.perf_counter() - ep_t0)

        model.eval()
        val_t0 = time.perf_counter()
        with torch.no_grad():
            train_pred = model(X_train_t).argmax(dim=1).cpu().numpy()
            val_pred   = model(X_val_t).argmax(dim=1).cpu().numpy()
        val_inference_times.append(time.perf_counter() - val_t0)

        train_acc = accuracy_score(train_labels, train_pred)
        val_acc   = accuracy_score(val_labels, val_pred)
        val_f1    = f1_score(val_labels, val_pred, average="macro", zero_division=0)

        is_best = val_f1 > best_val_f1
        if is_best:
            best_val_f1      = float(val_f1)
            best_epoch       = epoch
            best_state       = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if VERBOSE:
            lr_now = scheduler.get_last_lr()[0]
            marker = " *" if is_best else ""
            print(f"  {epoch:>3d}/{EPOCHS} | {lr_now:>9.2e} | {train_loss:>7.4f} | "
                  f"{train_acc:>6.4f} | {val_acc:>6.4f} | {val_f1:>6.4f} | "
                  f"{patience_counter:>3d}/{PATIENCE} | "
                  f"{epoch_train_times[-1]:>5.1f}s | {val_inference_times[-1]:>4.1f}s"
                  f"{marker}")

        if patience_counter >= PATIENCE:
            stopped_at = epoch
            if VERBOSE:
                print(f"  --> early stop @ epoch {epoch} "
                      f"(best @ {best_epoch}, val_f1={best_val_f1:.4f})")
            break

    tm.stop("train_total")

    # ---------- Test inference ----------
    model.load_state_dict(best_state)
    model.eval()
    with tm.record("test_inference"):
        with torch.no_grad():
            test_pred = model(X_test_t).argmax(dim=1).cpu().numpy()

    # ---------- Post-processing ----------
    with tm.record("postproc"):
        test_acc      = float(accuracy_score(test_labels, test_pred))
        test_f1_macro = float(f1_score(test_labels, test_pred, average="macro",   zero_division=0))
        test_f1_pos   = float(f1_score(test_labels, test_pred, pos_label=1, average="binary", zero_division=0))
        f1_per_class  = f1_score(test_labels, test_pred, average=None, zero_division=0, labels=[0, 1])

    if VERBOSE:
        print(f"  TEST: acc={test_acc:.4f} | f1_macro={test_f1_macro:.4f} | "
              f"f1_pos={test_f1_pos:.4f}")
        print(f"  per-class: neg={f1_per_class[0]:.3f} | pos={f1_per_class[1]:.3f}")
        print(f"  pred dist: {np.bincount(test_pred, minlength=2)} "
              f"vs true: {np.bincount(test_labels, minlength=2)}")

    # ---------- GPU cleanup ----------
    with tm.record("cleanup"):
        if TORCH_DEVICE.type == "cuda":
            del model, X_train_t, X_val_t, X_test_t, y_train_t, class_weights
            torch.cuda.empty_cache()

    tm.stop("seed_total")

    ep_arr  = np.array(epoch_train_times)
    val_arr = np.array(val_inference_times)

    if VERBOSE:
        print(f"\n  [TIMING seed={seed} q={n_qubits}]")
        print(f"    data_slice          : {tm.elapsed('data_slice'):.4f}s")
        print(f"    proj_fit ({comp_label:<9s}): {tm.elapsed('proj_fit'):.4f}s")
        print(f"    proj_transform      : {tm.elapsed('proj_transform'):.4f}s")
        print(f"    pad                 : {tm.elapsed('pad'):.4f}s")
        print(f"    preproc_total       : {tm.elapsed('preproc_total'):.4f}s")
        print(f"    tensor_upload       : {tm.elapsed('tensor_upload'):.4f}s")
        print(f"    circuit_build       : {tm.elapsed('circuit_build'):.4f}s")
        print(f"    model_init          : {tm.elapsed('model_init'):.4f}s")
        print(f"    warmup              : {tm.elapsed('warmup'):.4f}s")
        print(f"    train_total         : {tm.elapsed('train_total'):.2f}s  ({stopped_at} epochs)")
        print(f"    train_epoch mean    : {ep_arr.mean():.2f}s | "
              f"min={ep_arr.min():.2f}s | max={ep_arr.max():.2f}s")
        print(f"    val_inf_total       : {val_arr.sum():.2f}s  (mean/epoch={val_arr.mean():.2f}s)")
        print(f"    test_inference      : {tm.elapsed('test_inference'):.4f}s")
        print(f"    postproc            : {tm.elapsed('postproc'):.6f}s")
        print(f"    cleanup             : {tm.elapsed('cleanup'):.4f}s")
        print(f"    seed_total          : {tm.elapsed('seed_total'):.2f}s")

    return {
        # -- Identity --
        "seed":                       seed,
        "n_qubits":                   n_qubits,
        "n_amplitudes":               n_amplitudes if ENCODING_METHOD == "amplitude" else n_qubits,
        "n_layers":                   N_LAYERS,
        "n_params":                   n_params,
        "n_q_params":                 n_q_params,
        # -- Config --
        "encoding":                   ("amplitude"
                                       if ENCODING_METHOD == "amplitude"
                                       else f"angle_{ANGLE_ROTATION}"),
        "ansatz":                     "stronglyentangling",
        "diff_method":                DIFF_METHOD,
        "backend":                    BACKEND_NAME,
        "shots":                      "None (statevector)",
        "sbert_model":                SBERT_MODEL,
        "embed_dim":                  EMBED_DIM,
        "torch_device":               str(TORCH_DEVICE),
        "compression_method":         comp_label,
        "w2k_rank":                   (WORD2KET_RANK
                                       if comp_label == "word2ket"
                                       else 0),
        "preproc":                    (f"sbert+{comp_label}"
                                       if ENCODING_METHOD == "amplitude"
                                       else f"sbert+{comp_label}+angle_{ANGLE_ROTATION}"),
        "loss":                       "weighted_ce",
        "optimizer":                  "adam",
        "scheduler":                  "cosine",
        "dataset":                    DATASET_TAG,
        "train_size":                 len(train_labels),
        # -- Results --
        "var_ratio_pca":              round(var_ratio, 6),
        "best_epoch":                 best_epoch,
        "stopped_at":                 stopped_at,
        "val_f1_best":                round(best_val_f1, 6),
        "test_acc":                   round(test_acc, 6),
        "test_f1_macro":              round(test_f1_macro, 6),
        "test_f1_pos":                round(test_f1_pos, 6),
        "f1_negative":                round(float(f1_per_class[0]), 6),
        "f1_positive":                round(float(f1_per_class[1]), 6),
        # -- Stage timings (seconds) --
        "t_data_slice_s":             round(tm.elapsed("data_slice"),        4),
        "t_proj_fit_s":               round(tm.elapsed("proj_fit"),          4),
        "t_proj_transform_s":         round(tm.elapsed("proj_transform"),    4),
        "t_pad_s":                    round(tm.elapsed("pad"),               4),
        "t_preproc_total_s":          round(tm.elapsed("preproc_total"),     4),
        "t_tensor_upload_s":          round(tm.elapsed("tensor_upload"),     4),
        "t_circuit_build_s":          round(tm.elapsed("circuit_build"),     4),
        "t_model_init_s":             round(tm.elapsed("model_init"),        4),
        "t_warmup_s":                 round(tm.elapsed("warmup"),            4),
        "t_train_total_s":            round(tm.elapsed("train_total"),       4),
        "t_train_epoch_mean_s":       round(float(ep_arr.mean()),            4),
        "t_train_epoch_min_s":        round(float(ep_arr.min()),             4),
        "t_train_epoch_max_s":        round(float(ep_arr.max()),             4),
        "t_val_inference_total_s":    round(float(val_arr.sum()),            4),
        "t_val_inference_mean_s":     round(float(val_arr.mean()),           4),
        "t_test_inference_s":         round(tm.elapsed("test_inference"),    4),
        "t_postproc_s":               round(tm.elapsed("postproc"),          6),
        "t_cleanup_s":                round(tm.elapsed("cleanup"),           4),
        "t_seed_total_s":             round(tm.elapsed("seed_total"),        4),
        # -- Global SBERT (0 if cache hit) --
        "t_embed_train_s":            round(GLOBAL_TIMING["embed_train_s"],  4),
        "t_embed_test_s":             round(GLOBAL_TIMING["embed_test_s"],   4),
    }


# ================================================================
#  Aggregation helper
# ================================================================
def agg(key: str, results: list[dict]) -> tuple[float, float]:
    vals = np.array([r[key] for r in results], dtype=float)
    std  = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
    return float(vals.mean()), std


# ================================================================
#  Main loop: qubit sweep x seed sweep
# ================================================================
total_runs = len(list(QUBIT_RANGE)) * len(SEEDS)
print(f"\n{'='*72}")
print(f"  v21 | backend={BACKEND_NAME} | diff={DIFF_METHOD} | shots=None (statevector)")
print(f"  encoding={ENCODING_METHOD.upper()}"
      + (f" rotation={ANGLE_ROTATION}" if ENCODING_METHOD == "angle" else ""))
print(f"  compression={COMPRESSION_METHOD.upper()}"
      + (f" rank={WORD2KET_RANK}" if COMPRESSION_METHOD == "word2ket"
                                     and ENCODING_METHOD == "amplitude" else "")
      + (" [rank ignored in angle mode]"
         if COMPRESSION_METHOD == "word2ket" and ENCODING_METHOD == "angle" else ""))
print(f"  SBERT={SBERT_MODEL} ({EMBED_DIM}d)")
print(f"  qubits={list(QUBIT_RANGE)} | layers={N_LAYERS} | seeds={SEEDS}")
print(f"  total runs={total_runs} | train={TRAIN_SUBSAMPLE} | val={VAL_SUBSAMPLE}")
print(f"  output -> {CSV_PATH}")
print(f"{'='*72}")

t_start_global = time.perf_counter()
all_results: list[dict] = []
csv_header_written = False

for n_qubits in QUBIT_RANGE:
    qubit_results: list[dict] = []
    n_amp     = 2 ** n_qubits
    target_d  = n_qubits if ENCODING_METHOD == "angle" else n_amp

    if ENCODING_METHOD == "amplitude":
        if COMPRESSION_METHOD == "word2ket":
            regime = ("compression" if EMBED_DIM > n_amp else
                      "expansion"   if EMBED_DIM < n_amp else "square")
            preproc_lbl = f"W2K(r={WORD2KET_RANK},{regime})"
        elif EMBED_DIM > n_amp:
            preproc_lbl = "PCA"
        else:
            preproc_lbl = "zero-pad"
    else:  # angle
        if COMPRESSION_METHOD == "word2ket":
            regime = ("compression" if EMBED_DIM > n_qubits else
                      "expansion"   if EMBED_DIM < n_qubits else "square")
            preproc_lbl = f"W2K-local({regime})+arccos"
        else:
            preproc_lbl = f"PCA({min(n_qubits, EMBED_DIM)}d)+tanh"

    enc_label = ("AmplitudeEmbedding"
                 if ENCODING_METHOD == "amplitude"
                 else f"AngleEmbedding(R{ANGLE_ROTATION})")

    print(f"\n{'='*72}")
    print(f"  QUBIT={n_qubits} | target_dim={target_d} | "
          f"{EMBED_DIM}d -> {target_d}d via {preproc_lbl} | "
          f"enc={enc_label} | diff={DIFF_METHOD}")
    print(f"{'='*72}")

    for i, seed in enumerate(SEEDS, 1):
        run_idx = (n_qubits - min(QUBIT_RANGE)) * len(SEEDS) + i
        print(f"\n>>> [{run_idx}/{total_runs}] n_qubits={n_qubits} seed={seed} "
              f"({i}/{len(SEEDS)} for this qubit)")

        res = run_one_seed(seed, n_qubits)
        all_results.append(res)
        qubit_results.append(res)
        print(f"  seed_total: {res['t_seed_total_s']:.1f}s | "
              f"train_total: {res['t_train_total_s']:.1f}s | "
              f"epoch_mean: {res['t_train_epoch_mean_s']:.2f}s")

        with CSV_PATH.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(res.keys()))
            if not csv_header_written:
                writer.writeheader()
                csv_header_written = True
            writer.writerow(res)

    # Per-qubit aggregate
    print(f"\n--- Aggregate q={n_qubits} (n={len(qubit_results)} seeds) ---")
    perf_keys   = ["test_acc", "test_f1_macro", "test_f1_pos",
                   "f1_negative", "f1_positive", "val_f1_best",
                   "var_ratio_pca", "best_epoch", "stopped_at"]
    timing_keys = ["t_proj_fit_s", "t_proj_transform_s", "t_preproc_total_s",
                   "t_tensor_upload_s", "t_circuit_build_s", "t_model_init_s",
                   "t_warmup_s", "t_train_total_s", "t_train_epoch_mean_s",
                   "t_val_inference_total_s", "t_test_inference_s",
                   "t_postproc_s", "t_seed_total_s"]
    print(f"  {'metric':<28s} {'mean':>10s} {'std':>10s}")
    for k in perf_keys + timing_keys:
        m, s = agg(k, qubit_results)
        unit = " s" if k.startswith("t_") else "  "
        print(f"  {k:<28s} {m:>10.4f} {s:>10.4f}{unit}")

    m_acc, s_acc = agg("test_acc",      qubit_results)
    m_f1,  s_f1  = agg("test_f1_macro", qubit_results)

    if ENCODING_METHOD == "amplitude":
        if EMBED_DIM <= n_amp:
            ptag = "SBERT+zero-pad"
        elif COMPRESSION_METHOD == "word2ket":
            ptag = f"SBERT+W2K(r={WORD2KET_RANK})"
        else:
            ptag = "SBERT+PCA+pad"
    else:  # angle
        if COMPRESSION_METHOD == "word2ket":
            ptag = f"SBERT+W2K-local+angle{ANGLE_ROTATION}"
        else:
            ptag = f"SBERT+PCA+tanh+angle{ANGLE_ROTATION}"

    print(f"\n  LaTeX row (q={n_qubits}):")
    print(f"  {enc_label} & StronglyEntanglingLayers & {ptag} & "
          f"{DATASET_TAG.upper()} & {n_qubits} & {N_LAYERS} & {TRAIN_SUBSAMPLE} & "
          f"{m_acc:.3f}$\\pm${s_acc:.3f} & {m_f1:.3f}$\\pm${s_f1:.3f} \\\\")

t_total = time.perf_counter() - t_start_global
print(f"\n{'='*72}")
print(f"Total wall-clock: {t_total:.1f}s ({t_total/60:.1f}min)")
print(f"Results -> {CSV_PATH.resolve()}")

# ================================================================
#  Global summary table
# ================================================================
print(f"\n=== Global summary — all qubits | {BACKEND_NAME} + {DIFF_METHOD} ===")
print(f"\n{'q':>4s} | {'f1_macro':>16s} | {'test_acc':>16s} | "
      f"{'warmup_s':>9s} | {'ep_mean_s':>10s} | {'seed_tot_s':>11s} | {'preproc':>12s}")
for nq in QUBIT_RANGE:
    sub = [r for r in all_results if r["n_qubits"] == nq]
    if not sub:
        continue
    m_f1,  s_f1  = agg("test_f1_macro",       sub)
    m_acc, s_acc = agg("test_acc",             sub)
    m_wu,  _     = agg("t_warmup_s",           sub)
    m_ep,  _     = agg("t_train_epoch_mean_s", sub)
    m_tot, _     = agg("t_seed_total_s",       sub)
    if ENCODING_METHOD == "amplitude":
        if EMBED_DIM > 2**nq:
            ptag = f"W2K(r={WORD2KET_RANK})" if COMPRESSION_METHOD == "word2ket" else "PCA"
        else:
            ptag = "pad"
    else:  # angle
        if COMPRESSION_METHOD == "word2ket":
            ptag = f"W2K-local+R{ANGLE_ROTATION}"
        else:
            ptag = f"PCA+tanh+R{ANGLE_ROTATION}"
    print(f"{nq:>4d} | {m_f1:.4f}+/-{s_f1:.4f} | "
          f"{m_acc:.4f}+/-{s_acc:.4f} | "
          f"{m_wu:>9.2f} | {m_ep:>10.2f} | {m_tot:>11.1f} | {ptag:>12s}")