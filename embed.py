"""embed.py — CLIP image embeddings for the meal photos.

    python embed.py                 # writes artifacts/clip_embeddings.npz

Encodes every resolvable meal photo with CLIP ViT-B/32 and caches the result, so
rung 3 and the serving path both read vectors rather than re-running a vision
model. The same `encode_images` function backs both, which is what keeps
training and serving from drifting apart.
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np

# cgm/pandas are imported lazily inside embed_meal_photos(). The serving path
# only needs encode_images() and load_embeddings(), and should not have to pull
# the whole data stack into its container to get them.

ARTIFACTS = Path(__file__).parent / "artifacts"
MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"
EMBED_DIM = 512
BATCH = 32

_MODEL = None
_PREPROCESS = None
_DEVICE = None


def _load():
    """Load CLIP once. Prefers Apple's MPS backend, falls back to CPU."""
    global _MODEL, _PREPROCESS, _DEVICE
    if _MODEL is None:
        import open_clip
        import torch

        _DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
        _MODEL, _, _PREPROCESS = open_clip.create_model_and_transforms(
            MODEL_NAME, pretrained=PRETRAINED
        )
        _MODEL = _MODEL.to(_DEVICE).eval()
    return _MODEL, _PREPROCESS, _DEVICE


def encode_images(images: list, *, batch_size: int = BATCH) -> np.ndarray:
    """Encode PIL images (or raw JPEG bytes) to L2-normalised CLIP vectors.

    Normalising here rather than at the call site means every consumer gets the
    same scale — an unnormalised vector reaching a model trained on normalised
    ones is silently wrong rather than an error.
    """
    import torch
    from PIL import Image

    model, preprocess, device = _load()
    out = np.zeros((len(images), EMBED_DIM), dtype=np.float32)

    for start in range(0, len(images), batch_size):
        chunk = images[start : start + batch_size]
        tensors = []
        for item in chunk:
            img = Image.open(io.BytesIO(item)) if isinstance(item, bytes) else item
            tensors.append(preprocess(img.convert("RGB")))
        batch = torch.stack(tensors).to(device)
        with torch.no_grad():
            feats = model.encode_image(batch)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        out[start : start + len(chunk)] = feats.cpu().numpy().astype(np.float32)

    return out


def embed_meal_photos(*, limit: int | None = None) -> tuple[np.ndarray, list[str]]:
    """Encode every resolvable meal photo. Returns (vectors, zip_member keys)."""
    import zipfile

    import cgm

    meals = cgm.meals(with_photo=True)
    have = meals[meals["photo_ok"]].copy()
    if limit:
        have = have.head(limit)
    members = have["zip_member"].tolist()

    print(f"encoding {len(members):,} meal photos with {MODEL_NAME}/{PRETRAINED}")
    vectors = np.zeros((len(members), EMBED_DIM), dtype=np.float32)
    with zipfile.ZipFile(cgm.fetch_photos()) as z:
        for start in range(0, len(members), BATCH):
            chunk = members[start : start + BATCH]
            blobs = [z.read(name) for name in chunk]
            vectors[start : start + len(chunk)] = encode_images(blobs)
            done = start + len(chunk)
            if done % (BATCH * 10) == 0 or done == len(members):
                print(f"  {done:>5,} / {len(members):,}")
    return vectors, members


def load_embeddings() -> tuple[np.ndarray, list[str]]:
    """Read the cached embeddings written by :func:`main`."""
    path = ARTIFACTS / "clip_embeddings.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing — run `python embed.py` first")
    blob = np.load(path, allow_pickle=False)
    return blob["vectors"], [str(k) for k in blob["keys"]]


def main() -> None:
    vectors, members = embed_meal_photos()
    ARTIFACTS.mkdir(exist_ok=True)
    path = ARTIFACTS / "clip_embeddings.npz"
    # Fixed-width unicode, not dtype=object: object arrays need pickle to read
    # back, and this file is loaded by the serving path with allow_pickle=False.
    np.savez_compressed(path, vectors=vectors, keys=np.array(members, dtype="U"))
    size_mb = path.stat().st_size / 1048576
    print(f"\nwrote {path}  {vectors.shape}  {size_mb:.1f} MB")
    print(f"model: {MODEL_NAME} / {PRETRAINED}  (device: {_DEVICE})")


if __name__ == "__main__":
    main()
