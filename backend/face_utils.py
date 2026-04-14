import os
import io
import sys
import hashlib
import pickle
import logging
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance

# Suppress TF/DeepFace noise — set BEFORE importing DeepFace
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

IS_WINDOWS = sys.platform == "win32"
from deepface import DeepFace

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("FaceFind")

# ── Constants ──────────────────────────────────────────────────────────────────
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

MODEL_NAME = "Facenet512"          # Best accuracy/speed tradeoff
MIN_FACE_PX = 20                   # Minimum face dimension to accept

# Detectors tried in order for event images (retinaface is most accurate)
DETECTORS = ["retinaface", "mtcnn", "opencv"]

# Group photo threshold: run edge crops when this many faces found
GROUP_PHOTO_THRESHOLD = 3

# Cache dir for pre-computed event image embeddings
CACHE_DIR = Path("embeddings_cache")
CACHE_DIR.mkdir(exist_ok=True)

# Temp dir: use system temp on Windows to avoid permission issues
TMP_DIR = Path(tempfile.gettempdir()) if IS_WINDOWS else Path("/tmp")


# ── Utility ────────────────────────────────────────────────────────────────────

def _image_hash(img_path: str) -> str:
    """SHA256 of file bytes — used as cache key."""
    h = hashlib.sha256()
    with open(img_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def bytes_to_image_path(image_bytes: bytes, tmp_path: str) -> str:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img.save(tmp_path, quality=95)
    return tmp_path


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ── Detection helpers ──────────────────────────────────────────────────────────

def _represent(img_path: str, detector: str, enforce: bool = False) -> list:
    """Single DeepFace.represent call; returns list of face dicts or []."""
    try:
        return DeepFace.represent(
            img_path=img_path,
            model_name=MODEL_NAME,
            detector_backend=detector,
            enforce_detection=enforce,
            align=True,
        ) or []
    except Exception:
        return []


def _detect_with_detectors(img_path: str, enforce: bool = False) -> list:
    """Try each detector in priority order, return first non-empty result."""
    for det in DETECTORS:
        faces = _represent(img_path, det, enforce)
        if faces:
            return faces
    return []


def _deduplicated_add(
    all_faces: list,
    seen: set,
    new_faces: list,
    x_offset: int = 0,
    y_offset: int = 0,
    scale: float = 1.0,
    grid: int = 30,
):
    """Add faces to all_faces, deduplicating by grid-snapped original position."""
    for face in new_faces:
        area = face.get("facial_area", {})
        if area.get("w", 999) < MIN_FACE_PX or area.get("h", 999) < MIN_FACE_PX:
            continue
        ox = round((area.get("x", 0) + x_offset) / scale / grid) * grid
        oy = round((area.get("y", 0) + y_offset) / scale / grid) * grid
        key = (ox, oy)
        if key not in seen:
            seen.add(key)
            all_faces.append(face)


def detect_faces_multi_pass(img_path: str) -> list:
    """
    Multi-pass face detection for event images:
      Pass 1 — full image with all detectors
      Pass 2 — if group photo: left / right / top-center crops
      Pass 3 — if still 0 faces: 1.5× upscale fallback
    Returns deduplicated face list.
    """
    all_faces: list = []
    seen: set = set()

    # ── Pass 1: full image ─────────────────────────────────────────────────────
    faces = _detect_with_detectors(img_path)
    _deduplicated_add(all_faces, seen, faces)
    log.debug(f"  Pass 1 (full): {len(faces)}")

    # ── Pass 2: edge crops for group photos ────────────────────────────────────
    if len(all_faces) >= GROUP_PHOTO_THRESHOLD:
        log.debug(f"  Group ({len(all_faces)} faces) — edge crops...")
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        tmp = img_path + "_crop"

        crops = [
            # (save_path, box, x_offset, y_offset)
            (tmp + "_L.jpg",  (0,            0, int(w * 0.55), h), 0,            0),
            (tmp + "_R.jpg",  (int(w * 0.45), 0, w,            h), int(w * 0.45), 0),
            (tmp + "_TC.jpg", (int(w * 0.2), 0, int(w * 0.8), int(h * 0.6)), int(w * 0.2), 0),
        ]
        for path, box, xoff, yoff in crops:
            img.crop(box).save(path, quality=95)
            crop_faces = _detect_with_detectors(path)
            _deduplicated_add(all_faces, seen, crop_faces, x_offset=xoff, y_offset=yoff)
            log.debug(f"  Crop {Path(path).stem}: {len(crop_faces)}")
            if os.path.exists(path):
                os.remove(path)

    # ── Pass 3: upscale fallback if nothing found ──────────────────────────────
    if len(all_faces) == 0:
        log.debug("  No faces — upscale fallback...")
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        up = img_path + "_up.jpg"
        img.resize((int(w * 1.5), int(h * 1.5)), Image.LANCZOS).save(up, quality=95)
        up_faces = _detect_with_detectors(up)
        _deduplicated_add(all_faces, seen, up_faces, scale=1.5)
        log.debug(f"  Upscale: {len(up_faces)}")
        if os.path.exists(up):
            os.remove(up)

    return all_faces


# ── Selfie embeddings ──────────────────────────────────────────────────────────

_SELFIE_VARIANTS = {
    "original":  lambda img: img,
    "brighter":  lambda img: ImageEnhance.Brightness(img).enhance(1.35),
    "darker":    lambda img: ImageEnhance.Brightness(img).enhance(0.70),
    "flipped":   lambda img: img.transpose(Image.FLIP_LEFT_RIGHT),
    "contrast+": lambda img: ImageEnhance.Contrast(img).enhance(1.2),
}


def get_selfie_embeddings(image_path: str) -> list[np.ndarray]:
    """
    Build multiple embeddings from augmented versions of the selfie.
    Returns a list of unit-norm embedding vectors.
    If fewer than 2 succeed, raises ValueError.
    """
    embeddings: list[np.ndarray] = []
    base = Image.open(image_path).convert("RGB")
    tmp_dir = str(TMP_DIR)

    for name, transform in _SELFIE_VARIANTS.items():
        p = os.path.join(tmp_dir, f"_selfie_{name}.jpg")
        try:
            transform(base).save(p, quality=95)
            # enforce_detection=True — we NEED a face here
            faces = _represent(p, "retinaface", enforce=True)
            if not faces:
                # fallback detectors
                for det in ["mtcnn", "opencv"]:
                    faces = _represent(p, det, enforce=True)
                    if faces:
                        break
            if faces:
                emb = np.array(faces[0]["embedding"], dtype=np.float32)
                # unit-normalise for cosine via dot product
                n = np.linalg.norm(emb)
                if n > 0:
                    embeddings.append(emb / n)
                    log.info(f"  Selfie '{name}': OK")
                else:
                    log.warning(f"  Selfie '{name}': zero-norm embedding, skipped")
            else:
                log.warning(f"  Selfie '{name}': no face detected")
        except Exception as e:
            log.warning(f"  Selfie '{name}': {e}")
        finally:
            if os.path.exists(p):
                os.remove(p)

    if len(embeddings) < 2:
        raise ValueError(
            "Could not extract reliable embeddings from the selfie. "
            "Please use a clear, well-lit, front-facing photo."
        )

    log.info(f"  Total selfie embeddings: {len(embeddings)}")
    return embeddings


def selfie_representative_embedding(embeddings: list[np.ndarray]) -> np.ndarray:
    """
    Average all selfie embeddings into one representative unit vector.
    This is more robust than comparing against each individually.
    """
    avg = np.mean(np.stack(embeddings, axis=0), axis=0)
    n = np.linalg.norm(avg)
    return avg / n if n > 0 else avg


# ── Event image embedding (with disk cache) ────────────────────────────────────

def get_event_embeddings(img_path: str) -> list[np.ndarray]:
    """
    Get face embeddings for an event image.
    Uses a disk cache keyed by file SHA-256 so re-runs are instant.
    Returns list of unit-norm embedding vectors (one per detected face).
    """
    cache_key = _image_hash(img_path)
    cache_file = CACHE_DIR / f"{cache_key}.pkl"

    if cache_file.exists():
        try:
            with open(cache_file, "rb") as f:
                return pickle.load(f)
        except Exception:
            cache_file.unlink(missing_ok=True)

    faces = detect_faces_multi_pass(img_path)
    embeddings = []
    for face in faces:
        emb = np.array(face["embedding"], dtype=np.float32)
        n = np.linalg.norm(emb)
        if n > 0:
            embeddings.append(emb / n)

    try:
        with open(cache_file, "wb") as f:
            pickle.dump(embeddings, f)
    except Exception as e:
        log.warning(f"  Cache write failed: {e}")

    return embeddings


def clear_embedding_cache():
    """Remove all cached embeddings (call after replacing event folder)."""
    for p in CACHE_DIR.glob("*.pkl"):
        p.unlink(missing_ok=True)
    log.info("Embedding cache cleared.")


# ── Matching ───────────────────────────────────────────────────────────────────

def _score_image(
    img_path: Path,
    representative_emb: np.ndarray,
    all_selfie_embs: list[np.ndarray],
    threshold: float,
) -> dict | None:
    """
    Score one event image against the selfie.
    Uses the representative (averaged) embedding for speed,
    then double-checks with all variants only if near the threshold.
    Returns a result dict or None.
    """
    try:
        event_embs = get_event_embeddings(str(img_path))
        if not event_embs:
            log.info(f"  {img_path.name}: no faces")
            return None

        # Fast pass: representative embedding vs all faces
        best = max(
            float(np.dot(representative_emb, ev))
            for ev in event_embs
        )

        # Near-threshold: full multi-selfie comparison (avoids false negatives)
        if best >= threshold * 0.90:
            best = max(
                float(np.dot(se, ev))
                for se in all_selfie_embs
                for ev in event_embs
            )

        status = "MATCH" if best >= threshold else "miss"
        log.info(f"  {img_path.name}: {best:.4f} [{status}]")

        if best >= threshold:
            return {
                "filename":   img_path.name,
                "filepath":   str(img_path.resolve()),
                "confidence": round(best * 100, 2),
            }
    except Exception as e:
        log.warning(f"  {img_path.name}: error — {e}")
    return None


def find_matching_images(
    selfie_bytes: bytes,
    event_folder: str,
    threshold: float = 0.65,
    tmp_selfie_path: str = "",
    max_workers: int = 1,   # kept for API compat, ignored (sequential is safer on Windows)
) -> list[dict]:
    """
    Main entry point. Runs sequentially to avoid TensorFlow multi-thread
    crashes on Windows. The embedding cache makes repeated searches fast.
    """
    # Use a safe temp path
    if not tmp_selfie_path:
        tmp_selfie_path = str(TMP_DIR / "facefind_selfie_tmp.jpg")

    # 1. Build selfie embeddings
    bytes_to_image_path(selfie_bytes, tmp_selfie_path)
    log.info("\n[FaceFind] Building selfie embeddings...")
    all_selfie_embs = get_selfie_embeddings(tmp_selfie_path)
    rep_emb = selfie_representative_embedding(all_selfie_embs)
    log.info(f"  Representative embedding built from {len(all_selfie_embs)} variants.")

    # 2. Gather event images
    folder = Path(event_folder)
    if not folder.exists():
        raise ValueError(f"Event folder not found: {event_folder}")

    event_images = [
        p for p in folder.iterdir()
        if p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    log.info(f"\n[FaceFind] Scanning {len(event_images)} event images (sequential)...\n")

    # 3. Sequential scoring — safe on all platforms including Windows + TF
    matched: list[dict] = []
    for img_path in event_images:
        result = _score_image(img_path, rep_emb, all_selfie_embs, threshold)
        if result:
            matched.append(result)

    # 4. Cleanup
    try:
        if os.path.exists(tmp_selfie_path):
            os.remove(tmp_selfie_path)
    except Exception:
        pass

    matched.sort(key=lambda x: x["confidence"], reverse=True)
    log.info(f"\n[FaceFind] Done. {len(matched)}/{len(event_images)} matched.")
    return matched
