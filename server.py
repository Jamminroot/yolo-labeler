"""YOLO Labeler - Universal labeling tool for YOLO datasets."""

from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import functools
import threading
from pathlib import Path
from urllib.parse import parse_qs, urlparse
import json
import yaml
import cv2
from ultralytics import YOLO

DEFAULT_HOTKEYS = "1234567890qwertyuiopasdfghjklzxcvbnm"

_model = None
_model_path = None


def count_images_in_folder(folder: Path, recursive: bool = False) -> int:
    """Count images in a folder."""
    if not folder.exists() or not folder.is_dir():
        return 0
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if recursive:
        return sum(1 for p in folder.rglob("*") if p.suffix.lower() in exts)
    return sum(1 for p in folder.iterdir() if p.suffix.lower() in exts)


def get_images_in_folder_recursive(folder: Path) -> list:
    """Get all images in folder recursively."""
    if not folder.exists() or not folder.is_dir():
        return []
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted([p for p in folder.rglob("*") if p.suffix.lower() in exts])


_YAML_CANDIDATES = ["dataset.yaml", "dataset.yml", "data.yaml", "data.yml"]


def resolve_yaml_path(path: Path) -> Path:
    if path.is_dir():
        for name in _YAML_CANDIDATES:
            candidate = path / name
            if candidate.exists():
                return candidate
        return path / _YAML_CANDIDATES[0]
    return path


def load_data_yaml(yaml_path: str) -> dict:
    path = resolve_yaml_path(Path(yaml_path))

    if not path.exists():
        return {
            "names": {},
            "path": "",
            "train": "images",
            "val": "images",
            "has_split": False,
            "structure": "flat",
            "nc": 0,
            "yaml_train": "",
            "yaml_val": "",
            "needs_fix": False,
            "fix_suggestion": None,
        }

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    names = data.get("names", {})
    if isinstance(names, list):
        names = {i: n for i, n in enumerate(names)}

    base = Path(data.get("path", str(path.parent)))
    yaml_train = data.get("train") or "images/train"
    yaml_val = data.get("val") or "images/val"

    # Store original yaml values for reference
    train_path = yaml_train
    val_path = yaml_val

    train_full = base / train_path
    val_full = base / val_path

    # Count images in yaml-specified paths
    train_count = count_images_in_folder(train_full)
    val_count = count_images_in_folder(val_full)
    yaml_paths_have_images = train_count > 0 or val_count > 0

    # Check alternative locations
    images_folder = base / "images"
    image_folder = base / "image"
    images_train_folder = base / "images" / "train"
    images_val_folder = base / "images" / "val"

    # Count in various locations
    images_direct_count = count_images_in_folder(images_folder)  # directly in images/
    images_train_count = count_images_in_folder(images_train_folder)
    images_val_count = count_images_in_folder(images_val_folder)
    image_count = count_images_in_folder(image_folder)
    root_count = sum(
        1 for p in base.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )

    # Determine structure and whether yaml needs fixing
    has_split = False
    structure = "flat"
    needs_fix = False
    fix_suggestion = None
    needs_reorganize = False
    reorganize_suggestion = None

    if yaml_paths_have_images:
        # Yaml paths are valid, use them
        if train_full.exists() and val_full.exists() and train_path != val_path:
            has_split = True
            structure = "split"
        else:
            structure = "flat" if train_path == val_path else "split"
            has_split = train_path != val_path
    else:
        # Yaml paths don't have images, check alternatives
        # First check if proper split structure exists
        if images_train_count > 0 or images_val_count > 0:
            # Proper structure exists
            train_path = "images/train"
            val_path = "images/val"
            has_split = True
            structure = "split"
            needs_fix = True
            fix_suggestion = {
                "train": "images/train",
                "val": "images/val",
                "reason": f"Found {images_train_count} train and {images_val_count} val images in proper structure",
            }
        elif images_direct_count > 0:
            # Images directly in images/ folder - suggest reorganization
            needs_reorganize = True
            reorganize_suggestion = {
                "source": "images",
                "count": images_direct_count,
                "reason": f"Found {images_direct_count} images directly in 'images/' folder. They should be in 'images/train/' for proper YOLO structure.",
            }
            # For now, use flat structure
            train_path = "images"
            val_path = "images"
            structure = "flat"
        elif image_count > 0:
            needs_fix = True
            fix_suggestion = {
                "train": "image",
                "val": "image",
                "reason": f"'image' folder has {image_count} images",
            }
            train_path = "image"
            val_path = "image"
            structure = "flat"
        elif root_count > 0:
            needs_fix = True
            fix_suggestion = {
                "train": ".",
                "val": ".",
                "reason": f"Root folder has {root_count} images",
            }
            train_path = "."
            val_path = "."
            structure = "same"
        # else: no images found anywhere, keep yaml paths

    return {
        "names": names,
        "path": str(base),
        "train": train_path,
        "val": val_path,
        "yaml_train": yaml_train,
        "yaml_val": yaml_val,
        "has_split": has_split,
        "structure": structure,
        "nc": data.get("nc", len(names)),
        "needs_fix": needs_fix,
        "fix_suggestion": fix_suggestion,
        "needs_reorganize": needs_reorganize,
        "reorganize_suggestion": reorganize_suggestion,
    }


def get_dataset_structure(yaml_data: dict) -> dict:
    """Get YOLO dataset folder structure based on detected structure type."""
    base = Path(yaml_data["path"])
    structure_type = yaml_data.get("structure", "flat")
    train_path = yaml_data.get("train", "images")
    val_path = yaml_data.get("val", "images")

    result = {
        "base": str(base),
        "structure": structure_type,
        "train_path": train_path,
        "val_path": val_path,
    }

    # Determine image and label folders based on structure
    if structure_type == "split":
        result["images_train"] = str(base / train_path)
        result["images_val"] = str(base / val_path)
        # Labels mirror images structure
        labels_train = (
            train_path.replace("images", "labels")
            if "images" in train_path
            else f"labels/{train_path.split('/')[-1]}"
        )
        labels_val = (
            val_path.replace("images", "labels")
            if "images" in val_path
            else f"labels/{val_path.split('/')[-1]}"
        )
        result["labels_train"] = str(base / labels_train)
        result["labels_val"] = str(base / labels_val)
    elif structure_type == "flat":
        # Use actual train_path from yaml detection (handles image vs images)
        result["images_train"] = str(base / train_path)
        result["images_val"] = str(base / val_path)
        # Determine labels folder (label vs labels)
        labels_dir = "label" if (base / "label").is_dir() else "labels"
        result["labels_train"] = str(base / labels_dir)
        result["labels_val"] = str(base / labels_dir)
    else:  # same folder
        result["images_train"] = str(base)
        result["images_val"] = str(base)
        labels_dir = "label" if (base / "label").is_dir() else "labels"
        result["labels_train"] = str(base / labels_dir)
        result["labels_val"] = str(base / labels_dir)

    return result


def get_images_in_folder(folder_path: str):
    """Get all images in folder."""
    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        return []
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    images = [str(p) for p in sorted(folder.iterdir()) if p.suffix.lower() in exts]
    return images


def get_label_path(image_path: str, dataset_base: str, structure: str = "flat") -> Path:
    """Get corresponding label path for an image based on structure type."""
    img_path = Path(image_path)
    base = Path(dataset_base)

    if structure == "same":
        # Images in root, labels go to labels/ folder
        labels_dir = base / "labels" if (base / "labels").is_dir() else base / "label"
        return labels_dir / img_path.with_suffix(".txt").name
    else:
        # For split and flat: replace 'images'/'image' with 'labels'/'label' in path
        try:
            rel = img_path.relative_to(base)
            parts = list(rel.parts)
            # Handle both plural and singular folder names
            if "images" in parts:
                idx = parts.index("images")
                parts[idx] = "labels"
                return base / Path(*parts).with_suffix(".txt")
            elif "image" in parts:
                idx = parts.index("image")
                parts[idx] = "label"
                return base / Path(*parts).with_suffix(".txt")
            else:
                # Fallback: check which labels folder exists
                labels_dir = "labels" if (base / "labels").is_dir() else "label"
                return base / labels_dir / rel.with_suffix(".txt")
        except ValueError:
            # Image not under base, use labels/ + filename
            labels_dir = "labels" if (base / "labels").is_dir() else "label"
            return base / labels_dir / img_path.with_suffix(".txt").name


def load_labels(
    image_path: str, dataset_base: str, class_names: dict, structure: str = "flat"
) -> list:
    """Load labels for an image."""
    label_path = get_label_path(image_path, dataset_base, structure)

    if not label_path.exists():
        return None

    regions = []
    for line in label_path.read_text().strip().split("\n"):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) >= 5:
            cls_idx = int(parts[0])
            cls_name = class_names.get(cls_idx, f"unknown_{cls_idx}")
            regions.append(
                {
                    "class": cls_name,
                    "class_id": cls_idx,
                    "box": [float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])],
                }
            )
    return regions


def save_labels(
    image_path: str, regions: list, dataset_base: str, class_names: dict, structure: str = "flat"
):
    """Save labels for an image."""
    label_path = get_label_path(image_path, dataset_base, structure)
    label_path.parent.mkdir(parents=True, exist_ok=True)

    name_to_idx = {v: k for k, v in class_names.items()}

    lines = []
    for r in regions:
        cls_name = r.get("class", "")
        cls_idx = r.get("class_id")

        if cls_idx is None:
            cls_idx = name_to_idx.get(cls_name, -1)

        if cls_idx < 0:
            continue

        box = r["box"]
        lines.append(f"{cls_idx} {box[0]} {box[1]} {box[2]} {box[3]}")

    label_path.write_text("\n".join(lines))
    return str(label_path)


# requests run in threads (a grid loads dozens of images at once); the models are used by one request at a time
_models_lock = threading.RLock()


def serial(fn):
    @functools.wraps(fn)
    def run(*a, **k):
        with _models_lock:
            return fn(*a, **k)
    return run


class LabelerServer(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 256  # the default 5 refused the grid's parallel image requests


def load_model(model_path: str):
    """Load YOLO model for auto-detection."""
    global _model, _model_path

    if model_path and Path(model_path).exists():
        if _model_path != model_path:
            _model = YOLO(model_path)
            _model_path = model_path
            print(f"Loaded model: {model_path}")
        return _model
    return None


def train_imgsz(model) -> int:
    """The size the model was trained at (detecting at another size costs accuracy)."""
    try:
        return int(model.ckpt["train_args"]["imgsz"])
    except Exception:
        return 640


# ---- smart boxes: a point / stroke -> a box. Our own detector's low-confidence proposals first, SAM otherwise ----
_sam = None          # ultralytics SAM predictor (MobileSAM), image features cached for _sam_path
_sam_path = None
_proposals = {}      # (image, model) -> [(cls_id, conf, x1, y1, x2, y2)] pixels, low confidence
SAM_WEIGHTS = Path(__file__).resolve().parent / "mobile_sam.pt"


def model_proposals(image_path: str, model_path: str, frame) -> list:
    key = (image_path, model_path)
    if key not in _proposals:
        model = load_model(model_path)
        out = []
        if model is not None:
            b = model.predict(frame, conf=0.05, imgsz=train_imgsz(model), verbose=False)[0].boxes
            out = [(int(c), float(f), *xy) for c, f, xy in zip(b.cls.tolist(), b.conf.tolist(), b.xyxy.tolist())]
        if len(_proposals) > 64:
            _proposals.clear()
        _proposals[key] = out
    return _proposals[key]


def sam_box(image_path: str, frame, pts, clip=None):
    """Box of the SAM mask for these foreground points (one object); clip: only the mask inside this rectangle."""
    global _sam, _sam_path
    import numpy as np
    if _sam is None:
        from ultralytics.models.sam import Predictor as SAMPredictor
        _sam = SAMPredictor(overrides=dict(conf=0.25, task="segment", mode="predict", imgsz=1024,
                                           model=str(SAM_WEIGHTS), save=False, verbose=False))
    if _sam_path != image_path:
        _sam.set_image(frame)
        _sam_path = image_path
    r = _sam(points=[[list(p) for p in pts]], labels=[[1] * len(pts)])
    if not r or r[0].masks is None or not len(r[0].masks.data):
        return None
    m = r[0].masks.data[0].cpu().numpy() > 0.5
    if clip is not None:
        H, W = m.shape
        x1, y1, x2, y2 = (int(max(0, clip[0])), int(max(0, clip[1])), int(min(W, clip[2])), int(min(H, clip[3])))
        c = np.zeros_like(m)
        c[y1:y2, x1:x2] = m[y1:y2, x1:x2]
        m = c
    ys, xs = np.nonzero(m)
    if not len(xs):
        return None
    return float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)


@serial
def smart_box(image_path: str, points: list, cls_id, model_path: str, use_model: bool) -> dict:
    """points: normalized [[x, y], ...] (one point = hover / click, several = a stroke over the object)."""
    frame = cv2.imread(image_path)
    if frame is None or not points:
        return {}
    H, W = frame.shape[:2]
    pts = [(x * W, y * H) for x, y in points]
    sx1, sy1 = min(p[0] for p in pts), min(p[1] for p in pts)
    sx2, sy2 = max(p[0] for p in pts), max(p[1] for p in pts)
    stroke = len(pts) > 1
    # a stroke paints the object: the box hugs it - nothing beyond the stroke and a small margin (a brush edge)
    m = 0.15 * max(sx2 - sx1, sy2 - sy1) + 4
    near = (sx1 - m, sy1 - m, sx2 + m, sy2 + m)
    within = lambda b, r: b[0] >= r[0] and b[1] >= r[1] and b[2] <= r[2] and b[3] <= r[3]
    box, src, conf = None, None, 0.0
    if use_model and model_path and cls_id is not None:
        # a proposal of the wanted class holding (almost) every point; the most confident wins
        inside = lambda b, p: b[0] <= p[0] <= b[2] and b[1] <= p[1] <= b[3]
        cands = [c for c in model_proposals(image_path, model_path, frame)
                 if c[0] == cls_id and sum(inside(c[2:], p) for p in pts) >= max(1, 0.8 * len(pts))
                 and (not stroke or within(c[2:], near))]  # for a stroke: only a proposal that fits the stroke
        if cands:
            best = max(cands, key=lambda c: c[1])
            box, src, conf = best[2:], "model", best[1]
    if box is None:
        box, src = sam_box(image_path, frame, pts, clip=near if stroke else None), "sam"
        if stroke:  # never smaller than the stroke itself
            box = (sx1, sy1, sx2, sy2) if box is None else \
                (min(box[0], sx1), min(box[1], sy1), max(box[2], sx2), max(box[3], sy2))
    if box is None:
        return {}
    if not stroke and (box[2] - box[0]) * (box[3] - box[1]) > 0.5 * W * H:
        return {}  # one point grabbed the background (a tiny element on a flat table): no suggestion
    x1, y1, x2, y2 = box
    return {"box": [(x1 + x2) / 2 / W, (y1 + y2) / 2 / H, (x2 - x1) / W, (y2 - y1) / H], "source": src, "conf": conf}


# ---- F1: read the text of a box (PP-OCRv5 recognizers, one per script; the most confident reading wins) ----
OCR_MODELS = ("PP-OCRv5_server_rec", "latin_PP-OCRv5_mobile_rec", "cyrillic_PP-OCRv5_mobile_rec", "korean_PP-OCRv5_mobile_rec")
_ocr = {}


# offline translation to English (Opus-MT, downloaded once per language): the script of the text picks the model
_mt = {}
MT_MODELS = {"zh": "Helsinki-NLP/opus-mt-zh-en", "ja": "Helsinki-NLP/opus-mt-ja-en", "ko": "Helsinki-NLP/opus-mt-ko-en",
             "ru": "Helsinki-NLP/opus-mt-ru-en", "mul": "Helsinki-NLP/opus-mt-mul-en"}


def text_lang(t: str):
    """Language family by script; None for plain English / digits."""
    if any("\u3040" <= c <= "\u30ff" for c in t):
        return "ja"
    if any("\uac00" <= c <= "\ud7af" or "\u1100" <= c <= "\u11ff" for c in t):
        return "ko"
    if any("\u4e00" <= c <= "\u9fff" for c in t):
        return "zh"
    if any("\u0400" <= c <= "\u04ff" for c in t):
        return "ru"
    if any(ord(c) > 127 and c.isalpha() for c in t):
        return "mul"
    return None


# poker terms a general translator gets wrong (弃牌 -> "Abandon"): exact phrase first, then terms found inside
POKER_TERMS = {
    "跟注": "Call", "弃牌": "Fold", "棄牌": "Fold", "加注": "Raise", "再加注": "Re-raise", "全下": "All-in", "全押": "All-in",
    "看牌": "Check", "过牌": "Check", "過牌": "Check", "让牌": "Check", "讓牌": "Check", "下注": "Bet", "底池": "Pot",
    "主池": "Main pot", "边池": "Side pot", "邊池": "Side pot", "盲注": "Blinds", "小盲": "Small blind", "大盲": "Big blind",
    "前注": "Ante", "抓头": "Straddle", "庄家": "Dealer", "庄": "Dealer", "等待盲注": "Waiting for the big blind",
    "等待": "Waiting", "坐下": "Sit down", "留座离桌": "Sit out (seat kept)", "留座": "Keep seat", "离桌": "Leave table", "离开": "Sit out", "離開": "Sit out", "暂离": "Sit out", "买入": "Buy-in",
    "買入": "Buy-in", "补码": "Add chips", "赢": "Won", "赢得": "Won", "摊牌": "Showdown", "保险": "Insurance",
    "对子": "Pair", "两对": "Two pair", "三条": "Three of a kind", "顺子": "Straight", "同花": "Flush", "葫芦": "Full house",
    "四条": "Four of a kind", "同花顺": "Straight flush", "皇家同花顺": "Royal flush", "高牌": "High card",
    "Пас": "Fold", "Чек": "Check", "Колл": "Call", "Рейз": "Raise", "Ставка": "Bet", "Олл-ин": "All-in", "Банк": "Pot",
    "Сброс": "Fold", "Уравнять": "Call", "Повысить": "Raise",
    "콜": "Call", "폴드": "Fold", "레이즈": "Raise", "체크": "Check", "베팅": "Bet", "올인": "All-in", "팟": "Pot",
    "コール": "Call", "フォールド": "Fold", "レイズ": "Raise", "チェック": "Check", "ベット": "Bet", "オールイン": "All-in",
}


def translate(text: str) -> dict:
    lang = text_lang(text)
    if not lang or not text.strip():
        return {"lang": "en", "text": text}
    t = text.strip()
    if t in POKER_TERMS:
        return {"lang": lang, "text": POKER_TERMS[t], "terms": {t: POKER_TERMS[t]}}
    terms = {k: v for k, v in sorted(POKER_TERMS.items(), key=lambda kv: -len(kv[0])) if k in t}
    from transformers import MarianMTModel, MarianTokenizer
    import torch
    if lang not in _mt:
        name = MT_MODELS[lang]
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        _mt[lang] = (MarianTokenizer.from_pretrained(name), MarianMTModel.from_pretrained(name).to(dev).eval(), dev)
    tok, model, dev = _mt[lang]
    with torch.no_grad():
        out = model.generate(**tok([text], return_tensors="pt", truncation=True).to(dev), max_new_tokens=64)
    return {"lang": lang, "text": tok.decode(out[0], skip_special_tokens=True), "terms": terms}


@serial
def read_box(image_path: str, box) -> dict:
    """box: normalized cx, cy, w, h -> {"best": {text, score, model}, "all": [...]}"""
    import os
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    from paddleocr import TextRecognition
    frame = cv2.imread(image_path)
    if frame is None:
        return {}
    H, W = frame.shape[:2]
    cx, cy, w, h = box
    pad = 0.08
    x1, y1 = max(0, int((cx - w / 2 - w * pad) * W)), max(0, int((cy - h / 2 - h * pad) * H))
    x2, y2 = min(W, int((cx + w / 2 + w * pad) * W) + 1), min(H, int((cy + h / 2 + h * pad) * H) + 1)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return {}
    if crop.shape[0] < 48:  # the recognizers read ~48 px text
        f = 48 / crop.shape[0]
        crop = cv2.resize(crop, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
    out = []
    for name in OCR_MODELS:
        if name not in _ocr:
            _ocr[name] = TextRecognition(model_name=name)
        r = _ocr[name].predict(crop)[0]
        out.append({"model": name, "text": r["rec_text"], "score": round(float(r["rec_score"]), 3)})
    # the longest confident reading wins: a recognizer that can't read a script drops it (1留座离桌 -> "1" at 0.96)
    sure = [o for o in out if o["score"] >= 0.5 and o["text"].strip()] or out
    best = max(sure, key=lambda o: (len(o["text"].strip()), o["score"]))
    try:
        tr = translate(best["text"])
    except Exception as e:  # no translation is still a reading
        tr = {"lang": "?", "text": "", "error": str(e)}
    return {"best": best, "all": out, "translation": tr}


@serial
def detect_regions(image_path: str, model_path: str, class_names: dict) -> list:
    """Run detection on image."""
    model = load_model(model_path)
    if model is None:
        return []

    frame = cv2.imread(image_path)
    if frame is None:
        return []

    h, w = frame.shape[:2]
    results = model.predict(frame, conf=0.25, imgsz=train_imgsz(model), verbose=False)

    regions = []
    if results and len(results) > 0:
        boxes = results[0].boxes
        if boxes is not None:
            for box in boxes:
                cls_id = int(box.cls[0].item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cx = ((x1 + x2) / 2) / w
                cy = ((y1 + y2) / 2) / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h

                cls_name = class_names.get(cls_id, f"unknown_{cls_id}")
                regions.append({"class": cls_name, "class_id": cls_id, "box": [cx, cy, bw, bh]})
    return regions


def get_progress_file(folder_path: str):
    return Path(folder_path).parent.parent / ".labeler_progress.json"


def level_rules_file(dataset_base: str) -> Path:
    """Constraints are shared by every dataset of a level: a file next to the datasets (their parent folder)."""
    return Path(dataset_base).parent / ".labeler_constraints.json"


def get_cache_file(dataset_base: str):
    return Path(dataset_base) / ".yolo_labeler_cache.json"


def load_progress(folder_path: str):
    pf = get_progress_file(folder_path)
    if pf.exists():
        return json.loads(pf.read_text())
    return {"completed": [], "current_index": 0}


def save_progress(folder_path: str, progress: dict):
    pf = get_progress_file(folder_path)
    pf.write_text(json.dumps(progress, indent=2))


def build_label_cache(dataset_base: str, structure: str = "flat"):
    """Scan all images and check which have label files."""
    base = Path(dataset_base)
    cache_file = get_cache_file(dataset_base)

    # Find all image folders based on structure
    image_folders = []
    if structure == "split":
        for split in ["train", "val", "test"]:
            img_dir = base / "images" / split
            if img_dir.exists():
                image_folders.append(img_dir)
    else:
        for name in ["images", "image"]:
            img_dir = base / name
            if img_dir.exists():
                image_folders.append(img_dir)
                break
        if not image_folders and base.exists():
            image_folders.append(base)

    completed = []
    for img_folder in image_folders:
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]:
            for img_path in img_folder.glob(ext):
                label_path = get_label_path(str(img_path), dataset_base, structure)
                if label_path.exists():
                    completed.append(str(img_path))

    cache_data = {
        "completed": completed,
        "structure": structure,
        "timestamp": str(Path(cache_file).stat().st_mtime if cache_file.exists() else 0),
    }
    cache_file.write_text(json.dumps(cache_data, indent=2))
    return cache_data


def load_label_cache(dataset_base: str):
    """Load cached label state."""
    cache_file = get_cache_file(dataset_base)
    if cache_file.exists():
        try:
            return json.loads(cache_file.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {"completed": [], "structure": "flat", "timestamp": "0", "settings": {}}


def save_settings_to_cache(dataset_base: str, settings: dict):
    """Save settings to cache file."""
    cache_file = get_cache_file(dataset_base)
    cache = {}
    if cache_file.exists():
        try:
            cache = json.loads(cache_file.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    cache["settings"] = settings
    cache_file.write_text(json.dumps(cache, indent=2))


class YoloLabelHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == "/":
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            html_path = Path(__file__).parent / "index.html"
            self.wfile.write(html_path.read_bytes())

        elif parsed.path == "/load_dataset":
            qs = parse_qs(parsed.query)
            yaml_path = qs.get("yaml", [""])[0]

            yaml_data = load_data_yaml(yaml_path)
            structure = get_dataset_structure(yaml_data)

            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(
                json.dumps(
                    {
                        "classes": yaml_data["names"],
                        "structure": structure,
                        "has_split": yaml_data["has_split"],
                        "train_path": yaml_data["train"],
                        "val_path": yaml_data["val"],
                        "yaml_train": yaml_data.get("yaml_train", ""),
                        "yaml_val": yaml_data.get("yaml_val", ""),
                        "yaml_path": yaml_path,
                        "needs_fix": yaml_data.get("needs_fix", False),
                        "fix_suggestion": yaml_data.get("fix_suggestion"),
                        "needs_reorganize": yaml_data.get("needs_reorganize", False),
                        "reorganize_suggestion": yaml_data.get("reorganize_suggestion"),
                    }
                ).encode()
            )

        elif parsed.path == "/folder":
            qs = parse_qs(parsed.query)
            folder_path = qs.get("path", [""])[0]
            images = get_images_in_folder(folder_path)
            progress = (
                load_progress(folder_path) if images else {"completed": [], "current_index": 0}
            )

            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"images": images, "progress": progress}).encode())

        elif parsed.path == "/load_cache":
            qs = parse_qs(parsed.query)
            dataset_base = qs.get("base", [""])[0]
            cache = load_label_cache(dataset_base)

            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(cache).encode())

        elif parsed.path == "/rebuild_cache":
            qs = parse_qs(parsed.query)
            dataset_base = qs.get("base", [""])[0]
            structure = qs.get("structure", ["flat"])[0]
            cache = build_label_cache(dataset_base, structure)

            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(cache).encode())

        elif parsed.path == "/labels":
            qs = parse_qs(parsed.query)
            image_path = qs.get("path", [""])[0]
            dataset_base = qs.get("base", [""])[0]
            structure = qs.get("structure", ["flat"])[0]
            classes_json = qs.get("classes", ["{}"])[0]
            class_names = json.loads(classes_json)
            class_names = {int(k): v for k, v in class_names.items()}

            regions = load_labels(image_path, dataset_base, class_names, structure)

            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"regions": regions}).encode())

        elif parsed.path == "/constraints":
            f = level_rules_file(parse_qs(parsed.query).get("base", [""])[0])
            rules = None
            if f.exists():
                try:
                    rules = json.loads(f.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    pass
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"constraints": rules, "file": str(f)}).encode())

        elif parsed.path == "/detect":
            qs = parse_qs(parsed.query)
            image_path = qs.get("path", [""])[0]
            model_path = qs.get("model", [""])[0]
            classes_json = qs.get("classes", ["{}"])[0]
            class_names = json.loads(classes_json)
            class_names = {int(k): v for k, v in class_names.items()}

            regions = detect_regions(image_path, model_path, class_names)

            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(regions).encode())

        elif parsed.path == "/image":
            qs = parse_qs(parsed.query)
            image_path = qs.get("path", [""])[0]
            p = Path(image_path)
            if image_path and p.exists() and p.is_file():
                self.send_response(200)
                ext = p.suffix.lower()
                ct = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}.get(
                    ext, "image/jpeg"
                )
                self.send_header("Content-type", ct)
                # images may be re-cut in place: never let the browser show a stale copy next to fresh labels
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(p.read_bytes())
            else:
                self.send_error(404, "Image not found")
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == "/constraints":
            length = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(length))
            level_rules_file(data["base"]).write_text(json.dumps(data.get("constraints") or [], indent=2, ensure_ascii=False),
                                                      encoding="utf-8")
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"ok": true}')
            return

        if self.path == "/ocr":
            length = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(length))
            try:
                out = read_box(data["path"], data["box"])
            except Exception as e:  # never kill the server over a helper
                print("ocr failed:", e)
                out = {"error": str(e)}
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(out, ensure_ascii=False).encode("utf-8"))
            return

        if self.path == "/smart":
            # {path, points: [[x, y], ...] normalized, cls_id, model, use_model} -> {box, source, conf} | {}
            length = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(length))
            try:
                out = smart_box(data["path"], data.get("points") or [], data.get("cls_id"), data.get("model") or "",
                                bool(data.get("use_model", True)))
            except Exception as e:  # never kill the server over a helper
                print("smart box failed:", e)
                out = {"error": str(e)}
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(out).encode())
            return

        if self.path == "/label_boxes":
            # {images:[...], dataset_base, structure} -> {path: [[cls,x,y,w,h], ...] | null (no label file)}
            length = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(length))
            out = {}
            for image_path in data.get("images", []):
                lp = get_label_path(image_path, data["dataset_base"], data.get("structure", "flat"))
                if not lp.exists():
                    out[image_path] = None
                    continue
                boxes = []
                for line in lp.read_text().splitlines():
                    parts = line.split()
                    if len(parts) >= 5 and parts[0].lstrip("-").isdigit():
                        boxes.append([int(parts[0])] + [float(v) for v in parts[1:5]])
                out[image_path] = boxes
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(out).encode())
            return

        if self.path == "/save_batch":
            length = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(length))

            items = data["items"]  # [{image_path, regions}, ...]
            dataset_base = data["dataset_base"]
            structure = data.get("structure", "flat")
            classes_json = data.get("classes", "{}")
            class_names = {int(k): v for k, v in classes_json.items()}

            saved = []
            errors = []

            for item in items:
                try:
                    image_path = item["image_path"]
                    regions = item["regions"]
                    saved_path = save_labels(
                        image_path, regions, dataset_base, class_names, structure
                    )
                    saved.append(image_path)

                    # Update cache
                    cache_file = get_cache_file(dataset_base)
                    if cache_file.exists():
                        try:
                            cache = json.loads(cache_file.read_text())
                            completed = cache.setdefault("completed", [])
                            if image_path not in completed:
                                completed.append(image_path)
                                cache_file.write_text(json.dumps(cache, indent=2))
                        except (json.JSONDecodeError, OSError):
                            pass
                except Exception as e:
                    errors.append({"path": item.get("image_path", "unknown"), "error": str(e)})

            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(
                json.dumps({"saved": saved, "count": len(saved), "errors": errors}).encode()
            )

        elif self.path == "/save":
            length = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(length))

            image_path = data["image_path"]
            regions = data["regions"]
            dataset_base = data["dataset_base"]
            structure = data.get("structure", "flat")
            classes_json = data.get("classes", "{}")
            class_names = {int(k): v for k, v in classes_json.items()}

            saved_path = save_labels(image_path, regions, dataset_base, class_names, structure)

            # Update cache
            cache_file = get_cache_file(dataset_base)
            if cache_file.exists():
                try:
                    cache = json.loads(cache_file.read_text())
                    completed = cache.setdefault("completed", [])
                    if image_path not in completed:
                        completed.append(image_path)
                        cache_file.write_text(json.dumps(cache, indent=2))
                except (json.JSONDecodeError, OSError):
                    pass

            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"saved": saved_path}).encode())

        elif self.path == "/progress":
            length = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(length))
            folder_path = data["folder_path"]
            progress = data["progress"]
            save_progress(folder_path, progress)

            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"ok": True}).encode())

        elif self.path == "/save_settings":
            length = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(length))
            dataset_base = data["dataset_base"]
            settings = data["settings"]
            save_settings_to_cache(dataset_base, settings)

            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"ok": True}).encode())

        elif self.path == "/add_class":
            length = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(length))
            yaml_path = data["yaml_path"]
            class_name = data["class_name"]

            try:
                path = resolve_yaml_path(Path(yaml_path))

                if not path.exists():
                    raise FileNotFoundError(f"No yaml found at {path}")

                with open(path, "r", encoding="utf-8") as f:
                    yaml_data = yaml.safe_load(f)

                names = yaml_data.get("names", {})
                if isinstance(names, list):
                    names = {i: n for i, n in enumerate(names)}

                new_idx = len(names)
                names[new_idx] = class_name
                yaml_data["names"] = names
                yaml_data["nc"] = len(names)

                with open(path, "w", encoding="utf-8") as f:
                    yaml.dump(yaml_data, f, default_flow_style=False, allow_unicode=True)

                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"ok": True, "index": new_idx}).encode())
            except Exception as e:
                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode())

        elif self.path == "/fix_yaml":
            length = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(length))
            yaml_path = data["yaml_path"]
            train_path = data["train"]
            val_path = data["val"]

            try:
                path = resolve_yaml_path(Path(yaml_path))

                if not path.exists():
                    raise FileNotFoundError(f"No yaml found at {path}")

                # Read existing yaml
                with open(path, "r", encoding="utf-8") as f:
                    yaml_data = yaml.safe_load(f)

                # Update train/val paths
                yaml_data["train"] = train_path
                yaml_data["val"] = val_path

                # Write back
                with open(path, "w", encoding="utf-8") as f:
                    yaml.dump(yaml_data, f, default_flow_style=False, allow_unicode=True)

                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"ok": True, "path": str(path)}).encode())
            except Exception as e:
                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"ok": False, "error": str(e)}).encode())

        elif self.path == "/reorganize":
            length = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(length))
            dataset_base = data["dataset_base"]
            yaml_path = data.get("yaml_path", "")

            try:
                base = Path(dataset_base)
                images_dir = base / "images"
                labels_dir = base / "labels"

                # Create train subdirectories
                train_images = images_dir / "train"
                train_labels = labels_dir / "train"
                val_images = images_dir / "val"
                val_labels = labels_dir / "val"

                train_images.mkdir(parents=True, exist_ok=True)
                train_labels.mkdir(parents=True, exist_ok=True)
                val_images.mkdir(parents=True, exist_ok=True)
                val_labels.mkdir(parents=True, exist_ok=True)

                # Move images from images/ to images/train/
                moved_images = 0
                moved_labels = 0
                exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

                for img_path in list(images_dir.iterdir()):
                    if img_path.is_file() and img_path.suffix.lower() in exts:
                        dest = train_images / img_path.name
                        img_path.rename(dest)
                        moved_images += 1

                        # Move corresponding label if exists
                        label_path = labels_dir / img_path.with_suffix(".txt").name
                        if label_path.exists():
                            label_dest = train_labels / label_path.name
                            label_path.rename(label_dest)
                            moved_labels += 1

                if yaml_path:
                    yp = resolve_yaml_path(Path(yaml_path))
                    if yp.exists():
                        with open(yp, "r", encoding="utf-8") as f:
                            yaml_data = yaml.safe_load(f)
                        yaml_data["train"] = "images/train"
                        yaml_data["val"] = "images/val"
                        with open(yp, "w", encoding="utf-8") as f:
                            yaml.dump(yaml_data, f, default_flow_style=False, allow_unicode=True)

                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(
                    json.dumps(
                        {"ok": True, "moved_images": moved_images, "moved_labels": moved_labels}
                    ).encode()
                )
            except Exception as e:
                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"ok": False, "error": str(e)}).encode())

        elif self.path == "/split_dataset":
            length = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(length))
            dataset_base = data["dataset_base"]
            yaml_path = data.get("yaml_path", "")
            val_percent = data.get("val_percent", 20)
            only_labeled = data.get("only_labeled", False)
            shuffle = data.get("shuffle", True)

            try:
                import random

                base = Path(dataset_base)

                # Find all images in train and val folders
                train_images_dir = base / "images" / "train"
                val_images_dir = base / "images" / "val"
                train_labels_dir = base / "labels" / "train"
                val_labels_dir = base / "labels" / "val"

                # Ensure directories exist
                train_images_dir.mkdir(parents=True, exist_ok=True)
                val_images_dir.mkdir(parents=True, exist_ok=True)
                train_labels_dir.mkdir(parents=True, exist_ok=True)
                val_labels_dir.mkdir(parents=True, exist_ok=True)

                # Collect all images from both train and val
                exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
                all_images = []

                for img_path in train_images_dir.iterdir():
                    if img_path.is_file() and img_path.suffix.lower() in exts:
                        label_path = train_labels_dir / img_path.with_suffix(".txt").name
                        has_label = label_path.exists()
                        if not only_labeled or has_label:
                            all_images.append(("train", img_path, has_label))

                for img_path in val_images_dir.iterdir():
                    if img_path.is_file() and img_path.suffix.lower() in exts:
                        label_path = val_labels_dir / img_path.with_suffix(".txt").name
                        has_label = label_path.exists()
                        if not only_labeled or has_label:
                            all_images.append(("val", img_path, has_label))

                if shuffle:
                    random.shuffle(all_images)

                # Calculate split
                total = len(all_images)
                val_count = int(total * val_percent / 100)
                train_count = total - val_count

                # Split: first train_count go to train, rest to val
                new_train = all_images[:train_count]
                new_val = all_images[train_count:]

                # Move files
                moved_to_train = 0
                moved_to_val = 0

                for current_split, img_path, has_label in new_train:
                    if current_split != "train":
                        # Move to train
                        dest = train_images_dir / img_path.name
                        img_path.rename(dest)
                        moved_to_train += 1
                        if has_label:
                            label_src = val_labels_dir / img_path.with_suffix(".txt").name
                            label_dest = train_labels_dir / img_path.with_suffix(".txt").name
                            if label_src.exists():
                                label_src.rename(label_dest)

                for current_split, img_path, has_label in new_val:
                    if current_split != "val":
                        # Move to val
                        dest = val_images_dir / img_path.name
                        img_path.rename(dest)
                        moved_to_val += 1
                        if has_label:
                            label_src = train_labels_dir / img_path.with_suffix(".txt").name
                            label_dest = val_labels_dir / img_path.with_suffix(".txt").name
                            if label_src.exists():
                                label_src.rename(label_dest)

                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(
                    json.dumps(
                        {
                            "ok": True,
                            "total": total,
                            "train_count": train_count,
                            "val_count": val_count,
                            "moved_to_train": moved_to_train,
                            "moved_to_val": moved_to_val,
                        }
                    ).encode()
                )
            except Exception as e:
                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"ok": False, "error": str(e)}).encode())

        elif self.path == "/create_yaml":
            length = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(length))
            yaml_path = data["path"]
            classes = data["classes"]

            try:
                path = Path(yaml_path)
                if path.is_dir() or not path.suffix:
                    base_dir = path if path.is_dir() else path
                    yaml_file = resolve_yaml_path(base_dir)
                else:
                    yaml_file = path
                    base_dir = path.parent

                base_dir.mkdir(parents=True, exist_ok=True)

                # Create yaml content
                yaml_content = {
                    "path": str(base_dir),
                    "train": "images",
                    "val": "images",
                    "names": {i: name for i, name in enumerate(classes)},
                }

                with open(yaml_file, "w", encoding="utf-8") as f:
                    yaml.dump(yaml_content, f, default_flow_style=False, allow_unicode=True)

                # Create images and labels folders
                (base_dir / "images").mkdir(exist_ok=True)
                (base_dir / "labels").mkdir(exist_ok=True)

                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"ok": True, "path": str(yaml_file)}).encode())
            except Exception as e:
                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"ok": False, "error": str(e)}).encode())
        elif self.path == "/delete_entry":
            length = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(length))
            image_path = data["image_path"]
            dataset_base = data["dataset_base"]
            structure = data.get("structure", "flat")

            deleted = []
            img = Path(image_path)
            if img.exists():
                img.unlink()
                deleted.append(str(img))

            label = get_label_path(image_path, dataset_base, structure)
            if label.exists():
                label.unlink()
                deleted.append(str(label))

            cache_file = get_cache_file(dataset_base)
            if cache_file.exists():
                try:
                    cache = json.loads(cache_file.read_text())
                    completed = cache.get("completed", [])
                    if image_path in completed:
                        completed.remove(image_path)
                        cache_file.write_text(json.dumps(cache, indent=2))
                except (json.JSONDecodeError, OSError):
                    pass

            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"ok": True, "deleted": deleted}).encode())

        elif self.path == "/delete_batch":
            length = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(length))
            items = data["items"]
            dataset_base = data["dataset_base"]
            structure = data.get("structure", "flat")

            deleted = []
            cache_file = get_cache_file(dataset_base)
            cache = None
            if cache_file.exists():
                try:
                    cache = json.loads(cache_file.read_text())
                except (json.JSONDecodeError, OSError):
                    cache = None

            for image_path in items:
                img = Path(image_path)
                if img.exists():
                    img.unlink()
                label = get_label_path(image_path, dataset_base, structure)
                if label.exists():
                    label.unlink()
                if cache:
                    completed = cache.get("completed", [])
                    if image_path in completed:
                        completed.remove(image_path)
                deleted.append(image_path)

            if cache:
                try:
                    cache_file.write_text(json.dumps(cache, indent=2))
                except OSError:
                    pass

            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"ok": True, "deleted": deleted}).encode())

        elif self.path == "/clear_labels_batch":
            length = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(length))
            items = data["items"]
            dataset_base = data["dataset_base"]
            structure = data.get("structure", "flat")

            cleared = []
            cache_file = get_cache_file(dataset_base)
            cache = None
            if cache_file.exists():
                try:
                    cache = json.loads(cache_file.read_text())
                except (json.JSONDecodeError, OSError):
                    cache = None

            for image_path in items:
                label = get_label_path(image_path, dataset_base, structure)
                if label.exists():
                    label.unlink()
                if cache:
                    completed = cache.get("completed", [])
                    if image_path in completed:
                        completed.remove(image_path)
                cleared.append(image_path)

            if cache:
                try:
                    cache_file.write_text(json.dumps(cache, indent=2))
                except OSError:
                    pass

            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"ok": True, "cleared": cleared}).encode())

        else:
            self.send_error(404)

    def log_message(self, format, *args):
        pass


def main():
    import webbrowser

    port = 8770
    server = LabelerServer(("localhost", port), YoloLabelHandler)
    print(f"YOLO Labeler running at http://localhost:{port}")
    print("Press Ctrl+C to stop")
    webbrowser.open(f"http://localhost:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped")


if __name__ == "__main__":
    main()
