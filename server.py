"""YOLO Labeler - Universal labeling tool for YOLO datasets."""

from http.server import HTTPServer, SimpleHTTPRequestHandler
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


def load_data_yaml(yaml_path: str) -> dict:
    """Load and parse data.yaml file, detecting dataset structure."""
    path = Path(yaml_path)

    # If path is a directory, look for data.yaml inside
    if path.is_dir():
        path = path / "data.yaml"

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


def detect_regions(image_path: str, model_path: str, class_names: dict) -> list:
    """Run detection on image."""
    model = load_model(model_path)
    if model is None:
        return []

    frame = cv2.imread(image_path)
    if frame is None:
        return []

    h, w = frame.shape[:2]
    results = model.predict(frame, conf=0.25, verbose=False)

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
                self.end_headers()
                self.wfile.write(p.read_bytes())
            else:
                self.send_error(404, "Image not found")
        else:
            self.send_error(404)

    def do_POST(self):
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
                            if image_path not in cache.get("completed", []):
                                cache["completed"].append(image_path)
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
                    if image_path not in cache.get("completed", []):
                        cache["completed"].append(image_path)
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
                path = Path(yaml_path)
                if path.is_dir():
                    path = path / "data.yaml"

                if not path.exists():
                    raise FileNotFoundError(f"data.yaml not found at {path}")

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
                path = Path(yaml_path)
                if path.is_dir():
                    path = path / "data.yaml"

                if not path.exists():
                    raise FileNotFoundError(f"data.yaml not found at {path}")

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

                # Update data.yaml
                if yaml_path:
                    yp = Path(yaml_path)
                    if yp.is_dir():
                        yp = yp / "data.yaml"
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
                # If path is directory, create data.yaml inside
                if path.is_dir() or not path.suffix:
                    base_dir = path if path.is_dir() else path
                    yaml_file = base_dir / "data.yaml"
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
        else:
            self.send_error(404)

    def log_message(self, format, *args):
        pass


def main():
    import webbrowser

    port = 8770
    server = HTTPServer(("localhost", port), YoloLabelHandler)
    print(f"YOLO Labeler running at http://localhost:{port}")
    print("Press Ctrl+C to stop")
    webbrowser.open(f"http://localhost:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped")


if __name__ == "__main__":
    main()
