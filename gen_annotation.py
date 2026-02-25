import json
import random
from pathlib import Path
from typing import Tuple, List

# Base paths
BASES = [
    Path(r"I:\Smart\human-accident"),
    Path(r"I:\Smart\human-accident\20251006PM_frames"),
]

OUT_DIR = Path(r"I:\Smart\cnn-lstm-master\datasets")
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
FRAME_COUNT = 20
random.seed(42)

INCIDENT_CLASSES = ["bump", "fall-down", "fall-off", "hit", "jam"]
NEGATIVE_CLASS = "no-accident"
CLASS_DIRS = INCIDENT_CLASSES + [NEGATIVE_CLASS]


def is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMAGE_EXTS


def count_frames(clip_dir: Path) -> int:
    return sum(1 for f in clip_dir.iterdir() if is_image(f))


def detect_subset_from_path(p: Path):
    parts = [s.lower() for s in p.parts]
    for key in ("training", "validation", "testing"):
        if key in parts:
            return key
    return None


def gather_all_clips():
    items: List[Tuple[Path, str, str, int, Path]] = []
    excluded = []
    for BASE in BASES:
        if not BASE.exists():
            continue

        if BASE.name == "human-accident":
            for cls in CLASS_DIRS:
                class_dir = BASE / cls
                if not class_dir.exists():
                    continue
                for clip in class_dir.rglob("*"):
                    if clip.is_dir():
                        actual_n = count_frames(clip)
                        if actual_n == FRAME_COUNT:
                            subset = detect_subset_from_path(clip)
                            items.append((clip, cls, subset, actual_n, BASE))
                        elif actual_n > 0:
                            excluded.append((clip, actual_n))
        elif BASE.name == "20251006PM_frames":
            cls = NEGATIVE_CLASS
            for video_dir in BASE.iterdir():
                if not video_dir.is_dir():
                    continue
                for clip in video_dir.iterdir():
                    if clip.is_dir():
                        actual_n = count_frames(clip)
                        if actual_n == FRAME_COUNT:
                            subset = detect_subset_from_path(clip)
                            items.append((clip, cls, subset, actual_n, BASE))
                        elif actual_n > 0:
                            excluded.append((clip, actual_n))
        elif BASE.name in CLASS_DIRS:
            cls = BASE.name
            for clip in BASE.rglob("*"):
                if clip.is_dir():
                    actual_n = count_frames(clip)
                    if actual_n == FRAME_COUNT:
                        subset = detect_subset_from_path(clip)
                        items.append((clip, cls, subset, actual_n, BASE))
                    elif actual_n > 0:
                        excluded.append((clip, actual_n))

    return items, excluded


def split_if_needed(items):
    with_subset = [(c, cls, s, n, base) for (c, cls, s, n, base) in items if s is not None]
    without_subset = [(c, cls, n, base) for (c, cls, s, n, base) in items if s is None]

    by_class = {}
    for clip, cls, n, base in without_subset:
        by_class.setdefault(cls, []).append((clip, n, base))

    new_items = with_subset[:]
    for cls, arr in by_class.items():
        random.shuffle(arr)
        k = int(len(arr) * 0.8)
        train_part = arr[:k]
        val_part = arr[k:]
        new_items += [(c, cls, "training", n, base) for (c, n, base) in train_part]
        new_items += [(c, cls, "validation", n, base) for (c, n, base) in val_part]

    return new_items


def build_database_json(items, positive_class):
    if positive_class not in INCIDENT_CLASSES:
        raise ValueError(f"지원하지 않는 positive 클래스: {positive_class}")

    labels_map = {NEGATIVE_CLASS: 0, positive_class: 1}
    database = {}
    class_counts = {cls: 0 for cls in CLASS_DIRS}
    subset_class_counts = {cls: {"training": 0, "validation": 0, "testing": 0, "unspecified": 0} for cls in CLASS_DIRS}
    effective_counts = {NEGATIVE_CLASS: 0, positive_class: 0}

    for clip_dir, cls, subset, _actual_n, BASE in items:
        if cls not in class_counts:
            class_counts[cls] = 0
            subset_class_counts.setdefault(cls, {"training": 0, "validation": 0, "testing": 0, "unspecified": 0})
        class_counts[cls] += 1
        subset_key = subset if subset in {"training", "validation", "testing"} else "unspecified"
        subset_class_counts[cls][subset_key] = subset_class_counts[cls].get(subset_key, 0) + 1

        is_positive = cls == positive_class
        label_name = positive_class if is_positive else NEGATIVE_CLASS
        label_id_final = 1 if is_positive else 0
        effective_counts[label_name] = effective_counts.get(label_name, 0) + 1

        rel_path = clip_dir.relative_to(BASE)
        if BASE.name == "20251006PM_frames":
            rel = Path(BASE.name) / rel_path
        else:
            rel = rel_path
        rel = rel.as_posix()

        database[rel] = {
            "subset": subset,
            "annotations": {
                "label": label_name,
                "label_id": label_id_final,
                "n_frames": FRAME_COUNT,
            },
        }

    print(f"[INFO] Positive 클래스: {positive_class}")
    print(f" - 총 클립 수: {len(items)}")
    for cls, count in sorted(class_counts.items()):
        subsets = subset_class_counts.get(cls, {})
        tr = subsets.get("training", 0)
        va = subsets.get("validation", 0)
        te = subsets.get("testing", 0)
        unspec = subsets.get("unspecified", 0)
        if count > 0:
            print(f"   * {cls}: {count} (train {tr} / val {va} / test {te} / other {unspec})")
    print(f"   > label 분포: {positive_class}={effective_counts.get(positive_class, 0)}, {NEGATIVE_CLASS}={effective_counts.get(NEGATIVE_CLASS, 0)}")

    out = {"labels": [NEGATIVE_CLASS, positive_class], "database": database}
    return out


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    clips, excluded = gather_all_clips()
    items = split_if_needed(clips)

    for positive in INCIDENT_CLASSES:
        data = build_database_json(items, positive)
        out_json = OUT_DIR / f"annotation_{positive}.json"
        print(f"[INFO] JSON 저장 경로: {out_json}")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        subset_counts = {"training": 0, "validation": 0, "testing": 0, "unspecified": 0}
        for v in data["database"].values():
            key = v["subset"] if v["subset"] in subset_counts else "unspecified"
            subset_counts[key] = subset_counts.get(key, 0) + 1

        print("[완료] 저장 완료")
        print("subset 통계:", subset_counts)
        print("라벨 맵:", {NEGATIVE_CLASS: 0, positive: 1})
        print(f"기록된 n_frames(고정): {FRAME_COUNT}\n")

    print("=== 제외된 클립 목록 (20프레임 아님) ===")
    for clip, n in excluded:
        print(f"{clip} → {n} frames")
