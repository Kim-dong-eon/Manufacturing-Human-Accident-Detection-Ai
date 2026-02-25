import argparse
from pathlib import Path
from typing import Iterable

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="20프레임 미만/초과 클립 검사기")
    parser.add_argument(
        "root",
        type=Path,
        nargs="+",
        help="검사할 루트 디렉터리(여러 개 지정 가능)",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=20,
        help="기준 프레임 수 (기본 20)",
    )
    parser.add_argument(
        "--ext",
        nargs="*",
        default=sorted(IMAGE_EXTS),
        help="인식할 이미지 확장자 목록",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="출력할 최대 결과 수 (기본 전체)",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="기준을 만족하지 않는 클립 디렉터리를 삭제",
    )
    return parser.parse_args()


def is_image(path: Path, exts: Iterable[str]) -> bool:
    return path.is_file() and path.suffix.lower() in exts


def count_frames(folder: Path, exts: set[str]) -> int:
    return sum(1 for item in folder.iterdir() if is_image(item, exts))


def scan(root: Path, expected_frames: int, exts: set[str]) -> list[tuple[Path, int]]:
    results: list[tuple[Path, int]] = []
    for clip_dir in root.rglob("*"):
        if not clip_dir.is_dir():
            continue
        frame_count = count_frames(clip_dir, exts)
        if frame_count == 0:
            continue
        if frame_count != expected_frames:
            results.append((clip_dir, frame_count))
    return results


def main() -> None:
    args = parse_args()
    exts = {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in args.ext}
    expected = args.frames

    all_results: list[tuple[Path, int]] = []
    for root in args.root:
        if not root.exists():
            print(f"[경고] 존재하지 않는 경로: {root}")
            continue
        print(f"[검사] 루트 디렉터리: {root}")
        results = scan(root, expected, exts)
        all_results.extend(results)

    if not all_results:
        print(f"[완료] 모든 클립이 {expected} 프레임 기준을 만족합니다.")
        return

    all_results.sort(key=lambda x: x[0])
    total = len(all_results)
    print(f"[결과] {total}개의 클립이 기준({expected})과 다릅니다.")

    limit = args.limit or total
    for idx, (clip, count) in enumerate(all_results[:limit], start=1):
        print(f" {idx:4d}. {clip} → {count} frames")

    if args.delete:
        import shutil

        for clip, _ in all_results:
            shutil.rmtree(clip, ignore_errors=True)
        print(f"[삭제 완료] 총 {total}개 클립을 제거했습니다.")


if __name__ == "__main__":
    main()
