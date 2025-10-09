#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
将文件夹中的图片重命名为 1..N（自动零填充），保留原扩展名。
Rename images in a folder to 1..N with zero-padding, preserving extensions.
"""

from pathlib import Path
import argparse
import uuid
import sys

IMAGE_EXTS = {
    # 通用图片 (common images)
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".gif",
    # RAW 通用 (generic RAW)
    ".raw", ".dng",
    # Canon
    ".cr2", ".cr3", ".crw",
    # Nikon
    ".nef", ".nrw",
    # Sony
    ".arw", ".sr2", ".srf",
    # Panasonic
    ".rw2",
    # Olympus / OM System
    ".orf",
    # Fujifilm
    ".raf",
    # Pentax / Ricoh
    ".pef",
    # Leica
    ".rwl",
    # Hasselblad
    ".3fr",
    # Phase One
    ".iiq",
    # Kodak / Others
    ".kdc", ".erf", ".mos", ".mef", ".srw",
    # 兼容“cv2”误写（compat unusual/typo）
    ".cv2",
}

def is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMAGE_EXTS

def natural_key(p: Path):
    """
    自然排序键（natural sort key with type tags）
    将文件名拆成数字片段与非数字片段，并为每段加类型标签，保证比较时不会出现 str 与 int 的直接比较。
    """
    import re
    name = p.name
    parts = re.findall(r'\d+|[^\d]+', name)
    tagged = []
    for part in parts:
        if part.isdigit():
            # 数字片段 -> (1, int值)
            tagged.append((1, int(part)))
        else:
            # 文本片段 -> (0, 小写字符串)
            tagged.append((0, part.lower()))
    return tuple(tagged)

def main():
    ap = argparse.ArgumentParser(description="Rename images in a folder to 1..N with zero-padding.")
    ap.add_argument("folder", help="目标文件夹（target folder）")
    ap.add_argument("--start", type=int, default=1, help="起始编号（start index），默认 1")
    ap.add_argument("--dry-run", action="store_true", help="试运行（dry run），仅打印不改名")
    ap.add_argument("--recursive", "-r", action="store_true", help="递归子目录（recursive）")
    args = ap.parse_args()

    root = Path(args.folder).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        print(f"错误（error）: 路径无效（invalid path） -> {root}", file=sys.stderr)
        sys.exit(1)

    files = []
    if args.recursive:
        for p in root.rglob("*"):
            if is_image(p):
                files.append(p)
    else:
        for p in root.iterdir():
            if is_image(p):
                files.append(p)

    if not files:
        print("未找到图片（no images found）")
        return

    # 按自然顺序排序（natural sort）
    files.sort(key=natural_key)

    # 统一宽度（width for zero-padding）
    start = args.start
    end = start + len(files) - 1
    width = len(str(end))

    # 生成目标名（target names），与源文件所在同目录
    plans = []
    for i, src in enumerate(files, start=start):
        dst = src.with_name(f"{str(i).zfill(width)}{src.suffix.lower()}")
        plans.append((src, dst))

    # Phase 0: 打印计划
    for src, dst in plans:
        if args.dry_run:
            print(f"[DRY] {src}  ->  {dst}")

    if args.dry_run:
        return

    # Phase 1: 源 -> 临时名（避免同名冲突）
    temp_map = {}
    for src, dst in plans:
        if src == dst:
            continue
        temp = src.with_name(f".__TMP_RENAME__{uuid.uuid4().hex}{src.suffix.lower()}")
        src.rename(temp)
        temp_map[temp] = dst

    # Phase 2: 临时名 -> 最终名
    for temp, final in temp_map.items():
        if final.exists():
            # 若更严格，可改成 raise FileExistsError
            final.unlink()  # 覆盖（overwrite）
        temp.rename(final)

    print(f"完成（done）: 共重命名 {len(plans)} 个文件（files）到 {start}..{end}，位宽（width）={width}")

if __name__ == "__main__":
    main()
