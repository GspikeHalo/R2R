#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
将 CSV 中列出的文件归档为 paired，其余归档为 unpaired。（archive to paired/unpaired）

输出结构（Output structure）：外层相机（camera），内层 raw/jpg：
paired/
  huawei/
    raw/
    jpg/
  nikon/
    raw/
    jpg/
  iphone/
    raw/
    jpg/
  samsung/
    raw/
    jpg/
unpaired/
  huawei/
    raw/
    jpg/
  nikon/
    raw/
    jpg/
  iphone/
    raw/
    jpg/
  samsung/
    raw/
    jpg/
"""

import csv
import os
import shutil
from pathlib import Path

# ======== 用户可编辑（User-editable） ========
# 根目录（root dir）：包含各相机的源目录（source folders）与 CSV
ROOT = Path(r"C:\local_file\Research\R2R\r2r_odb\db").resolve()

# 新的 CSV 文件（new CSV path），需含以下列：
# huawei, iphone, Nikon, Samsung, huawei_raw, iphone_raw, Nikon_raw, samsung_raw
CSV_FILE = ROOT / "images_with_raw_columns.csv"

COPY_INSTEAD_OF_MOVE = True  # True=复制 copy（安全 safe），False=移动 move（迁移 migrate）

# 相机配置（camera config）：<camera>/{raw,rgb} 为源目录（source dirs）
# 注意：CSV 列名区分大小写（column names are case-sensitive）。
CAMERAS = {
    # Huawei
    "huawei": {
        "jpg_dir": ROOT / "huawei" / "rgb",
        "raw_dir": ROOT / "huawei" / "raw",
        "jpg_exts": {".jpg", ".jpeg", ".JPG", ".JPEG"},
        "raw_exts": {".dng", ".DNG"},
        "csv_jpg_col": "huawei",
        "csv_raw_col": "huawei_raw",
        "raw_from_jpg_ext": ".dng",
    },
    # Nikon
    "nikon": {
        "jpg_dir": ROOT / "nikon" / "rgb",
        "raw_dir": ROOT / "nikon" / "raw",
        "jpg_exts": {".jpg", ".jpeg", ".JPG", ".JPEG"},
        "raw_exts": {".nef", ".NEF"},
        "csv_jpg_col": "Nikon",         # 注意 CSV 列名首字母大写（capital N）
        "csv_raw_col": "Nikon_raw",
        "raw_from_jpg_ext": ".NEF",
    },
    # iPhone
    "iphone": {
        "jpg_dir": ROOT / "iphone" / "rgb",
        "raw_dir": ROOT / "iphone" / "raw",
        "jpg_exts": {".jpg", ".jpeg", ".JPG", ".JPEG"},
        "raw_exts": {".dng", ".DNG"},
        "csv_jpg_col": "iphone",        # CSV 列名为全小写 iphone
        "csv_raw_col": "iphone_raw",
        "raw_from_jpg_ext": ".DNG",     # 你要求 iPhone RAW 后缀用 .DNG（大写）
    },
    # Samsung
    "samsung": {
        "jpg_dir": ROOT / "samsung" / "rgb",
        "raw_dir": ROOT / "samsung" / "raw",
        "jpg_exts": {".jpg", ".jpeg", ".JPG", ".JPEG"},
        "raw_exts": {".dng", ".DNG"},
        "csv_jpg_col": "Samsung",       # 注意 CSV 列名首字母大写（capital S）
        "csv_raw_col": "samsung_raw",   # CSV 列名为小写 samsung_raw
        "raw_from_jpg_ext": ".dng",     # 你要求 Samsung RAW 后缀用 .dng（小写）
    },
}

# 输出根目录（output roots）
OUT_PAIRED_ROOT = ROOT / "paired"
OUT_UNPAIRED_ROOT = ROOT / "unpaired"
# ============================================


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def load_csv_rows(csv_path: Path):
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        rows = [row for row in reader]
    if not rows:
        raise ValueError("CSV has no data rows.")
    return header, rows


def replace_ext(name: str, new_ext: str) -> str:
    root, _ = os.path.splitext(name)
    return root + new_ext


def index_files(dir_path: Path, allowed_exts: set):
    """非递归索引 non-recursive index：{basename_lower: Path}"""
    mapping = {}
    if not dir_path.exists():
        return mapping
    allowed_lower = {e.lower() for e in allowed_exts}
    for p in dir_path.iterdir():
        if p.is_file() and (p.suffix.lower() in allowed_lower):
            mapping[p.name.lower()] = p
    return mapping


def copy_or_move(src: Path, dst: Path, copy: bool):
    ensure_dir(dst.parent)
    if copy:
        shutil.copy2(src, dst)
    else:
        shutil.move(src, dst)


def main():
    # 读 CSV（load CSV）
    header, rows = load_csv_rows(CSV_FILE)

    # 预建输出目录（pre-create outputs）
    for cam in CAMERAS:
        for kind in ("raw", "jpg"):
            ensure_dir(OUT_PAIRED_ROOT / cam / kind)
            ensure_dir(OUT_UNPAIRED_ROOT / cam / kind)

    # 建立磁盘索引（disk index）
    disk_index = {}
    for cam, cfg in CAMERAS.items():
        disk_index[(cam, "jpg")] = index_files(cfg["jpg_dir"], cfg["jpg_exts"])
        disk_index[(cam, "raw")] = index_files(cfg["raw_dir"], cfg["raw_exts"])

    # 从 CSV 解析“期望文件名集合”（expected sets）
    expected = {(cam, "jpg"): set() for cam in CAMERAS}
    expected.update({(cam, "raw"): set() for cam in CAMERAS})

    for cam, cfg in CAMERAS.items():
        col_jpg = cfg["csv_jpg_col"]
        col_raw = cfg["csv_raw_col"]
        for row in rows:
            jpg_name = (row.get(col_jpg) or "").strip()
            if jpg_name:
                expected[(cam, "jpg")].add(jpg_name.lower())
                # RAW 优先取 CSV 的 *_raw 列，否则按 jpg 改后缀推断（infer from jpg）
                raw_name = (row.get(col_raw) or "").strip()
                if not raw_name and jpg_name:
                    raw_name = replace_ext(jpg_name, cfg["raw_from_jpg_ext"])
                if raw_name:
                    expected[(cam, "raw")].add(raw_name.lower())

    # === CSV 命中的文件 → paired/<camera>/{jpg,raw}
    paired_counts = {(cam, "jpg"): 0 for cam in CAMERAS}
    paired_counts.update({(cam, "raw"): 0 for cam in CAMERAS})
    missing = []  # 记录 CSV 中列出但磁盘未找到（listed in CSV but not on disk）

    for cam, cfg in CAMERAS.items():
        # JPG
        for name in sorted(expected[(cam, "jpg")]):
            src = disk_index[(cam, "jpg")].get(name)
            if src is None:
                missing.append((cam, "jpg", name))
                continue
            dst = OUT_PAIRED_ROOT / cam / "jpg" / src.name
            copy_or_move(src, dst, COPY_INSTEAD_OF_MOVE)
            paired_counts[(cam, "jpg")] += 1
        # RAW
        for name in sorted(expected[(cam, "raw")]):
            src = disk_index[(cam, "raw")].get(name)
            if src is None:
                missing.append((cam, "raw", name))
                continue
            dst = OUT_PAIRED_ROOT / cam / "raw" / src.name
            copy_or_move(src, dst, COPY_INSTEAD_OF_MOVE)
            paired_counts[(cam, "raw")] += 1

    # === 其余未在 CSV 的文件 → unpaired/<camera>/{jpg,raw}
    unpaired_counts = {(cam, "jpg"): 0 for cam in CAMERAS}
    unpaired_counts.update({(cam, "raw"): 0 for cam in CAMERAS})

    for cam, cfg in CAMERAS.items():
        # JPG others
        for name, src in disk_index[(cam, "jpg")].items():
            if name not in expected[(cam, "jpg")]:
                dst = OUT_UNPAIRED_ROOT / cam / "jpg" / src.name
                copy_or_move(src, dst, COPY_INSTEAD_OF_MOVE)
                unpaired_counts[(cam, "jpg")] += 1
        # RAW others
        for name, src in disk_index[(cam, "raw")].items():
            if name not in expected[(cam, "raw")]:
                dst = OUT_UNPAIRED_ROOT / cam / "raw" / src.name
                copy_or_move(src, dst, COPY_INSTEAD_OF_MOVE)
                unpaired_counts[(cam, "raw")] += 1

    # 汇总（summary）
    op = "Copied" if COPY_INSTEAD_OF_MOVE else "Moved"
    print("\n=== SUMMARY ===")
    print(f"{op} to paired/")
    for k, v in paired_counts.items():
        print(f"  {k}: {v}")
    print(f"{op} to unpaired/")
    for k, v in unpaired_counts.items():
        print(f"  {k}: {v}")

    if missing:
        print("\nMissing files (not found on disk):")
        for cam, kind, name in missing:
            print(f"  [{cam}/{kind}] {name}")
        logp = ROOT / "missing_files.log"
        with open(logp, "w", encoding="utf-8") as f:
            for cam, kind, name in missing:
                f.write(f"[{cam}/{kind}] {name}\n")
        print(f"\nMissing list saved to: {logp}")


if __name__ == "__main__":
    main()
