#!/usr/bin/env bash
# merge_with_suffix.sh
# 用法：./merge_with_suffix.sh <dirA> <dirB> <targetDir>

if [ $# -ne 3 ]; then
  echo "Usage: $0 <dirA> <dirB> <targetDir>"
  exit 1
fi

dirA="$1"
dirB="$2"
dst="$3"

mkdir -p "$dst"

# 处理 A 目录：加 _A 后缀
for src in "$dirA"/*; do
  fn=$(basename "$src")
  ext="${fn##*.}"
  name="${fn%.*}"
  cp "$src" "$dst/${name}_A.${ext}"
done

# 处理 B 目录：加 _B 后缀
for src in "$dirB"/*; do
  fn=$(basename "$src")
  ext="${fn##*.}"
  name="${fn%.*}"
  cp "$src" "$dst/${name}_B.${ext}"
done
