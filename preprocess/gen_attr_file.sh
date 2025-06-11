#!/usr/bin/env bash
#
# gen_attr_file.sh
#
# 用法：
#   ./gen_attr_file.sh output.txt folder suffix1 [suffix2 ...]
#
# 示例：
#   ./gen_attr_file.sh list_attr_myraw.txt /path/to/images A B

if [ $# -lt 3 ]; then
  echo "Usage: $0 <output.txt> <folder> <suffix1> [suffix2 ...]" >&2
  exit 1
fi

out="$1"; shift
dir="$1"; shift
suffixes=("$@")
num_suffixes=${#suffixes[@]}

if [ ! -d "$dir" ]; then
  echo "Error: '$dir' is not a directory." >&2
  exit 1
fi

# 1. 收集所有文件（不含子目录），按文件名排序
mapfile -t files < <(find "$dir" -maxdepth 1 -type f | sort | xargs -n1 basename)
total=${#files[@]}

# 2. 写第一行：总文件数
echo "$total" > "$out"

# 3. 写第二行：属性名 header
header=""
for suf in "${suffixes[@]}"; do
  header+="camera${suf} "
done
echo "${header% }" >> "$out"

# 4. 逐文件写入：filename + 属性向量
for fname in "${files[@]}"; do
  # 去掉扩展名后提取后缀（假设形式为 name_SUFFIX.ext）
  base="${fname%.*}"
  suf="${base##*_}"

  # 构造属性向量：匹配该文件后缀的列为 1，其它列为 -1
  line="$fname"
  for s in "${suffixes[@]}"; do
    if [ "$suf" = "$s" ]; then
      val=1
    else
      val=-1
    fi
    line+=" $val"
  done

  echo "$line" >> "$out"
done

echo "Generated '$out' with $total entries and ${num_suffixes} attributes."
