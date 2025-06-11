#!/usr/bin/env bash
# compare_names.sh
# 对比两个目录下所有文件（包括子目录）同名文件的数量，并列出文件名

if [ $# -ne 2 ]; then
  echo "用法：$0 <目录1> <目录2>"
  exit 1
fi

dir1="$1"
dir2="$2"

# 在各自目录中查找所有文件，提取文件名，去重并排序
files1=$(find "$dir1" -type f | sed 's#.*/##' | sort -u)
files2=$(find "$dir2" -type f | sed 's#.*/##' | sort -u)

# 找到交集：即两边都出现的文件名
common=$(comm -12 <(printf "%s\n" "$files1") <(printf "%s\n" "$files2"))

# 统计行数（排除空行）
count=$(printf "%s\n" "$common" | grep -cve '^$')

echo "共找到 $count 个重名文件："
