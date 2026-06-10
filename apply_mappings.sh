#!/usr/bin/env bash
set -euo pipefail

# apply_mappings.sh
# 根据当前目录下 mappings/*.map 文件，搜索系统设备并为匹配的设备创建符号链接
# 链接位置默认: /dev/ur5e/<logical>

MAPPINGS_DIR="./mappings"
SYMLINK_DIR="/dev/ur5e"

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || { echo "需要命令: $1" >&2; exit 1; }
}

require_cmd udevadm
require_cmd readlink

dry_run=0
if [ "${1:-}" = --dry-run ]; thenDEVNODE
  dry_run=1
fi

if [ ! -d "$MAPPINGS_DIR" ]; then
  echo "没有 mappings 目录: $MAPPINGS_DIR" >&2
  exit 1
fi

mkdir -p "$SYMLINK_DIR"

match_props() {
  # $1 = mapping file path
  # $2 = device node (e.g. /dev/ttyUSB0)
  local mapfile="$1" devnode="$2"
  # load mapping
  declare -A m
  while IFS='=' read -r k v; do
    case "$k" in
      "#"*) continue;;
      "") continue;;
    esac
    k=$(printf "%s" "$k" | tr -d '\r' )
    v=$(printf "%s" "$v" | tr -d '\r' )
    m["$k"]="$v"
  done < <(grep -v '^\s*#' "$mapfile" || true)

  props=$(udevadm info --query=property --name="$devnode" 2>/dev/null || true)
  if [ -z "$props" ]; then
    return 1
  fi

  # Prefer exact ID_SERIAL_SHORT match
  if [ -n "${m[ID_SERIAL_SHORT]:-}" ]; then
    if printf "%s" "$props" | grep -q "^ID_SERIAL_SHORT=${m[ID_SERIAL_SHORT]}$"; then
      return 0
    else
      return 1
    fi
  fi

  # Then ID_SERIAL
  if [ -n "${m[ID_SERIAL]:-}" ]; then
    if printf "%s" "$props" | grep -q "^ID_SERIAL=${m[ID_SERIAL]}$"; then
      return 0
    else
      return 1
    fi
  fi

  # Then vendor+model
  if [ -n "${m[ID_VENDOR_ID]:-}" ] && [ -n "${m[ID_MODEL_ID]:-}" ]; then
    if printf "%s" "$props" | grep -q "^ID_VENDOR_ID=${m[ID_VENDOR_ID]}$" && printf "%s" "$props" | grep -q "^ID_MODEL_ID=${m[ID_MODEL_ID]}$"; then
      return 0
    else
      return 1
    fi
  fi

  # Fallback: match DEVNAME
  if [ -n "${m[DEVNAME]:-}" ]; then
    local base
    base=$(basename "${m[DEVNAME]}")
    if [ "$(basename "$devnode")" = "$base" ]; then
      return 0
    fi
  fi

  return 1
}

apply_map_to_dev() {
  local mapfile="$1" devnode="$2"
  local logical
  logical=$(grep -E '^LOGICAL=' "$mapfile" | head -n1 | cut -d'=' -f2-)
  [ -n "$logical" ] || logical=$(basename "$mapfile" .map)

  target="$SYMLINK_DIR/$logical"

  echo "匹配: $devnode -> $target"
  if [ $dry_run -eq 1 ]; then
    return 0
  fi

  if [ ! -e "$devnode" ]; then
    echo "设备节点不存在: $devnode" >&2
    return 1
  fi

  # remove existing symlink if exists and points elsewhere
  if [ -L "$target" ]; then
    cur=$(readlink -f "$target") || true
    if [ "$cur" != "$(readlink -f "$devnode")" ]; then
      if [ "$EUID" -ne 0 ]; then
        sudo rm -f "$target"
      else
        rm -f "$target"
      fi
    fi
  elif [ -e "$target" ]; then
    echo "目标路径存在且不是符号链接: $target，跳过" >&2
    return 1
  fi

  # create symlink
  if [ "$EUID" -ne 0 ]; then
    sudo ln -sfn "$(readlink -f "$devnode")" "$target"
  else
    ln -sfn "$(readlink -f "$devnode")" "$target"
  fi
}

echo "应用 mappings 中的配置到系统设备（symlink 目标: $SYMLINK_DIR）"
echo

shopt -s nullglob
mapfiles=("$MAPPINGS_DIR"/*.map)
if [ ${#mapfiles[@]} -eq 0 ]; then
  echo "未找到映射文件于 $MAPPINGS_DIR" >&2
  exit 1
fi

# enumerate candidate devices: serial and video
candidates=()
for d in /dev/serial/by-id/*; do
  [ -e "$d" ] || continue
  candidates+=("$(readlink -f "$d")")
done
for d in /dev/ttyUSB* /dev/ttyACM* /dev/video*; do
  [ -e "$d" ] || continue
  # avoid duplicates
  real=$(readlink -f "$d")
  if ! printf '%s\n' "${candidates[@]}" | grep -Fxq "$real"; then
    candidates+=("$real")
  fi
done

if [ ${#candidates[@]} -eq 0 ]; then
  echo "未发现候选设备（/dev/ttyUSB*, /dev/ttyACM*, /dev/video* 或 /dev/serial/by-id）" >&2
fi

for mapfile in "${mapfiles[@]}"; do
  for dev in "${candidates[@]}"; do
    if match_props "$mapfile" "$dev"; then
      apply_map_to_dev "$mapfile" "$dev"
      # once matched, stop searching other devices for this map
      break
    fi
  done
done

echo "完成。要使规则在设备重新插入后持续生效，请将映射转换为 udev 规则或在开机脚本中运行本脚本。"

exit 0
