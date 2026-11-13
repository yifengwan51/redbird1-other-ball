#!/usr/bin/env bash
# Install an isolated FFmpeg runtime for this checkout.  Nothing is written to
# the system PATH or to another project.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${MOTRACK_PYTHON:-python3}"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
"$PYTHON" -m pip install --disable-pip-version-check --target "$STAGE/pkg" imageio-ffmpeg
BIN="$(find "$STAGE/pkg" -type f -name 'ffmpeg-linux-*' -perm -u+x | head -n 1)"
if [[ -z "$BIN" ]]; then
  echo "Could not find the Linux FFmpeg executable in imageio-ffmpeg." >&2
  exit 2
fi
mkdir -p "$ROOT/tools/ffmpeg/bin"
install -m 755 "$BIN" "$ROOT/tools/ffmpeg/bin/ffmpeg"
"$ROOT/tools/ffmpeg/bin/ffmpeg" -version | head -n 1
