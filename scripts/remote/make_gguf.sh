#!/bin/bash
# Convert a merged HF model to GGUF (f16) and optionally quantize.
# Best-effort: exits non-zero on failure but never takes down the pipeline.
set -uo pipefail
IN="$(readlink -f "${1:?input model dir}")"
OUT="$(readlink -f "$(mkdir -p "${2:?output dir}" && echo "${2}")")"
echo "IN=$IN"; echo "OUT=$OUT"
[ -d "$IN" ] || { echo "input model dir missing: $IN"; exit 1; }

cd /opt
if [ ! -d llama.cpp ]; then
  git clone --depth 1 https://github.com/ggerganov/llama.cpp.git || exit 1
fi
cd /opt/llama.cpp
pip install -q -r requirements/requirements-convert_hf_to_gguf.txt 2>/dev/null \
  || pip install -q gguf sentencepiece protobuf numpy || true

python3 convert_hf_to_gguf.py "$IN" --outfile "$OUT/model-f16.gguf" --outtype f16 || exit 2

if command -v cmake >/dev/null 2>&1; then
  cmake -B build -DGGML_CUDA=OFF >/dev/null 2>&1 \
    && cmake --build build --config Release -j"$(nproc)" >/dev/null 2>&1
  if [ -x build/bin/llama-quantize ]; then
    build/bin/llama-quantize "$OUT/model-f16.gguf" "$OUT/model-Q8_0.gguf" Q8_0 || true
    build/bin/llama-quantize "$OUT/model-f16.gguf" "$OUT/model-Q4_K_M.gguf" Q4_K_M || true
  fi
fi
ls -la "$OUT"