#!/usr/bin/env bash
set -euo pipefail

# 使用方式：
#   bash download_vlm_safety_benchmarks.sh /your/target/path
# 若不传参数，则默认使用当前目录下的 ./vlm_safety_benchmarks
TARGET_DIR="/data/data-pool/dingyifan/GeminiEvaluation/workspace/data/raw"
mkdir -p "$TARGET_DIR"

echo "[INFO] All datasets will be stored under: $TARGET_DIR"
echo "[INFO] Make sure 'huggingface-cli' is installed and you have run 'huggingface-cli login'."
echo

export HF_ENDPOINT="https://hf-mirror.com"

hf_download () {
  local repo_id="$1"   # huggingface 数据集 ID
  local subdir="$2"    # 本地存放子目录名

  local out_dir="$TARGET_DIR/$subdir"
  # if [ -d "$out_dir" ]; then
  #   echo "[SKIP] $subdir already exists at $out_dir"
  #   return 0
  # fi

  echo "[HF] Downloading dataset '$repo_id' -> $out_dir"
  huggingface-cli download "$repo_id" \
    --token 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 \
    --repo-type dataset \
    --local-dir "$out_dir" \
    --local-dir-use-symlinks False \
    # --force-download
  echo
}

git_clone () {
  local repo_url="$1"
  local subdir="$2"

  local out_dir="$TARGET_DIR/$subdir"
  if [ -d "$out_dir" ]; then
    echo "[SKIP] $subdir already exists at $out_dir"
    return 0
  fi

  echo "[GIT] Cloning $repo_url -> $out_dir"
  git clone "$repo_url" "$out_dir"
  echo
}

#############################
# 1. UNICORN / vllm_safety_evaluation
#    - 包含 OODCV-VQA, Sketchy-VQA 等
#############################
hf_download "PahaII/vllm_safety_evaluation" "UNICORN_vllm_safety_evaluation"

#############################
# 2. JailbreakV-28K
#############################
hf_download "JailbreakV-28K/JailBreakV-28k" "JailbreakV-28k"

#############################
# 3. MIS（Train & Test）
#############################
hf_download "Tuwhy/MIS_Test"  "MIS_Test"

#############################
# 4. VLJailbreakBench / VLBreakBench
#############################
hf_download "wang021/VLBreakBench" "VLJailbreakBench"

#############################
# 5. USB（Unified Safety Benchmark）
#############################
hf_download "cgjacklin/USB" "USB"

#############################
# 6. MemeSafetyBench（Meme-Safety-Bench）
#    同样是 gated dataset，需先在网页上同意条款
#############################
hf_download "oneonlee/Meme-Safety-Bench" "MemeSafetyBench"

#############################
# 7. MM-SafetyBench
#############################
hf_download "PKU-Alignment/MM-SafetyBench" "MM-SafetyBench"

#############################
# 8. Argus
#############################
git_clone "https://github.com/evigbyen/argus" "Argus"

#############################
# 9. MSSBench
#############################
hf_download "kzhou35/mssbench" "MSSBench"

#############################
# 10. SIUO
#############################
hf_download "sinwang/SIUO" "SIUO"

echo "[DONE] All available VLM safety benchmarks in this script have been downloaded."