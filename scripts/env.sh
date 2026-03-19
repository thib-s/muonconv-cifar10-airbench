#!/usr/bin/env bash

set -euo pipefail

cuda_version="${1:-12.8}"
cudnn_version="${2:-9.8.0}"

cuda_root="/usr/local/cuda-${cuda_version}"
cudnn_root="/usr/local/cudnn/${cuda_version}-v${cudnn_version}"
venv_activate=".venv/bin/activate"

if [[ ! -d "${cuda_root}" ]]; then
  echo "Unsupported CUDA version: ${cuda_version} (${cuda_root} not found)." >&2
  return 1 2>/dev/null || exit 1
fi

if [[ ! -d "${cudnn_root}" ]]; then
  echo "Unsupported CUDA/CUDNN version: ${cuda_version}/${cudnn_version} (${cudnn_root} not found)." >&2
  return 1 2>/dev/null || exit 1
fi

if [[ ! -f "${venv_activate}" ]]; then
  echo "Virtual environment activation script not found at ${venv_activate}." >&2
  return 1 2>/dev/null || exit 1
fi

export CUDA_VERSION="${cuda_version}"
export CUDNN_VERSION="${cudnn_version}"
export CUDA_HOME="${cuda_root}"

case ":${PATH}:" in
  *:/usr/local/cuda-*/bin:*)
    PATH="$(printf '%s' "${PATH}" | sed -E "s#/usr/local/cuda(-[0-9]+\.[0-9]+)?/bin#${cuda_root}/bin#g")"
    ;;
  *)
    PATH="${cuda_root}/bin:${PATH}"
    ;;
esac
export PATH

if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
  LD_LIBRARY_PATH="$(printf '%s' "${LD_LIBRARY_PATH}" | sed -E "s#/usr/local/cuda(-[0-9]+\.[0-9]+)?/lib64#${cuda_root}/lib64#g")"
  LD_LIBRARY_PATH="$(printf '%s' "${LD_LIBRARY_PATH}" | sed -E "s#/usr/local/cudnn/[0-9]+\.[0-9]+-v[0-9.]+/cuda/lib64#${cudnn_root}/cuda/lib64#g")"
else
  LD_LIBRARY_PATH=""
fi

case ":${LD_LIBRARY_PATH}:" in
  *:"${cuda_root}/lib64":*) ;;
  *)
    LD_LIBRARY_PATH="${cuda_root}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    ;;
esac

case ":${LD_LIBRARY_PATH}:" in
  *:"${cudnn_root}/cuda/lib64":*) ;;
  *)
    LD_LIBRARY_PATH="${cudnn_root}/cuda/lib64:${LD_LIBRARY_PATH}"
    ;;
esac
export LD_LIBRARY_PATH

echo "Using CUDA ${cuda_version} at ${cuda_root}"
echo "Using cuDNN ${cudnn_version} at ${cudnn_root}"

# Intended to be sourced so the activated virtualenv persists in the caller shell.
source "${venv_activate}"
