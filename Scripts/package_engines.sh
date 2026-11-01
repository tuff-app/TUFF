#!/usr/bin/env bash
# Build family-scoped helpers and authenticated archives locally. No upload.
set -euo pipefail
cd "$(dirname "$0")/.."
output="${1:-dist/local-engines}"
key="${2:-dist/private-engine-publisher/signing-key}"
version="${3:-1.0.0}"
# TUFF_ENGINE_INPUT_DIR overrides the staging directory. TUFF_ENGINE_SCRATCH_PATH selects a separate SwiftPM build directory so family builds
# do not disturb the default one.
scratch=()
if [[ -n "${TUFF_ENGINE_SCRATCH_PATH:-}" ]]; then scratch=(--scratch-path "$TUFF_ENGINE_SCRATCH_PATH"); fi
inputs="${TUFF_ENGINE_INPUT_DIR:-dist/engine-build-inputs}"
mkdir -p "$output" "$inputs"
output="$(cd "$output" && pwd)"
input="$(cd "$inputs" && pwd)"
for family in gemma-affine qwen-hybrid-affine gpt-oss-mxfp4 minimax-affine flint-affine; do
  TUFF_ENGINE_FAMILY="$family" swift build -c release ${scratch[@]+"${scratch[@]}"} --product TUFFInferenceEngine
  binary="$(TUFF_ENGINE_FAMILY="$family" swift build -c release ${scratch[@]+"${scratch[@]}"} --show-bin-path)"
  mkdir -p "$input/$family"
  for product in TUFFInferenceEngine TUFF_TUFFEngine.bundle TUFF_TUFFAppCore.bundle swift-transformers_Hub.bundle swift-crypto_Crypto.bundle; do
    rm -rf "$input/$family/$product"
    ditto "$binary/$product" "$input/$family/$product"
  done
  echo "built $family"
done
# Restore the normal host configuration, whose metadata registry includes every
# supported model. The package builder contains no inference code.
swift build -c release ${scratch[@]+"${scratch[@]}"} --product TUFFEnginePackage
binary="$(swift build -c release ${scratch[@]+"${scratch[@]}"} --show-bin-path)"
package_arguments=("$input" "$output" "$key" "$version")
if [[ -n "${4:-}" ]]; then package_arguments+=("$4"); fi
"$binary/TUFFEnginePackage" "${package_arguments[@]}"
