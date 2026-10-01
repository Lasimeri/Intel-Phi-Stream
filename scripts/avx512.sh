# avx512.sh: where the co-processor repository (Intel-Phi-AVX512) is: its
# `scripts/phi-ggml.sh` starts the card workers and names the backend
# `libggml_phi.so` to ggml. Sourced by phi-stream.sh. In order: the
# environment, a checkout next to this one, one in $HOME; under its
# clone's name (Intel-Phi-AVX512) or the spaced one. Not finding it is
# not an error: the stream then runs on the GPU and the host alone, and
# PHI_AVX512_ROOT stays empty. See avx512.md.
if [ -z "${PHI_AVX512_ROOT:-}" ]; then
    for base in "$(dirname "$0")/../.." "${HOME:-/nonexistent}"; do
        for name in "Intel-Phi-AVX512" "Intel Phi AVX-512"; do
            if [ -f "$base/$name/scripts/phi-ggml.sh" ]; then
                PHI_AVX512_ROOT=$(cd "$base/$name" && pwd)
                break 2
            fi
        done
    done
fi
if [ -n "${PHI_AVX512_ROOT:-}" ] && [ ! -f "$PHI_AVX512_ROOT/scripts/phi-ggml.sh" ]; then
    echo "$0: PHI_AVX512_ROOT=$PHI_AVX512_ROOT holds no scripts/phi-ggml.sh" >&2
    exit 1
fi
export PHI_AVX512_ROOT
