# photoclear-models

ONNX model files downloaded on first use by the **Jinder Photo** app.
All processing happens on the user's device — photos are never uploaded.

| File (release asset) | Release | Source | License |
|---|---|---|---|
| `realesrgan-x4plus-t256-fp16.onnx` | `models-v1` | [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) `RealESRGAN_x4plus.pth` (v0.1.0) | BSD-3-Clause ([LICENSE](LICENSE)) |
| `lama-512-w16.onnx` | `models-v2` | [LaMa](https://github.com/advimman/lama) big-lama, via the ONNX port [Carve/LaMa-ONNX](https://huggingface.co/Carve/LaMa-ONNX) | Apache-2.0 ([LICENSE-LaMa](LICENSE-LaMa)) |

## Conversion

### realesrgan-x4plus-t256-fp16.onnx

- Architecture: RRDBNet (23 blocks, 64 features), ×4.
- Input `input`: float32 `[1, 3, 256, 256]` RGB in 0..1. Output `output`: float32 `[1, 3, 1024, 1024]`.
- Weights stored as FP16 (I/O kept float32). The nearest-neighbour ×2 upsample is expressed as
  channel repeat + PixelShuffle (bit-exact with `F.interpolate(mode="nearest")`).
- Source checkpoint SHA-256: `4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1`

### lama-512-w16.onnx (modified from the original)

- Source: `lama_fp32.onnx` from Carve/LaMa-ONNX, SHA-256
  `1faef5301d78db7dda502fe59966957ec4b79dd64e16f03ed96913c7a4eb68d6`.
- Changes: fixed batch size 1; input/output wrapper so it matches the app's MI-GAN interface;
  constant folding; Einsum (DFT matrix products) rewritten as MatMul; BatchNormalization rewritten as
  Mul + Add with precomputed scale/bias; weights stored as FP16 and cast to float32 at load
  (the first 7×7 convolution is kept float32). Computation is float32.
- Input `input`: float32 `[1, 4, 512, 512]` = concat(mask − 0.5, image × mask), image in −1..1, mask 1 = keep.
  Output `output`: float32 `[1, 3, 512, 512]` in −1..1.
- Max difference from the original model: 2.7 / 255 (mean 0.03) on test inputs.

## Checksums

The app verifies size and SHA-256 after download and discards the file if they don't match.

| File | Bytes | SHA-256 |
|---|---|---|
| `realesrgan-x4plus-t256-fp16.onnx` | 33736881 | `9afe00a4c0fe8457fa9b24237505411a234c2c343b3caf38d0cc227f2e9c1e9e` |
| `lama-512-w16.onnx` | 105125307 | `8664161dbb317f00a2a74a88fe90058c6137bc805d70a75167383d7acb60040b` |

Real-ESRGAN: Xintao Wang, Liangbin Xie, Chao Dong, Ying Shan —
*Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data* (ICCVW 2021).

LaMa: Roman Suvorov, Elizaveta Logacheva, Anton Mashikhin, Anastasia Remizova, Arsenii Ashukha,
Aleksei Silvestrov, Naejin Kong, Harshith Goka, Kiwoong Park, Victor Lempitsky —
*Resolution-robust Large Mask Inpainting with Fourier Convolutions* (WACV 2022).
