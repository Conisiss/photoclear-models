# photoclear-models

ONNX model files downloaded on first use by the **ภาพชัด** photo app.
All processing happens on the user's device — photos are never uploaded.

| File (release asset) | Source | License |
|---|---|---|
| `realesrgan-x4plus-t256-fp16.onnx` | [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) `RealESRGAN_x4plus.pth` (v0.1.0) | BSD-3-Clause ([LICENSE](LICENSE)) |

## Conversion

- Architecture: RRDBNet (23 blocks, 64 features), ×4.
- Input `input`: float32 `[1, 3, 256, 256]` RGB in 0..1. Output `output`: float32 `[1, 3, 1024, 1024]`.
- Weights stored as FP16 (I/O kept float32). The nearest-neighbour ×2 upsample is expressed as
  channel repeat + PixelShuffle (bit-exact with `F.interpolate(mode="nearest")`).
- Source checkpoint SHA-256: `4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1`

## Checksums

The app verifies size and SHA-256 after download and discards the file if they don't match.

| File | Bytes | SHA-256 |
|---|---|---|
| `realesrgan-x4plus-t256-fp16.onnx` | 33736881 | `9afe00a4c0fe8457fa9b24237505411a234c2c343b3caf38d0cc227f2e9c1e9e` |

Real-ESRGAN: Xintao Wang, Liangbin Xie, Chao Dong, Ying Shan —
*Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data* (ICCVW 2021).
