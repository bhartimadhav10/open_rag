"""One-time script: download bge-small ONNX FP32 and produce an INT8 variant
under ./data/onnx-bge-small-int8/. Run once, then use `--embed-precision int8`
from the bench.
"""
from pathlib import Path
from huggingface_hub import snapshot_download
from optimum.onnxruntime import ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from transformers import AutoTokenizer
import shutil
import sys


MODEL_ID = "BAAI/bge-small-en-v1.5"
OUT_DIR = Path("data/onnx-bge-small-int8")


def main():
    OUT_DIR.parent.mkdir(parents=True, exist_ok=True)
    if OUT_DIR.exists():
        print(f"[quantize] {OUT_DIR} exists — removing for clean rebuild")
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True)

    print(f"[quantize] downloading {MODEL_ID} (ONNX FP32 + tokenizer)...")
    src_dir = Path(snapshot_download(
        MODEL_ID,
        allow_patterns=[
            "onnx/model.onnx",
            "onnx/model.onnx_data",
            "config.json",
            "tokenizer*",
            "vocab.txt",
            "special_tokens_map.json",
            "sentence_bert_config.json",
            "1_Pooling/*",
            "2_Normalize/*",
            "modules.json",
            "config_sentence_transformers.json",
        ],
    ))

    # Copy the non-onnx files (tokenizer, pooling, etc) to the output dir
    for item in src_dir.iterdir():
        if item.name == "onnx":
            continue
        dst = OUT_DIR / item.name
        if item.is_dir():
            shutil.copytree(item, dst)
        else:
            shutil.copy2(item, dst)

    # Copy the FP32 onnx model into out_dir/onnx/model.onnx so the quantizer finds it
    (OUT_DIR / "onnx").mkdir(exist_ok=True)
    shutil.copy2(src_dir / "onnx" / "model.onnx", OUT_DIR / "onnx" / "model.onnx")
    # Some exports have a companion _data file for large weights
    data_file = src_dir / "onnx" / "model.onnx_data"
    if data_file.exists():
        shutil.copy2(data_file, OUT_DIR / "onnx" / "model.onnx_data")

    print(f"[quantize] quantizing to INT8 (dynamic, AVX-512 VNNI)...")
    quantizer = ORTQuantizer.from_pretrained(OUT_DIR / "onnx", file_name="model.onnx")
    # avx512_vnni is the best config for recent Intel CPUs; falls back cleanly
    # on older hardware. per_channel=True gives best accuracy for weights.
    cfg = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=True)
    quantizer.quantize(save_dir=OUT_DIR / "onnx", quantization_config=cfg)
    # optimum names the output `model_quantized.onnx`; rename for clarity
    produced = OUT_DIR / "onnx" / "model_quantized.onnx"
    if not produced.exists():
        print("[quantize] ERROR: optimum did not produce model_quantized.onnx", file=sys.stderr)
        sys.exit(1)
    target = OUT_DIR / "onnx" / "model_qint8_avx512_vnni.onnx"
    produced.rename(target)
    print(f"[quantize] wrote {target}  ({target.stat().st_size // 1024 // 1024} MB)")
    print(f"[quantize] done — use `--embed-precision int8` now.")


if __name__ == "__main__":
    main()
