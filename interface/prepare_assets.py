"""One-time static assets preparation. No Node.js, no server inference."""
import argparse
import hashlib
import json
import shutil
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
SOURCE = PROJECT / "outputs/scifact_laptop"
TOOLS = HERE / ".build"


def download(url, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 0:
        return
    print(f"Downloading {target.name}", flush=True)
    request = urllib.request.Request(url, headers={"User-Agent": "SciFind-research-demo/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        target.write_bytes(response.read())


def frontend_assets():
    urls = {
        "vendor/react.mjs": "https://esm.sh/react@18.3.1/es2022/react.bundle.mjs",
        "vendor/react-dom.mjs": "https://esm.sh/react-dom@18.3.1/es2022/client.bundle.mjs?external=react",
        "vendor/htm.mjs": "https://cdn.jsdelivr.net/npm/htm@3.1.1/dist/htm.module.js",
        "vendor/transformers.js": "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.2/dist/transformers.web.js",
    }
    for path, url in urls.items():
        download(url, HERE / path)
    # Read the exact runtime version required by this Transformers.js release.
    package_path = TOOLS / "transformers-package.json"
    download("https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.2/package.json", package_path)
    package = json.loads(package_path.read_text())
    ort_version = package["dependencies"]["onnxruntime-web"].lstrip("^~")
    print("ONNX Runtime Web:", ort_version, flush=True)
    for filename in ["ort-wasm-simd-threaded.jsep.mjs", "ort-wasm-simd-threaded.jsep.wasm"]:
        download(f"https://cdn.jsdelivr.net/npm/onnxruntime-web@{ort_version}/dist/{filename}", HERE / "vendor/ort" / filename)
    download(f"https://cdn.jsdelivr.net/npm/onnxruntime-web@{ort_version}/dist/ort.webgpu.bundle.min.mjs", HERE / "vendor/ort/ort.webgpu.bundle.min.mjs")
    # Worker modules have no import-map support. Point both ONNX imports at the
    # same bundled runtime, so they share Tensor/InferenceSession implementations.
    transformers_file = HERE / "vendor/transformers.js"
    text = transformers_file.read_text(encoding="utf-8")
    for package_name in ["onnxruntime-common", "onnxruntime-web"]:
        text = text.replace(f'from "{package_name}"', 'from "./ort/ort.webgpu.bundle.min.mjs"')
    transformers_file.write_text(text, encoding="utf-8")
    react_dom = HERE / "vendor/react-dom.mjs"
    text = react_dom.read_text(encoding="utf-8").replace('"/react@18.3.1/es2022/react.mjs"', '"./react.mjs"')
    react_dom.write_text(text, encoding="utf-8")
    for name, url in {
        "react": "https://raw.githubusercontent.com/facebook/react/v18.3.1/LICENSE",
        "htm": "https://raw.githubusercontent.com/developit/htm/3.1.1/LICENSE",
        "tailwind": "https://raw.githubusercontent.com/tailwindlabs/tailwindcss/v3.4.17/LICENSE",
        "transformers-js": "https://raw.githubusercontent.com/huggingface/transformers.js/3.7.2/LICENSE",
        "onnxruntime": "https://raw.githubusercontent.com/microsoft/onnxruntime/v1.22.0/LICENSE",
    }.items():
        download(url, HERE / "vendor/licenses" / f"{name}.txt")
    download("https://github.com/tailwindlabs/tailwindcss/releases/download/v3.4.17/tailwindcss-windows-x64.exe", TOOLS / "tailwindcss.exe")
    print("Frontend libraries downloaded.", flush=True)


def export_data():
    import numpy as np
    data = HERE / "data"
    data.mkdir(exist_ok=True)
    ids = json.loads((SOURCE / "doc_ids.json").read_text(encoding="utf-8"))
    documents = json.loads((SOURCE / "documents.json").read_text(encoding="utf-8"))
    by_id = {doc["_id"]: doc for doc in documents}
    vectors = np.load(SOURCE / "document_embeddings.npy").astype("<f4")
    assert vectors.shape == (len(ids), 384)
    assert np.isfinite(vectors).all()
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-4)
    # Preserve exact embedding row order; omit duplicated model_text.
    docs = [{"id": did, "title": by_id[did]["title"], "abstract": by_id[did]["text"]} for did in ids]
    (data / "documents.json").write_text(json.dumps(docs, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    vectors.tofile(data / "embeddings.f32")
    context = json.loads((SOURCE / "analysis_context.json").read_text(encoding="utf-8"))
    examples = [{"id": q, "text": context["queries"][q]["text"], "relevant": list(context["qrels_test"][q])}
                for q in ["1", "1175", "133", "1012"]]
    (data / "examples.json").write_text(json.dumps(examples, ensure_ascii=False), encoding="utf-8")
    metrics = {name: json.loads((SOURCE / file).read_text()) for name, file in [
        ("pretrained", "metrics_pretrained.json"), ("finetuned", "metrics_finetuned.json")]}
    config = json.loads((SOURCE / "experiment_config.json").read_text())
    manifest = {"documents": len(ids), "dimensions": 384, "maxLength": 256, "normalized": True,
                "model": "all-MiniLM-L6-v2 · SciFact MNRL", "epochs": 3, "batchSize": 8,
                "metrics": metrics, "source": "BEIR–SciFact", "license": "CC BY-SA 4.0",
                "embeddingSha256": hashlib.sha256((data / "embeddings.f32").read_bytes()).hexdigest(),
                "baseRevision": config["resolved_revisions"]["model"]}
    manifest["variants"] = {
        "finetuned": {"model": "scifact-minilm", "embeddings": "embeddings.f32", "label": "Fine-tuned"},
        "pretrained": {"model": "pretrained-minilm", "embeddings": "embeddings-pretrained.f32", "label": "Pretrained"},
    }
    manifest["training"] = json.loads((SOURCE / "training_summary.json").read_text())
    manifest["dataAudit"] = json.loads((SOURCE / "data_audit.json").read_text())
    manifest["truncation"] = json.loads((SOURCE / "truncation_audit.json").read_text())
    shutil.copy2(SOURCE / "training_loss.png", data / "training_loss.png")
    (data / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Exported", len(ids), "documents and", vectors.nbytes, "bytes of embeddings.", flush=True)


def export_model(variant="finetuned"):
    sys.path.insert(0, str(TOOLS / "python"))
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(4)
    if variant == "pretrained":
        experiment = json.loads((SOURCE / "experiment_config.json").read_text())
        encoder = SentenceTransformer(experiment["config"]["model_id"],
            revision=experiment["resolved_revisions"]["model"], local_files_only=True, device="cpu")
        weights_source = Path(encoder[0].auto_model.config._name_or_path)
    else:
        encoder = SentenceTransformer(str(SOURCE / "model_final"), device="cpu")
    encoder.eval()

    class TokenEncoder(torch.nn.Module):
        def __init__(self, auto_model):
            super().__init__()
            self.encoder = auto_model

        def forward(self, input_ids, attention_mask, token_type_ids):
            return self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                                token_type_ids=token_type_ids, return_dict=False)[0]

    model_dir = HERE / "models" / ("pretrained-minilm" if variant == "pretrained" else "scifact-minilm")
    (model_dir / "onnx").mkdir(parents=True, exist_ok=True)
    encoder[0].auto_model.config.save_pretrained(str(model_dir))
    encoder.tokenizer.save_pretrained(str(model_dir))
    wrapper = TokenEncoder(encoder[0].auto_model)
    wrapper.eval()
    tokens = encoder.tokenizer("Does vitamin D improve bone density?", return_tensors="pt")
    model_path = model_dir / "onnx/model.onnx"
    if not model_path.exists():
        print(f"Exporting {variant} weights to ONNX (FP32)...", flush=True)
        with torch.no_grad():
            torch.onnx.export(wrapper, tuple(tokens[k] for k in ["input_ids", "attention_mask", "token_type_ids"]),
                str(model_path), input_names=["input_ids", "attention_mask", "token_type_ids"],
                output_names=["last_hidden_state"], opset_version=17, dynamo=False,
                dynamic_axes={"input_ids": {0: "batch", 1: "sequence"},
                              "attention_mask": {0: "batch", 1: "sequence"},
                              "token_type_ids": {0: "batch", 1: "sequence"},
                              "last_hidden_state": {0: "batch", 1: "sequence"}})
    onnx.checker.check_model(str(model_path))
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    # Confirm numerical parity, including truncation, mean pooling and normalization.
    texts = ["Does vitamin D improve bone density?", "The PPR MDA5 has two N-terminal CARD domains.",
             "Radioiodine treatment of non-toxic multinodular goitre reduces thyroid volume.",
             "A scientific query " * 200]
    expected = encoder.encode(texts, normalize_embeddings=True)
    features = encoder.tokenizer(texts, return_tensors="np", padding=True, truncation=True, max_length=256)
    hidden = session.run(None, dict(features))[0]
    mask = features["attention_mask"][..., None]
    actual = (hidden * mask).sum(axis=1) / mask.sum(axis=1)
    actual /= np.linalg.norm(actual, axis=1, keepdims=True)
    difference = float(np.abs(expected - actual).max())
    assert difference < 1e-4, difference
    docs = json.loads((HERE / "data/documents.json").read_text(encoding="utf-8"))
    vector_path = HERE / "data" / ("embeddings-pretrained.f32" if variant == "pretrained" else "embeddings.f32")
    if variant == "pretrained" and not vector_path.exists():
        by_id = {d["_id"]: d for d in json.loads((SOURCE / "documents.json").read_text(encoding="utf-8"))}
        corpus_texts = [by_id[d["id"]]["model_text"] for d in docs]
        print("Encoding pretrained corpus (5183 documents)...", flush=True)
        encoder.encode(corpus_texts, batch_size=32, normalize_embeddings=True, show_progress_bar=True).astype("<f4").tofile(vector_path)
    vectors = np.fromfile(vector_path, dtype="<f4").reshape(-1, 384)
    fixtures = []
    for text, vector in zip(texts[:3], expected[:3]):
        scores = vectors @ vector
        top = np.argsort(-scores, kind="stable")[:5]
        fixtures.append({"query": text, "embedding": vector.tolist(),
                         "top5": [{"id": docs[int(i)]["id"], "score": float(scores[i])} for i in top]})
    (HERE / f"data/parity-fixtures-{variant}.json").write_text(json.dumps(fixtures), encoding="utf-8")
    (HERE / f"data/export-validation-{variant}.json").write_text(json.dumps({"maxEmbeddingError": difference, "onnxChecked": True}), encoding="utf-8")
    print("ONNX validated. Maximum embedding difference:", difference, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--pretrained", action="store_true")
    args = parser.parse_args()
    if args.download:
        frontend_assets()
    if args.export:
        export_data()
        export_model()
    if args.pretrained:
        export_data()
        export_model("pretrained")
