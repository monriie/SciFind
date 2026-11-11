"""Validate the exact data/model pairing and recorded Python retrieval parity."""
import json
from pathlib import Path
import numpy as np

root=Path(__file__).resolve().parent
engine=(root/"search-engine.js").read_text(encoding="utf-8")
assert "env.allowLocalModels = true;" in engine
assert "env.allowRemoteModels = false;" in engine
source=root.parent/"outputs/scifact_laptop"
def read(path):return json.loads(path.read_text(encoding="utf-8"))
manifest=read(root/"data/manifest.json")
docs=read(root/"data/documents.json")
assert len(docs)==manifest["documents"]==5183
ids=[d["id"] for d in docs]
assert ids==read(source/"doc_ids.json")
assert len(set(ids))==len(ids)
for variant,config in manifest["variants"].items():
    vectors=np.fromfile(root/"data"/config["embeddings"],dtype="<f4").reshape(-1,384)
    assert vectors.shape==(len(docs),384)
    assert np.isfinite(vectors).all()
    assert np.allclose(np.linalg.norm(vectors,axis=1),1,atol=1e-4)
    model=root/"models"/config["model"]
    assert (model/"onnx/model.onnx").stat().st_size>80_000_000
    for filename in ["config.json","tokenizer.json","tokenizer_config.json"]:assert (model/filename).exists()
    validation=read(root/f"data/export-validation-{variant}.json")
    assert validation["maxEmbeddingError"]<1e-4 and validation["onnxChecked"]
    for fixture in read(root/f"data/parity-fixtures-{variant}.json"):
        scores=vectors@np.array(fixture["embedding"],dtype=np.float32)
        order=np.argsort(-scores,kind="stable")[:5]
        assert [ids[int(i)] for i in order]==[r["id"] for r in fixture["top5"]]
        assert np.allclose(scores[order],[r["score"] for r in fixture["top5"]],atol=1e-6)
        qid={"The PPR MDA5 has two N-terminal CARD domains.":"1175",
             "Radioiodine treatment of non-toxic multinodular goitre reduces thyroid volume.":"1012"}.get(fixture["query"])
        if qid:
            notebook_run=read(source/("run_pretrained_top100.json" if variant=="pretrained" else "run_finetuned_top100.json"))
            assert [ids[int(i)] for i in order]==[row[0] for row in notebook_run[qid][:5]]
            assert np.allclose(scores[order],[row[1] for row in notebook_run[qid][:5]],atol=1e-4)
    print(f"PASS {variant}: model, 5183 normalized embeddings, document order, ONNX parity and fixture rankings.")
final=np.fromfile(root/"data/embeddings.f32",dtype="<f4").reshape(-1,384)
assert np.array_equal(final,np.load(source/"document_embeddings.npy"))
before=np.fromfile(root/"data/embeddings-pretrained.f32",dtype="<f4").reshape(-1,384)
assert not np.allclose(before,final)
for filename in ["vendor/react.mjs","vendor/react-dom.mjs","vendor/htm.mjs","vendor/transformers.js",
                 "vendor/ort/ort.webgpu.bundle.min.mjs","vendor/ort/ort-wasm-simd-threaded.jsep.mjs",
                 "vendor/ort/ort-wasm-simd-threaded.jsep.wasm","styles.css"]:
    assert (root/filename).stat().st_size>0
js=(root/"vendor/transformers.js").read_text(encoding="utf-8")
assert 'from "onnxruntime-common"' not in js and 'from "onnxruntime-web"' not in js
assert '"/react@' not in (root/"vendor/react-dom.mjs").read_text(encoding="utf-8")
print("PASS final embeddings unchanged; pretrained and fine-tuned arrays differ; browser assets self-hosted.")
