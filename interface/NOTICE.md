# Sources and acknowledgements

SciFind is a research demonstration by Kelompok 6: Charlistio Aditirta Wijaya, Andreas Calvin, Monica Amrina Rosyada, and Fitran Husein.

- Documents: BEIR–SciFact, https://huggingface.co/datasets/BeIR/scifact and https://huggingface.co/datasets/BeIR/scifact-qrels — CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/). The static JSON preserves the source document IDs, titles and abstract text; fields were reorganized and duplicate model input text omitted. Claim relevance labels remain the source qrels. Preserve this attribution and dataset license when redistributing these data.
- Base encoder: sentence-transformers/all-MiniLM-L6-v2, https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 — Apache 2.0. The pretrained export is the recorded base revision. The fine-tuned export is the group's checkpoint after three epochs of SciFact MNRL training. Both were converted to ONNX FP32; no training occurs in the application.
- React / React DOM 18.3.1 — MIT.
- HTM 3.1.1 — Apache 2.0.
- Tailwind CSS 3.4.17 — MIT. Styles are precompiled using the official standalone compiler.
- Transformers.js 3.7.2 — Apache 2.0.
- ONNX Runtime Web — MIT, version supplied by the pinned Transformers.js release.

Library license texts are included in `vendor/licenses/`. Dependencies are served from this site; no remote inference API is used. Results are document retrieval suggestions, not factual verification or medical advice.
