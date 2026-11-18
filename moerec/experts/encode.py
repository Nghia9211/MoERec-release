"""MPNet document vectors of the semantic expert: sentence-transformers all-mpnet-base-v2 (revision e8c3b32), float32, max sequence
length 384, L2-normalised. Row 0 (padding) is zero. Resumable in chunks of 4,096 items.
Usage: python -m moerec.experts.encode <dataset dir>   (env MPNET_PATH = local model dir, optional GPU_FRACTION)"""
import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault('HF_HUB_OFFLINE', '1'); os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import numpy as np
import torch

from moerec.runtime import save_json, sha256


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('exp'); E = Path(ap.parse_args().exp)
    if torch.cuda.is_available(): torch.cuda.set_per_process_memory_fraction(float(os.environ.get('GPU_FRACTION', '.12')), 0)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(os.environ['MPNET_PATH'], device='cuda' if torch.cuda.is_available() else 'cpu'); model.eval()
    assert next(model.parameters()).dtype == torch.float32 and model.max_seq_length == 384

    def embed(texts):
        x = model.encode([t.replace('\n', ' ') for t in texts], batch_size=64, show_progress_bar=False, normalize_embeddings=False, convert_to_numpy=True).astype(np.float32)
        assert np.isfinite(x).all() and np.all(np.linalg.norm(x, axis=1) > 0)
        return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)

    texts = json.loads((E / 'data/item_texts.json').read_text()); count = len(texts) - 1; out = E / 'document_vectors.npy'
    v = np.lib.format.open_memmap(out, mode='r+' if out.exists() else 'w+', dtype=np.float32, shape=(count + 1, 768))
    st = E / 'encode-docs-state.json'; begin = json.loads(st.read_text())['completed_rows'] if st.exists() else 0
    if begin == 0: v[:1] = 0
    t0 = time.monotonic()
    for i in range(begin, count, 4096):
        e = min(i + 4096, count); v[1 + i:1 + e] = embed(texts[1 + i:1 + e]); v.flush()
        save_json(st, {'completed_rows': e, 'total_rows': count, 'seconds': time.monotonic() - t0}); print('ENCODE', e, count, round(time.monotonic() - t0, 1), flush=True)
    probe = embed(texts[1:4]); assert np.allclose(probe, v[1:1 + len(probe)], atol=1e-5)
    save_json(E / 'encode-docs-verification.json', {'rows': count, 'sha256': sha256(out), 'model_path': os.environ['MPNET_PATH'], 'status': 'PASS'})


if __name__ == '__main__':
    main()
