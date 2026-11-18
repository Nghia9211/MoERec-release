"""The user's own star rating for every visible history item (latest review per user-item pair) -> data/history_stars.npy.
For Goodreads also data/item_texts.json, from the upstream goodreads_rich FAISS docstore (built with item_text.goodreads_text).
Usage: python -m moerec.data.stars <dataset> --raw <AgentRecBench Dataset dir> --data <work dir>/<dataset>/data [--docstore index.pkl]"""
import argparse, json, pickle, datetime, numpy as np
from pathlib import Path
ap = argparse.ArgumentParser(); ap.add_argument('dataset'); ap.add_argument('--raw', required=True); ap.add_argument('--data', required=True); ap.add_argument('--docstore')
A = ap.parse_args(); DS = A.dataset
SRC = {'industrial': ('amazon_industrial', None), 'musical': ('amazon_musical_raw', None), 'video_games': ('amazon_video_games', 'amazon'),
       'yelp': ('merged_all', 'yelp'), 'goodreads': ('merged_all', 'goodreads')}[DS]
E = Path(A.data)


def ts_of(d):
    if d.get('timestamp') not in (None, ''): return int(d['timestamp'])
    for k in ('date', 'date_added', 'date_updated'):
        v = d.get(k)
        if v:
            for fmt in ('%Y-%m-%d %H:%M:%S', '%a %b %d %H:%M:%S %z %Y'):
                try: return int(datetime.datetime.strptime(v, fmt).timestamp() * 1000)
                except ValueError: pass
    return 0


users = json.loads((E / 'users.json').read_text()); u2i = {u: k for k, u in enumerate(users)}
raw = json.loads((E / 'id2rawid.json').read_text()); r2i = {r: k for k, r in enumerate(raw) if r}
best = {}
for n, line in enumerate(open(Path(A.raw) / SRC[0] / 'review.json')):
    d = json.loads(line)
    if SRC[1] and d.get('source') != SRC[1]: continue
    u = u2i.get(str(d['user_id'])); i = r2i.get(str(d['item_id']).strip())
    if u is None or i is None: continue
    t = (ts_of(d), n)
    if (u, i) not in best or t >= best[(u, i)][0]: best[(u, i)] = (t, float(d.get('stars') or 0))
h = np.load(E / 'histories.npz'); off, items = h['offsets'], h['items']
stars = np.zeros(len(items), np.float32); miss = 0
for u in range(len(users)):
    for p in range(off[u], off[u + 1]):
        v = best.get((u, int(items[p])))
        if v is None: miss += 1
        else: stars[p] = v[1]
np.save(E / 'history_stars.npy', stars)
info = {'dataset': DS, 'history_positions': int(len(items)), 'missing': miss, 'star_counts': {str(k): int((stars == k).sum()) for k in [0, 1, 2, 3, 4, 5]}}
if DS == 'goodreads':
    with open(A.docstore, 'rb') as f: docstore, _ = pickle.load(f)
    docs = {str(d.metadata['item_id']): d.page_content for d in docstore._dict.values()}
    texts = [''] + [docs.get(r, '') for r in raw[1:]]
    info['item_texts_missing'] = sum(1 for t in texts[1:] if not t)
    (E / 'item_texts.json').write_text(json.dumps(texts, ensure_ascii=False))
print(json.dumps(info))
