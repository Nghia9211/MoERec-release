"""Item documents of the semantic expert, copied from the upstream MoERec code (baseline/semantic/faiss_{amazon,yelp,goodreads}.py,
github.com/Nghia9211/MoERec, commit 008f5b3). Only the function names are changed (amazon_text, yelp_text, goodreads_text)."""
import json
import os
import re
from typing import Dict

def amazon_text(item: dict) -> str:
    """
    Tạo document text semantic-rich cho 1 sản phẩm Amazon.
    """
    parts = []

    # ── Title (boost 3×) ──────────────────────────────────────────────────
    title = (item.get('title') or '').strip()
    if title:
        parts.extend([title, title, title])

    # ── Brand / Store ─────────────────────────────────────────────────────
    store = (item.get('store') or '').strip()
    if store:
        parts.append(f"Brand: {store}")

    # ── Categories (Lấy 3 category sâu nhất) ──────────────────────────────
    categories = item.get('categories')
    if categories and isinstance(categories, list):
        specific_cats = categories[-3:] if len(categories) >= 3 else categories
        parts.append("Category: " + ", ".join(specific_cats))

    # ── Features (Bullet points) ──────────────────────────────────────────
    features = item.get('features')
    if features and isinstance(features, list):
        # Lọc bỏ các phần tử None trong list (nếu có) trước khi join
        valid_features = [str(f) for f in features if f]
        feats_str = " ".join(valid_features)
        words = feats_str.split()
        if words:
            snippet = ' '.join(words[:60])
            parts.append(snippet)

    # ── Description ───────────────────────────────────────────────────────
    desc = item.get('description')
    if desc and isinstance(desc, list):
        valid_desc = [str(d) for d in desc if d]
        desc_str = " ".join(valid_desc).strip()
        words = desc_str.split()
        if words:
            snippet = ' '.join(words[:100])
            parts.append(snippet)

    # ── Rating ───────────────────────────────────────────────────────────
    rating = item.get('average_rating')
    if rating:
        try:
            r = float(rating)
            if 0 < r <= 5:
                parts.append(f"Rated {r:.1f} stars")
        except (ValueError, TypeError):
            pass

    # ── Fallback ──────────────────────────────────────────────────────────
    if not parts:
        return (item.get('title') or '') or str(item.get('item_id', 'unknown'))

    return " | ".join(filter(None, parts))


def parse_camel_case(text: str) -> str:
    return re.sub(r"(\w)([A-Z])", r"\1 \2", text)


def yelp_text(item: dict) -> str:
    parts = []

    # ── Name (boost 3×) ──────────────────────────────────────────────────
    name = (item.get('name') or '').strip()
    if name:
        parts.extend([name, name, name])

    # ── Categories ───────────────────────────────────────────────────────
    categories = (item.get('categories') or '').strip()
    if categories:
        parts.append(f"Categories: {categories}")

    # ── Location ─────────────────────────────────────────────────────────
    city = (item.get('city') or '').strip()
    state = (item.get('state') or '').strip()
    if city and state:
        parts.append(f"Location: {city}, {state}")

    # ── Attributes ───────────────────────
    attributes = item.get('attributes')
    if attributes and isinstance(attributes, dict):
        active_attrs = []
        for k, v in attributes.items():
            if str(v).lower() == 'true' or str(v).lower() == "u'free'":
                active_attrs.append(parse_camel_case(k))
        if active_attrs:
            parts.append("Features: " + ", ".join(active_attrs[:10]))

    # ── Rating ───────────────────────────────────────────────────────────
    stars = item.get('stars')
    if stars:
        try:
            r = float(stars)
            if 0 < r <= 5:
                parts.append(f"Rated {r:.1f} stars")
        except (ValueError, TypeError):
            pass

    # ── Fallback ──────────────────────────────────────────────────────────
    if not parts:
        return (item.get('name') or '') or str(item.get('item_id', 'unknown'))

    return " | ".join(filter(None, parts))


_SKIP_SHELVES = {
    'to-read', 'currently-reading', 'owned', 'favorites', 'books',
    'read', 'default', 'kindle', 'library', 'my-library', 'owned-books',
    'ebook', 'e-book', 'audio', 'audiobook', 'wish-list', 'book-club',
    'reviewed', 'ocpl-hold', 'hp', 'summer-15', 'gr-authors',
    'read-goodreads-authors', 'goodreads-author', 'thebookfairy',
    'book-fairy', 'real-men-read-books', 'read-reading-libraries',
}


def goodreads_text(item: dict, author_map: Dict[str, str] = None) -> str:
    parts = []

    # ── Title (boost 3×) ──────────────────────────────────────────────────
    title = (
        item.get('title_without_series', '').strip()
        or item.get('title', '').strip()
    )
    if title:
        # Lặp 3 lần để title có weight cao trong embedding
        parts.extend([title, title, title])

    # ── Authors ────────────────────
    if author_map:
        authors_raw = item.get('authors', [])
        author_names = []
        for a in authors_raw[:2]:   # max 2 authors
            if isinstance(a, dict):
                aid  = str(a.get('author_id', ''))
                role = a.get('role', '').strip()
                name = author_map.get(aid, '')
                if name and role.lower() not in {'illustrator', 'editor',
                                                 'translator', 'photographer'}:
                    author_names.append(name)
        if author_names:
            parts.append("by " + " and ".join(author_names))

    # ── Popular shelves → genres ──────────────────────────────────────────
    shelves_raw = item.get('popular_shelves', [])
    genre_names = []
    for shelf in shelves_raw:
        if isinstance(shelf, dict):
            name  = shelf.get('name', '').strip().lower()
            count = int(shelf.get('count', 0))
            if name and count >= 2 and name not in _SKIP_SHELVES:
                genre_names.append(name.replace('-', ' '))
        elif isinstance(shelf, str):
            name = shelf.strip().lower()
            if name not in _SKIP_SHELVES:
                genre_names.append(name.replace('-', ' '))

    if genre_names:
        parts.append("genres: " + ", ".join(genre_names[:8]))
    desc = item.get('description', '').strip()
    if desc:
        words = desc.split()
        if words:
            snippet = ' '.join(words[:120])
            parts.append(snippet)

    # ── Format ───────────────────────────────────────────────────────────
    fmt = item.get('format', '').strip()
    is_ebook = str(item.get('is_ebook', '')).lower()
    if is_ebook == 'true':
        parts.append("ebook")
    elif fmt and fmt.lower() not in {'', 'unknown'}:
        parts.append(fmt.lower())   # "hardcover", "paperback", etc.

    # ── Publication year ──────────────────────────────────────────────────
    pub_year = item.get('publication_year', '').strip()
    if pub_year and pub_year.isdigit() and 1800 <= int(pub_year) <= 2030:
        parts.append(f"published {pub_year}")

    # ── Rating ───────────────────────────────────────────────────────────
    rating = item.get('average_rating', '').strip()
    if rating:
        try:
            r = float(rating)
            if 0 < r <= 5:
                parts.append(f"rated {r:.1f} stars")
        except ValueError:
            pass

    # ── Fallback ──────────────────────────────────────────────────────────
    if not parts:
        return item.get('title', '') or str(item.get('item_id', 'unknown'))

    return " | ".join(filter(None, parts))


# ─────────────────────────────────────────────────────────────────────────────
# Load author map (optional)
# ─────────────────────────────────────────────────────────────────────────────


def load_author_map(author_file: str) -> Dict[str, str]:
    if not author_file or not os.path.exists(author_file):
        return {}

    author_map = {}
    with open(author_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                a = json.loads(line)
                aid  = str(a.get('author_id') or a.get('id') or '')
                name = a.get('name') or a.get('author_name') or ''
                if aid and name:
                    author_map[aid] = name.strip()
            except json.JSONDecodeError:
                pass

    print(f"[Build] Loaded {len(author_map)} authors")
    return author_map


# ─────────────────────────────────────────────────────────────────────────────
# Dry run
# ─────────────────────────────────────────────────────────────────────────────
