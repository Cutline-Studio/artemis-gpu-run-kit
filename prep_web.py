"""Quality filter + PII/secret strip + exact dedup for the Common Crawl JSONL export.
Line rules (C4-style): keep lines with >=5 words ending in terminal punctuation; drop boilerplate lines
(javascript/cookie/lorem ipsum/privacy/terms/rights reserved/sign up/log in/subscribe/curly braces);
drop lines with emails, phone-like digit runs (>=9 digits), secret-shaped tokens (long hex/base64, key
prefixes, JWTs). Doc rules: >=3 kept lines, >=400 chars, letter+space ratio >=0.75, mean word length
3-10. Dedup: exact SHA1 on normalized text and on its first 300 normalized chars.
Output: kept docs separated by a blank line; stats in <out>.stats.json."""
import json, re, sys, hashlib, multiprocessing as mp
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE = re.compile(r"(\+?\d[\d\s().-]{7,}\d)")
SECRET = re.compile(r"\b([A-Fa-f0-9]{32,}|[A-Za-z0-9+/_-]{40,}={0,2}|(sk|pk|rk)_(live|test)_\w+|AKIA[0-9A-Z]{16}|ghp_\w+|xox[bpa]-\w+|nvapi-\w+|eyJ[\w-]{10,}\.[\w-]{10,})")
BAD = re.compile(r"javascript|cookie|lorem ipsum|privacy policy|terms of use|all rights reserved|\{|\}|sign up|log in|subscribe|quisque|pellentesque|mauris|adipiscing|block is invalid", re.I)
END = ('.', '!', '?', '"', "'", '\u201d', '\u2019')
def clean(line_text):
    d = json.loads(line_text); kept = []; pii = 0
    for ln in d["text"].split("\n"):
        s = ln.strip()
        if len(s.split()) < 5 or not s.endswith(END) or BAD.search(s):
            continue
        if EMAIL.search(s) or SECRET.search(s) or any(sum(c.isdigit() for c in m) >= 9 for m in PHONE.findall(s)):
            pii += 1; continue
        kept.append(s)
    doc = "\n".join(kept)
    if len(kept) < 3 or len(doc) < 400:
        return None, pii
    letters = sum(c.isalpha() or c.isspace() for c in doc) / len(doc)
    words = doc.split(); mwl = sum(map(len, words)) / len(words)
    if letters < 0.75 or not 3 <= mwl <= 10:
        return None, pii
    return doc, pii
if __name__ == "__main__":
    SRC, OUT = sys.argv[1], sys.argv[2]
    seen, seen_head = set(), set(); st = {"docs_in": 0, "kept": 0, "dup": 0, "filtered": 0, "pii_lines_stripped": 0, "bytes_out": 0}
    with open(SRC) as f, open(OUT, "w") as o, mp.Pool(5) as p:
        for doc, pii in p.imap(clean, f, chunksize=256):
            st["docs_in"] += 1; st["pii_lines_stripped"] += pii
            if doc is None: st["filtered"] += 1; continue
            norm = re.sub(r"\W+", " ", doc.lower()).strip()
            h, hh = hashlib.sha1(norm.encode()).digest(), hashlib.sha1(norm[:300].encode()).digest()
            if h in seen or hh in seen_head: st["dup"] += 1; continue
            seen.add(h); seen_head.add(hh); st["kept"] += 1; st["bytes_out"] += len(doc.encode()) + 2
            o.write(doc + "\n\n")
    json.dump(st, open(OUT + ".stats.json", "w"), indent=1); print(st)
