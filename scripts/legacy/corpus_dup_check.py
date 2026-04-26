"""Check if the Moss benchmark corpus contains duplicate/near-duplicate texts.
If yes, recall@20 by doc_id is a broken metric for this corpus.
"""
from collections import Counter
from openrag.bench import DEFAULT_DOCS_URL, fetch_docs


def main():
    docs = fetch_docs(DEFAULT_DOCS_URL)
    texts = [d["text"] for d in docs]
    print(f"Total docs:         {len(texts):>7,}")
    print(f"Unique texts:       {len(set(texts)):>7,}")

    # first 80 chars as template key (catches near-dupes with same prefix)
    prefixes = Counter(t[:80] for t in texts)
    multi = {k: v for k, v in prefixes.items() if v > 1}
    print(f"Unique 80-char prefixes: {len(prefixes):>7,}")
    print(f"Prefixes appearing >1×:  {len(multi):>7,}")
    print(f"Docs in duplicated-prefix clusters: {sum(multi.values()):>7,}  ({sum(multi.values())/len(texts)*100:.1f}%)")

    if multi:
        print("\nTop 5 duplicated prefixes (count → prefix):")
        for prefix, count in sorted(multi.items(), key=lambda x: -x[1])[:5]:
            print(f"  {count:>5}×  {prefix!r}")


if __name__ == "__main__":
    main()
