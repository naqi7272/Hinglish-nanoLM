import numpy as np

from hnlm.data.dedup import (
    MinHasher,
    estimate_jaccard,
    exact_hash,
    find_duplicates,
    shingle_hashes,
)


def _jaccard(a: str, b: str) -> float:
    sa, sb = set(shingle_hashes(a).tolist()), set(shingle_hashes(b).tolist())
    return len(sa & sb) / len(sa | sb)


def test_minhash_estimates_jaccard():
    hasher = MinHasher(num_perm=256)
    a = "yaar aaj office mein bahut kaam tha ghar pahunchte pahunchte raat ho gayi " * 3
    b = a.replace("office", "college").replace("raat", "subah")
    est = estimate_jaccard(hasher.signature(a), hasher.signature(b))
    assert abs(est - _jaccard(a, b)) < 0.1


def test_find_duplicates_keeps_first_of_each_cluster(tiny_corpus):
    long_hindi = " ".join(tiny_corpus[10:14])
    docs = list(tiny_corpus) + [long_hindi]
    docs.append(docs[3].upper() + "!!!")  # exact dup after normalization
    docs.append(long_hindi.replace("पसंद", "पसन्द"))  # near dup (one word spelled differently)
    docs.append(docs[18] + " Really.")  # near dup (small addition)
    hasher = MinHasher()
    sigs = np.stack([hasher.signature(d) for d in docs])
    exact = np.array([exact_hash(d) for d in docs], dtype=np.uint64)
    keep = find_duplicates(sigs, exact)
    n = len(tiny_corpus) + 1
    assert keep[:n].all(), "originals must be kept"
    assert not keep[n:].any(), "all three copies must be dropped"
