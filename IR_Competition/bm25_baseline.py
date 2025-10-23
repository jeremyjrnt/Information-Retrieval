import math
from collections import defaultdict, Counter

# === Paramètres BM25 ===
k1 = 1.2
b = 0.75

# === Données globales ===
N = 0  # total documents
avgdl = 0  # average document length
doc_freqs = defaultdict(int)  # combien de documents contiennent chaque terme
documents = {}  # {doc_id: list of tokens}

# === Étape 1 : Charger les documents (à écrire)
# tu dois remplir "documents" et calculer avgdl

# === Étape 2 : Calculer IDF
def compute_idf(term):
    nt = doc_freqs[term]
    return math.log((N - nt + 0.5) / (nt + 0.5) + 1)

# === Étape 3 : BM25 score
def score(query_tokens, doc_tokens):
    score = 0.0
    doc_len = len(doc_tokens)
    freqs = Counter(doc_tokens)
    for term in query_tokens:
        if term in freqs:
            idf = compute_idf(term)
            f = freqs[term]
            denom = f + k1 * (1 - b + b * doc_len / avgdl)
            score += idf * (f * (k1 + 1)) / denom
    return score

# === Étape 4 : Exemple d’appel (à compléter)
# query = ["climate", "change"]
# for doc_id, doc_tokens in documents.items():
#     print(doc_id, score(query, doc_tokens))
import math
from collections import defaultdict, Counter

# === Paramètres BM25 ===
k1 = 1.2
b = 0.75

# === Données globales ===
N = 0  # total documents
avgdl = 0  # average document length
doc_freqs = defaultdict(int)  # combien de documents contiennent chaque terme
documents = {}  # {doc_id: list of tokens}

# === Étape 1 : Charger les documents (à écrire)
# tu dois remplir "documents" et calculer avgdl

# === Étape 2 : Calculer IDF
def compute_idf(term):
    nt = doc_freqs[term]
    return math.log((N - nt + 0.5) / (nt + 0.5) + 1)

# === Étape 3 : BM25 score
def score(query_tokens, doc_tokens):
    score = 0.0
    doc_len = len(doc_tokens)
    freqs = Counter(doc_tokens)
    for term in query_tokens:
        if term in freqs:
            idf = compute_idf(term)
            f = freqs[term]
            denom = f + k1 * (1 - b + b * doc_len / avgdl)
            score += idf * (f * (k1 + 1)) / denom
    return score

# === Étape 4 : Exemple d’appel (à compléter)
# query = ["climate", "change"]
# for doc_id, doc_tokens in documents.items():
#     print(doc_id, score(query, doc_tokens))

