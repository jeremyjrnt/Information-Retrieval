import os
import math
from collections import defaultdict
import sys
sys.path.append('/data/HW3')

from hw3_utils import parse_documents_or_queries, parse_collection_stats, output_xml

# === Paramètres ===
MU = 1000
BETA = 0.5
TOP_K = 25
NUM_DOCS = 10

# === Fichiers ===
QUERY_FILE = '/data/HW3/q_stemmed.tsv'
COLL_STATS_FILE = '/data/HW3/gov2_collection_stats.csv'
FEEDBACK_DIR = '/data/HW3/feedback_docs/'
OUTPUT_FILE = 'rm3.xml'

# === Chargement des données ===
queries, _ = parse_documents_or_queries(QUERY_FILE)
collection_stats, total_terms = parse_collection_stats(COLL_STATS_FILE)

# === Fonction P(q|Md) avec Dirichlet smoothing ===
def compute_query_likelihood(query, doc, doc_len, mu, collection_stats, total_terms):
    score = 0.0
    for term, tf_q in query.items():
        tf_d = doc.get(term, 0)
        p_wc = collection_stats.get(term, 0) / total_terms
        smoothed = (tf_d + mu * p_wc) / (doc_len + mu)
        score += tf_q * math.log(smoothed)
    return math.exp(score)

# === Calcul de P(w|R) ===
def compute_rm1(query, feedback_docs, collection_stats, total_terms):
    p_md_q = {}
    doc_lens = {}
    for docid, doc in feedback_docs.items():
        doc_len = sum(doc.values())
        doc_lens[docid] = doc_len
        p_md_q[docid] = compute_query_likelihood(query, doc, doc_len, MU, collection_stats, total_terms)
    
    # Normalisation
    total_p = sum(p_md_q.values())
    for d in p_md_q:
        p_md_q[d] /= total_p

    # RM1: P(w|R) = sum_d P(w|Md) * P(Md|q), avec P(w|Md) = MLE
    rm1 = defaultdict(float)
    for docid, doc in feedback_docs.items():
        doc_len = doc_lens[docid]
        for term, tf in doc.items():
            pw_md = tf / doc_len  # MLE
            rm1[term] += pw_md * p_md_q[docid]

    return rm1

# === Interpolation RM1 + requête originale ===
def interpolate_models(rm1, query, beta):
    all_terms = set(rm1.keys()).union(query.keys())
    total_query = sum(query.values())
    interpolated = {}
    for term in all_terms:
        p_q = query.get(term, 0) / total_query
        p_r = rm1.get(term, 0)
        interpolated[term] = beta * p_q + (1 - beta) * p_r
    return interpolated

# === Clipping Top-K ===
def select_top_k(model, k):
    top = sorted(model.items(), key=lambda x: -x[1])[:k]
    total = sum(w for _, w in top)
    return {t: w / total for t, w in top if total > 0}

# === Pipeline RM3 pour chaque requête ===
expanded_queries = {}
for qid in sorted(queries):
    query = queries[qid]
    feedback_path = os.path.join(FEEDBACK_DIR, f"{qid}.tsv")
    feedback_docs, _ = parse_documents_or_queries(feedback_path)
    
    rm1 = compute_rm1(query, feedback_docs, collection_stats, total_terms)
    interpolated = interpolate_models(rm1, query, BETA)
    final_model = select_top_k(interpolated, TOP_K)
    
    expanded_queries[qid] = final_model

# === Génération du fichier XML Indri ===
output_xml(expanded_queries, OUTPUT_FILE)
print(f"Generated File : {OUTPUT_FILE}")

