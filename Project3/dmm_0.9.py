import os
import math
from collections import defaultdict
import sys
sys.path.append('/data/HW3')

from hw3_utils import parse_documents_or_queries, parse_collection_stats, output_xml

# === DMM Parameters ===
DELTA = 0.1
LAMBDA = 0.1
TOP_K_TERMS = 25
ANCHOR_WEIGHT = 0.9

# === File Paths ===
QUERY_PATH = "/data/HW3/q_stemmed.tsv"
COLL_STATS_PATH = "/data/HW3/gov2_collection_stats.csv"
FEEDBACK_DIR = "/data/HW3/feedback_docs"
OUTPUT_XML = "dmm_0.9.xml"

# === Smoothed P(w|d) Function ===
def compute_smoothed_p_w_d(tf, doc_len, V_F):
    return (tf + DELTA) / (doc_len + DELTA * V_F)

# === Build DMM Model ===
def compute_dmm_model(doc_term_counts, doc_lengths, collection_stats, total_terms):
    vocab_f = set()
    for tf_dict in doc_term_counts.values():
        vocab_f.update(tf_dict.keys())
    V_F = len(vocab_f)

    log_avg_p_w_d = {}
    for term in vocab_f:
        log_sum = 0
        for doc_id, tf_dict in doc_term_counts.items():
            tf = tf_dict.get(term, 0)
            doc_len = doc_lengths[doc_id]
            p_w_d = compute_smoothed_p_w_d(tf, doc_len, V_F)
            log_sum += math.log(p_w_d)
        log_avg_p_w_d[term] = log_sum / len(doc_term_counts)

#    log_p_w_c = {
#        term: math.log(collection_stats.get(term, 1e-10) / total_terms)
#        for term in vocab_f
#    }
    log_p_w_c = {
        term: math.log(collection_stats[term] / total_terms)
        for term in vocab_f if collection_stats.get(term, 0) > 0
    }

    dmm = {
        term: math.exp((log_avg_p_w_d[term] - LAMBDA * log_p_w_c[term]) / (1 - LAMBDA))
        for term in vocab_f
    }
    return dmm

# === Interpolation with query model (anchoring) ===
def interpolate_with_query(feedback_model, query_tf_dict, beta):
    def normalize(dist):
        total = sum(dist.values())
        return {term: val / total for term, val in dist.items()} if total else dist

    query_mle = normalize(dict(query_tf_dict))
    feedback_norm = normalize(feedback_model)

    all_terms = set(query_mle).union(feedback_norm)
    interpolated = {
        term: beta * query_mle.get(term, 0.0) + (1 - beta) * feedback_norm.get(term, 0.0)
        for term in all_terms
    }
    return interpolated

# === Clip and Normalize Top Terms ===
def clip_top_n_normalized(model, n=TOP_K_TERMS):
    top_items = sorted(model.items(), key=lambda x: -x[1])[:n]
    total = sum(w for _, w in top_items)
    return {term: w / total for term, w in top_items} if total else dict(top_items)

# === Main Pipeline ===
def main():
    print("📄 Reading input files...")
    queries, _ = parse_documents_or_queries(QUERY_PATH)
    coll_stats, total_terms = parse_collection_stats(COLL_STATS_PATH)

    expanded_queries = {}
    for qid, query_tf in queries.items():
        print(f"📝 Processing query {qid}...")
        feedback_file = os.path.join(FEEDBACK_DIR, f"{qid}.tsv")
        doc_tfs, doc_lens = parse_documents_or_queries(feedback_file)

        feedback_model = compute_dmm_model(doc_tfs, doc_lens, coll_stats, total_terms)
        anchored_model = interpolate_with_query(feedback_model, query_tf, beta=ANCHOR_WEIGHT)
        clipped_model = clip_top_n_normalized(anchored_model)
        expanded_queries[qid] = clipped_model

    print("Number of queries parsed:", len(queries))
    print("📅 Writing output XML file...")
    output_xml(expanded_queries, OUTPUT_XML)
    print(f"✅ Output generated: {OUTPUT_XML}")

if __name__ == "__main__":
    main()

