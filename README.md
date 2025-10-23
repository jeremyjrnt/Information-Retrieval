# Information Retrieval Projects

This repository contains coursework for an Information Retrieval course, consisting of three main projects and an IR Competition. The projects progressively build from basic indexing and boolean retrieval to advanced query expansion techniques and competitive retrieval optimization.

---

## Table of Contents

1. [Project 1: Inverted Index and Boolean Retrieval](#project-1-inverted-index-and-boolean-retrieval)
2. [Project 2](#project-2)
3. [Project 3: Query Expansion Models](#project-3-query-expansion-models)
4. [IR Competition: Optimizing Retrieval Performance](#ir-competition-optimizing-retrieval-performance)

---

## Project 1: Inverted Index and Boolean Retrieval

### Objectives

- Build an inverted index from the AP (Associated Press) news collection
- Implement a Boolean retrieval model supporting AND, OR, and NOT operators
- Analyze term frequency distributions in the collection

### Implementation

#### Part 1: Inverted Index Construction

The `InvertedIndex` class constructs an index from the AP collection with the following features:

**Data Structures:**

- `index`: Dictionary mapping terms to posting lists (document IDs)
- `internal_to_original`: Maps internal integer doc IDs to original DOCNO strings
- `original_to_internal`: Reverse mapping for quick lookups

**Algorithm:**

1. Parse documents from XML-like format using regex patterns
2. Extract text from `<TEXT>` tags
3. Tokenize using word boundaries (`\b\w+\b`)
4. Build posting lists with sequential internal IDs for efficiency
5. Store unique terms per document (set-based approach)

**Key Methods:**

- `build_index()`: Main indexing pipeline
- `get_posting_list(term)`: Returns posting list for a given term
- `get_original_doc_id()`: Converts internal ID to original DOCNO

#### Part 2: Boolean Retrieval

The `BooleanRetrieval` class implements query processing using a stack-based approach:

**Query Processing:**

- Parses queries in postfix notation (e.g., "climate AND change OR warming")
- Uses a stack to evaluate operators

**Operators Implementation:**

- **AND**: Merge intersection using two-pointer technique (O(n+m))
- **OR**: Merge union maintaining sorted order
- **NOT**: Set difference operation (A NOT B)

**Algorithm Complexity:**

- AND/OR operations: O(n + m) where n, m are posting list lengths
- All operations maintain sorted posting lists for efficient merging

#### Part 3: Term Frequency Analysis

**Part 3a - Most Common Terms:**

```
the: 242067
of: 238932
in: 236964
a: 236963
and: 236670
to: 235720
for: 219079
said: 218301
on: 213235
that: 204381
```

**Part 3b - Rarest Terms:**

```
00000: 1
0000015: 1
0000033: 1
0000057: 1
0000066: 1
000008: 1
00001: 1
000010: 1
000012: 1
000021: 1
```

**Part 3c - Analysis:**
The most common terms are English stop words ("the", "of", "in", "a", "and") that appear in nearly every document and carry minimal semantic information. These high-frequency terms dominate the collection but provide little discriminative power for retrieval.

The rarest terms are primarily numeric strings appearing in exactly one document. These likely originate from specific numeric formats, dates, or identifiers rather than meaningful content words.

**Conclusion:** Both extremes of the frequency distribution contain terms with low semantic value. Effective retrieval systems typically filter stop words and may apply frequency-based thresholds to focus on discriminative terms with medium frequency ranges.

### Methods Used

- **Regex-based parsing**: Extracts structured information from document format
- **Set-based tokenization**: Ensures unique terms per document
- **Efficient merging algorithms**: Linear-time posting list operations
- **Dictionary-based indexing**: Fast term-to-postings lookup

---

## Project 2: Stemming, Stopwords, and Query Analysis

### Objectives

- Evaluate the impact of **stemming** and **stopword removal** on retrieval performance.  
- Analyze how **term frequency**, **morphological variation**, and **query intent** influence tf-idf rankings.  
- Compare configurations using **`trec_eval`** metrics (P@5, P@10, MAP).

---

### Implementation

#### Part A — Analytical Experiments

1. **Morphological Variation & Stemming**  
   Querying *“corporation”* retrieved only **D2**, missing **D3** (*“corporations”*).  
   With stemming enabled, both terms map to the same root and both documents are retrieved.

2. **Term Frequency Impact**  
   The token *IBM* appears only in **D2** and **D3**.  
   Since both share the same idf, the document with higher term frequency (**D2**) scores higher.

3. **Stopword Influence**  
   A query using *“in”* without stopword removal produced noisy results dominated by high-frequency function words.  
   Removing stopwords increased retrieval precision.

4. **Query Intent and Relevance**  
   Document **D4** includes *“Michael Jackson”* but mainly discusses *Lady Gaga*.  
   The system ranks *Lady Gaga* higher (1.857 > 1.366) because topical context better matches intent.

---

### Part B — Evaluation Results

| Stopword Removal | Stemming | P@5   | P@10  | MAP     |
|:-----------------|:---------|:-----:|:-----:|:-------:|
| Yes              | **Yes**  | 0.3919 | 0.3725 | **0.2113** |
| Yes              | **No**   | 0.3933 | 0.3658 | **0.1860** |

**Findings**  
- **Stemming** improved MAP (0.186 → 0.211) by merging morphological variants, boosting recall.  
- Precision at small cutoffs (P@5, P@10) changed minimally, suggesting gains appear in deeper recall.

---

### Insights

- **Stemming** mitigates morphological mismatch, improving recall.  
- **Stopword removal** filters non-informative terms, improving ranking quality.  
- **Query interpretation** should capture topical intent beyond literal phrase matches.  
- **tf-idf weighting** is sensitive to both term frequency and document length, reinforcing the need for normalization.

---

### Tools and Data

- **Model:** tf-idf retrieval  
- **Evaluation:** `trec_eval` (P@5, P@10, MAP)  
- **Corpus:** Toy AP-style collection  
- **Implementation:** Python scripts with preprocessing pipeline

---

### Conclusion

Stemming and stopword removal are essential preprocessing steps in classical Information Retrieval.  
Empirical results show that stemming enhances Mean Average Precision, while stopword filtering prevents irrelevant matches.  
These experiments lay the groundwork for the advanced feedback and expansion techniques implemented in **Project 3**.


---

## Project 3: Query Expansion Models

### Objectives

- Implement advanced query expansion techniques
- Compare Relevance Model 3 (RM3) and Divergence Minimization Model (DMM)
- Generate expanded queries in Indri XML format for improved retrieval

### Implementation

#### Relevance Model 3 (RM3)

**Theoretical Foundation:**

RM3 is a pseudo-relevance feedback model that expands queries by estimating term probabilities from top-retrieved documents. It combines:

1. **RM1**: Estimates P(w|R) where R is the relevance model
2. **Query interpolation**: Balances original query with feedback model

**Algorithm:**

1. **Compute Query Likelihood (Dirichlet Smoothing):**

   ```
   P(q|M_d) = ∏ P(w|M_d)^{tf_q(w)}

   P(w|M_d) = (tf_d(w) + μ·P(w|C)) / (|d| + μ)
   ```

   Where:

   - μ (MU) = 1000: Dirichlet smoothing parameter
   - P(w|C): Collection language model probability
   - |d|: Document length
2. **Compute RM1 Model:**

   ```
   P(w|R) = Σ_d P(w|M_d) · P(M_d|q)

   P(M_d|q) = P(q|M_d) / Σ_d' P(q|M_d')  [normalized]
   ```
3. **Interpolate with Original Query:**

   ```
   P(w|θ_RM3) = β·P(w|Q) + (1-β)·P(w|R)
   ```

   Where β = 0.5 balances original query and feedback
4. **Select Top-K Terms:**

   - Keep top 25 terms by probability
   - Renormalize to sum to 1.0

**Parameters:**

- `MU = 1000`: Controls smoothing strength
- `BETA = 0.5`: Original query weight (0.5 = equal weighting)
- `TOP_K = 25`: Number of expansion terms
- `NUM_DOCS = 10`: Feedback document count

**Code Structure:**

- `compute_query_likelihood()`: Scores documents using Dirichlet smoothing
- `compute_rm1()`: Builds relevance model from feedback docs
- `interpolate_models()`: Combines RM1 with original query
- `select_top_k()`: Clips and normalizes top terms

#### Divergence Minimization Model (DMM)

**Theoretical Foundation:**

DMM selects feedback terms by minimizing divergence between the feedback model and collection model. It emphasizes terms that are:

- Common in feedback documents
- Rare in the general collection (high discriminative power)

**Algorithm:**

1. **Smoothed Document Model:**

   ```
   P(w|d) = (tf_d(w) + δ) / (|d| + δ·|V_F|)
   ```

   Where:

   - δ (DELTA) = 0.1: Additive smoothing parameter
   - |V_F|: Vocabulary size in feedback set
2. **Average Document Model:**

   ```
   log P_avg(w|d) = (1/|F|) · Σ_{d∈F} log P(w|d)
   ```
3. **DMM Score (Divergence Minimization):**

   ```
   DMM(w) = exp[(log P_avg(w|d) - λ·log P(w|C)) / (1-λ)]
   ```

   Where:

   - λ (LAMBDA) = 0.1: Collection model influence
   - P(w|C): Collection probability
4. **Query Anchoring (Interpolation):**

   ```
   P(w|θ_DMM) = β·P(w|Q) + (1-β)·DMM(w)
   ```

**Parameters:**

- `DELTA = 0.1`: Additive smoothing for document models
- `LAMBDA = 0.1`: Collection model weight in divergence
- `ANCHOR_WEIGHT = 0.5` (standard) or `0.9` (high anchoring)
- `TOP_K_TERMS = 25`: Expansion vocabulary size

**Variants Implemented:**

- `dmm.py`: Standard anchoring (β = 0.5)
- `dmm_0.9.py`: High query anchoring (β = 0.9) - stays closer to original query

**Code Structure:**

- `compute_smoothed_p_w_d()`: Additive smoothing for term probabilities
- `compute_dmm_model()`: Builds DMM scores via divergence minimization
- `interpolate_with_query()`: Query anchoring interpolation
- `clip_top_n_normalized()`: Selects and normalizes top terms

### Comparison: RM3 vs DMM

| Aspect                         | RM3                           | DMM                               |
| ------------------------------ | ----------------------------- | --------------------------------- |
| **Philosophy**           | Probabilistic relevance       | Divergence minimization           |
| **Smoothing**            | Dirichlet (μ=1000)           | Additive (δ=0.1)                 |
| **Selection Criterion**  | P(w\|R) from query likelihood | Maximize distance from collection |
| **Collection Influence** | Through smoothing             | Explicit penalty (λ)             |
| **Typical Use**          | General-purpose PRF           | Specialized/discriminative terms  |

### Output Format

Both models generate Indri-compatible XML with normalized term weights:

```xml
<parameters>
  <query>
    <number>401</number>
    <term>foreign</term>
    <weight>0.15234</weight>
    <term>minorities</term>
    <weight>0.12456</weight>
    ...
  </query>
</parameters>
```

### Dependencies

- `hw3_utils`: Provides parsing and XML output utilities
- Collection statistics file: Pre-computed term frequencies
- Feedback documents: Top-k docs from initial retrieval

---

## IR Competition: Optimizing Retrieval Performance

### Objectives

- Achieve highest Mean Average Precision (MAP) on ROBUST04 test collection
- Experiment with retrieval models, parameters, and query processing techniques
- Compare baseline and advanced retrieval strategies

### Dataset

- **Collection**: ROBUST04 (approximately 528,000 news articles)
- **Queries**: 150 queries (301-450, 601-700)
- **Evaluation**: TREC evaluation using `qrels_50_Queries`

### Approaches Implemented

#### 1. BM25 Baseline

**Theory:**

BM25 (Best Match 25) is a probabilistic ranking function based on the Binary Independence Model with term frequency saturation and document length normalization.

**Formula:**

```
BM25(q,d) = Σ_{t∈q} IDF(t) · (f(t,d)·(k₁+1)) / (f(t,d) + k₁·(1-b+b·|d|/avgdl))

IDF(t) = log((N - n(t) + 0.5) / (n(t) + 0.5) + 1)
```

Where:

- **k₁**: Controls term frequency saturation (typical: 1.0-2.0)
- **b**: Document length normalization (0 = no norm, 1 = full norm)
- **N**: Total documents
- **n(t)**: Documents containing term t
- **f(t,d)**: Term frequency in document d
- **|d|**: Document length
- **avgdl**: Average document length

**Parameter Configurations:**

1. **Standard BM25** (`bm25_50queries.param`):

   - k₁ = 1.2, b = 0.75
   - Balanced saturation and length normalization
2. **Conservative BM25** (`bm25_k1_1.0_b_0.5.param`):

   - k₁ = 1.0, b = 0.5
   - Less aggressive saturation, reduced length penalty
   - Better for shorter queries or diverse document lengths

**Implementation:**

```python
k1 = 1.2
b = 0.75

def compute_idf(term):
    nt = doc_freqs[term]
    return math.log((N - nt + 0.5) / (nt + 0.5) + 1)

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
```

#### 2. Field-Based Boosting

**Theory:**

Field-based retrieval assigns different weights to document fields (title, body, abstract) based on their importance. Title words are typically more indicative of document relevance.

**Indri Weight Operator:**

```
#weight( w₁ #combine(body_terms) w₂ #combine(title.terms) )
```

**Algorithm:**

1. Define boost weight for title field (e.g., 0.1-0.5)
2. Complement weight for body: w_body = 1 - w_title
3. Use Indri's field indexing (title. prefix accesses title field)

**Implementation (`run_title_boost_loop.py`):**

Automated parameter sweep:

```python
boost_values = [0.1, 0.2, 0.3, 0.4, 0.5]

for boost in boost_values:
    body_weight = 1.0 - boost
  
    # Template substitution
    query = f"#weight( {body_weight} #combine(...) {boost} #combine(title...) )"
  
    # Run IndriRunQuery
    # Evaluate with trec_eval
```

**Evaluation Process:**

1. Generate parameter files with different boost values
2. Run `IndriRunQuery` for each configuration
3. Compute MAP using `trec_eval -c -m map qrels results.res`
4. Compare performance across boost values

**Typical Results:**

- Low boost (0.1-0.2): Marginal improvement over body-only
- Medium boost (0.3-0.4): Often optimal balance
- High boost (>0.5): May overemphasize title, hurting recall

#### 3. Pseudo-Relevance Feedback (RM3)

**Configuration** (`rm3_test.param`):

```xml
<rule>method:okapi,k1:1.0,b:0.5</rule>
<fbDocs>5</fbDocs>
<fbTerms>30</fbTerms>
<fbOrigWeight>0.8</fbOrigWeight>
```

**Parameters:**

- **fbDocs**: Number of feedback documents (5-10 typical)
- **fbTerms**: Expansion vocabulary size (20-50 typical)
- **fbOrigWeight**: Original query weight (0.5-0.9)
  - 0.8 = conservative expansion (80% original query)
  - Lower values = more aggressive expansion

**Strategy:**
Used for 100 queries (351-450, 601-700) with pseudo-relevance feedback to improve recall by adding related terms from top-ranked documents.

### Experimental Methodology

1. **Baseline Establishment:**

   - BM25 with standard parameters
   - Provides reference MAP score
2. **Parameter Tuning:**

   - Systematic variation of k₁, b values
   - Grid search over reasonable ranges
3. **Field Weighting:**

   - Title boost optimization
   - Automated evaluation loop
4. **Query Expansion:**

   - RM3 for difficult queries
   - Adjust expansion aggressiveness
5. **Evaluation Metrics:**

   - **MAP (Mean Average Precision)**: Primary metric
   - **Precision@10**: Top-result quality
   - **NDCG**: Graded relevance (if applicable)

### Tools and Infrastructure

**Indri Toolkit:**

- `IndriRunQuery`: Batch query execution
- `IndriIndex`: Index building and management
- Query language: Supports operators (#combine, #weight, #syn)

**TREC Evaluation:**

- `trec_eval`: Standard IR evaluation toolkit
- Qrels format: Query-document relevance judgments
- `-c` flag: Averages over all queries including zeros

**Query Format:**

- XML-based parameter files
- Indri query operators
- Support for structured queries

### Results Analysis

**Files Generated:**

- `run_2.res`: Result file in TREC format (query_id Q0 doc_id rank score run_name)
- `boost_results/`: Directory with field-boosting experiments
- Various `.param` files: Different retrieval configurations

**Competition Strategy:**

1. Establish BM25 baseline with standard parameters
2. Optimize BM25 parameters (k₁, b) for dataset characteristics
3. Apply field boosting to leverage title information
4. Use query expansion (RM3) for challenging queries
5. Ensemble or select best performing configuration per query

### Best Practices

**Parameter Selection:**

- Higher k₁ (1.5-2.0): Verbose documents, high term repetition
- Lower k₁ (0.8-1.2): Short documents, unique vocabulary
- Higher b (0.75-0.9): Variable document lengths
- Lower b (0.3-0.6): Uniform document lengths

**Query Processing:**

- Short queries: Benefit from expansion (RM3/DMM)
- Long queries: May not need expansion
- Ambiguous queries: Field boosting helps
- Specific queries: Conservative parameters

**Computational Considerations:**

- BM25: Fast, suitable for large collections
- Query expansion: 2-3x slower (multiple retrievals)
- Field indexing: Slight overhead, major performance gains

---

## Technical Requirements

### Dependencies

- Python 3.x
- Standard libraries: `collections`, `os`, `re`, `math`, `subprocess`
- Indri toolkit (for IR Competition)
- TREC evaluation tools

### Data Structure

```
Information-Retrieval/
├── Project1/
│   ├── inverted_index.py       # Main implementation
│   ├── Part_2.txt              # Boolean query results
│   ├── Part_3a.txt             # Most common terms
│   ├── Part_3b.txt             # Rarest terms
│   └── Part_3c.txt             # Term analysis
├── Project2/
│   └── project2.pdf            # Documentation
├── Project3/
│   ├── rm3.py                  # Relevance Model 3
│   ├── rm3.xml                 # RM3 output
│   ├── dmm.py                  # DMM (β=0.5)
│   ├── dmm.xml                 # DMM output
│   ├── dmm_0.9.py              # DMM (β=0.9)
│   └── dmm_0.9.xml             # DMM 0.9 output
├── IR_Competition/
│   ├── bm25_baseline.py        # BM25 implementation
│   ├── bm25_50queries.param    # BM25 standard config
│   ├── bm25_k1_1.0_b_0.5.param # BM25 conservative config
│   ├── rm3_test.param          # RM3 configuration
│   ├── boost_template.xml      # Field boosting template
│   ├── run_title_boost_loop.py # Automated boosting experiments
│   ├── queriesROBUST.xml       # Query collection
│   ├── qrels_50_Queries        # Relevance judgments
│   └── run_2.res               # Result file
└── README.md                   # This file
```

---

## Key Concepts Summary

### Information Retrieval Models

1. **Boolean Model** (Project 1)

   - Exact match retrieval
   - Set-based operations
   - No ranking
2. **Vector Space Model** (Background for BM25)

   - Documents and queries as vectors
   - Cosine similarity
   - TF-IDF weighting
3. **Probabilistic Models** (BM25, RM3)

   - Probability ranking principle
   - Relevance estimation
   - Statistical foundations
4. **Language Models** (RM3, DMM)

   - Documents as language samples
   - Query generation probability
   - Smoothing techniques

### Evaluation Metrics

- **Precision**: Relevant docs / Retrieved docs
- **Recall**: Relevant docs retrieved / Total relevant docs
- **Average Precision (AP)**: Area under precision-recall curve
- **MAP**: Mean AP across queries
- **P@k**: Precision at rank k (e.g., P@10)

### Advanced Techniques

- **Pseudo-Relevance Feedback**: Automatic query expansion
- **Query Likelihood**: P(q|d) instead of P(d|q)
- **Field-based Retrieval**: Structured document representations
- **Smoothing**: Handling zero probabilities
  - Dirichlet smoothing
  - Jelinek-Mercer smoothing
  - Additive smoothing

---

## References and Theory

### BM25

- Robertson, S., & Zaragoza, H. (2009). "The Probabilistic Relevance Framework: BM25 and Beyond"
- Based on 2-Poisson model and term frequency saturation

### Relevance Models

- Lavrenko, V., & Croft, W. B. (2001). "Relevance-based Language Models"
- Estimates P(w|R) from pseudo-relevant documents

### Divergence Minimization

- Zhai, C., & Lafferty, J. (2001). "Model-based Feedback in the Language Modeling Approach"
- Minimizes KL-divergence between feedback and collection models

### Indri Query Language

- Strohman, T., et al. (2005). "Indri: A Language Model-based Search Engine"
- Combines inference networks with language modeling

---

## Future Improvements

- **Learning to Rank**: Machine learning-based ranking
- **Neural IR**: Dense retrieval with transformers (BERT, ColBERT)
- **Query Understanding**: Entity recognition, intent classification
- **Diversification**: Result set diversity for ambiguous queries
- **Efficiency**: Approximate nearest neighbor search, caching strategies

---

## License

Academic coursework - All rights reserved
