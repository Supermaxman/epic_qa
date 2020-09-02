# EPIC-QA Project  
## Overview:  
### Pipeline for EPIC-QA System  
* Step 1: Passage Expansion  
* Step 2: Indexing & Retrieval  
* Step 3: Passage Re-Ranking  
* Step 4: Reading Comprehension  
 
### Phases for EPIC-QA Project  
* Phase 1: (EASY) No labels, entirely based on pre-trained systems on other datasets  
* Phase 2: (MEDIUM) Passage-level labels from coarse document-level labels provided, maybe utilize Active Learning  
* Phase 3: (HARD) Sentence-level labels from passage-level labels, utilize Active Learning  
 
### Issues / Unsolved Problems  
* Expert-level vs consumer-level   
 
### Citations  
* List of citations / prior work read  

# Pipeline for EPIC-QA System  
## Step 1: Passage Expansion  
* Perform passage expansion by predicting queries given a passage and appending those queries to the passage.  
* (Document Expansion by Query Prediction: https://arxiv.org/abs/1904.08375)  
* doc2query/docTTTTTquery: https://github.com/castorini/docTTTTTquery  
* TODO  
   1. Phase 1: Utilize docTTTTTquery trained model on MSMARCO to expand EPIC-QA  
   2. Phase 2: Fine-tune docTTTTTquery on queries from EPIC-QA to expand EPIC-QA  

## Step 2: Indexing & Retrieval  
* Index expanded passages from Step 1 using classical index (Lucene, etc.)  
* (BM25: find citations)  
* Retrieval of top-k passages given query using classical ranking retrieval (BM25, etc.)  
* TODO  
   1. Phase 1: Index expanded passages, configure BM25 for efficient top-k retrieval  
   2. Phase 2: TODO  

## Step 3: Passage Re-Ranking  
* Re-Rank passages from Step 2 using neural passage re-ranker   
* (Passage Re-ranking with BERT: https://arxiv.org/abs/1901.04085)  
* MSMARCO or TREC-CAR trained re-ranking models: https://github.com/nyu-dl/dl4marco-bert  
* TODO  
  1. Phase 1: Utilize MSMARCO or TREC-CAR trained re-ranking models on re-ranking EPIC-QA  
  2. Phase 2: Fine-tune re-ranking model on queries from EPIC-QA for re-ranking EPIC-QA  

## Step 4: Reading Comprehension  
* Re-Rank sentences within passages and select best contiguous subset.   
* Utilize MSMARCO or TREC-CAR trained re-ranking models on re-ranking all sentence n-grams and selecting maximum.  
* TODO  
  1. Phase 1: Utilize MSMARCO or TREC-CAR trained re-ranking models on re-ranking EPIC-QA sentence n-grams  
  2. Phase 2: Fine-tune re-ranking model on queries from EPIC-QA for re-ranking EPIC-QA sentence n-grams  
 
## Phases for EPIC-QA Project  
### Phase 1: No Labels  
* No labels necessary. Use as baseline with pre-trained systems from other IR tasks and datasets.   
 
### Phase 2: Passage-level Labels  
* Multiple options for refining document-level query labels provided by 4th round of TREC-COVID  
   * Manual refinement of document-level query labels  
   * Automatic refinement of document-level query labels using passage re-ranking model  
   * Fusion: Provide re-ranked passages as provided rank to labeler, have feedback loop akin to Active Learning  

### Phase 3: Sentence-level Labels  
* Refine passage-level labels using sentence n-gram re-ranking. Similar options to Phase 2, may just become Phase 2.  
* Active Learning  
 
 
# Issues / Unsolved Problems  
* How to differentiate between expert-level and consumer-level systems.   
   * Do we need to differentiate due to different underlying document collection and different queries?  
	 
# Citations:  
* JULIE Lab & Med Uni Graz @ TREC 2019 Precision Medicine Track  
* IDST at TREC 2019 Deep Learning Track: Deep Cascade Ranking with Generation-based Document Expansion and Pre-trained Language Modeling  
* Passage Re-Ranking with BERT  
* Document Expansion by Query Prediction  
* Multi-Stage Document Ranking with BERT  
* D-NET: A Simple Framework for Improving the Generalization of Machine Reading Comprehension  
* Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer  
* docTTTTTquery  
* Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks  