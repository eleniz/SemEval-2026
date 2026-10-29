# Response Clarity Classification

This repository contains a progressive study of classifying whether an interview answer clearly addresses a question. All notebooks use the **QEvasion** dataset from Hugging Face (`ailsntua/QEvasion`), which contains 3,448 training examples and 308 test examples. The target is a three-class label:

- **Clear Reply**
- **Ambivalent**
- **Clear Non-Reply**

The training data is imbalanced, with **Ambivalent** as the majority class (approximately 59%), followed by **Clear Reply** (31%) and **Clear Non-Reply** (10%). The experiments therefore report accuracy together with macro- and weighted-averaged precision, recall, and F1.

## Notebook overview

### Project 1 — `tf_idf_word2vec.ipynb`

Establishes classical machine-learning baselines using the question and answer text:

- Performs dataset inspection, missing-value checks, label-distribution analysis, and text-length analysis.
- Builds normalized question-answer inputs.
- Compares sparse **TF-IDF** features with pretrained **Word2Vec** sentence representations.
- Trains multinomial **Logistic Regression** models with balanced class weights.
- Searches regularization strength, penalty, and solver settings using a stratified validation split.
- Adds 5-fold cross-validation, classification reports, confusion matrices, per-class F1, ROC/AUC analysis, and a final Kaggle submission.

The selected TF-IDF model outperforms Word2Vec on the reported validation split:

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| TF-IDF + Logistic Regression | 0.6525 | 0.6179 |
| Word2Vec + Logistic Regression | 0.5483 | 0.5013 |

This notebook provides the interpretability-oriented baseline against which the neural approaches are compared.

### Project 2 — `classification_transformers/`

Contains three supervised transformer classifiers:

- `bert_base_uncased.ipynb` — fine-tunes `bert-base-uncased`
- `distilbert_base_uncased.ipynb` — fine-tunes `distilbert-base-uncased`
- `microsoft_deberta_v3_base.ipynb` — fine-tunes `microsoft/deberta-v3-base`

The notebooks share a controlled experimental design:

- fixed random seed (`42`) and GPU/CPU detection;
- stratified 85/15 train-validation split;
- paired question-answer tokenization with a maximum length of 512;
- head-and-tail extraction for long answers;
- class-weighted cross-entropy to address label imbalance;
- AdamW optimization with a linear warm-up/decay schedule;
- checkpoint selection by validation F1;
- learning curves, confusion matrices, error inspection, and test-set submission generation.

Reported validation results:

| Model | Accuracy | Weighted F1 | Macro F1 |
|---|---:|---:|---:|
| BERT-base | 0.6718 | 0.6749 | 0.6538 |
| DistilBERT | 0.6448 | 0.6300 | 0.5944 |
| DeBERTa-v3-base | 0.6409 | 0.6199 | 0.5854 |

BERT achieves the strongest displayed validation performance. The notebooks also show that long answers frequently exceed the 512-token limit, so truncation or head-tail compression can remove potentially useful context.

### Project 3 — `prompting_classifier.ipynb`

Evaluates generative classification without supervised fine-tuning. The notebook:

- removes 24 conflicting duplicate question-answer pairs;
- creates balanced few-shot demonstrations;
- tests **zero-shot**, **few-shot**, and **chain-of-thought-style** prompts;
- evaluates Qwen 3.5 models at approximately 0.8B, 2B, and 4B scale;
- parses generated text into one of the three allowed labels;
- analyzes errors by question/answer length and produces a test submission.

The best reported prompting configuration by weighted F1 is **Qwen3.5-2B with chain-of-thought prompting**:

| Model | Strategy | Accuracy | Weighted F1 | Macro F1 |
|---|---|---:|---:|---:|
| Qwen3.5-2B | Chain-of-thought | 0.612 | 0.5208 | 0.3342 |
| Qwen3.5-4B | Chain-of-thought | 0.476 | 0.5044 | 0.4389 |

Prompting performance is substantially more variable than supervised classification. In particular, accuracy and macro F1 do not always rank models in the same order, reflecting the difficulty of the minority classes.

### Project 4 — `d3_agentic.ipynb`

Implements a D3-style agentic classifier using a local Qwen causal language model. The pipeline decomposes classification into specialized stages:

1. **Question Intent** — identifies what the question requires.
2. **Answer Content** — extracts what the answer actually addresses.
3. **Gap/Evasion** — identifies omissions, deflection, or evasion.
4. **Decision** — combines the intermediate analyses into the final clarity label.

The notebook includes structured JSON parsing, checkpointed inference, error analysis by evasion category and gap level, DSPy-based prompt optimization, and ablation experiments that remove selected agents.

On the reported 50-example agentic evaluation subset:

| System | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| 3-agent ablation, skipping Gap/Evasion | 0.52 | 0.3707 | 0.4911 |
| Original D3-Agentic pipeline | 0.53 | 0.3497 | 0.5142 |
| 3-agent ablation, skipping Question Intent | 0.46 | 0.3285 | 0.4419 |
| 2-agent Question Intent + Decision | 0.38 | 0.1836 | 0.2093 |

Among the original D3 variants, the notebook selects the original prompting pipeline by macro F1 and generates predictions for all 308 test examples. The ablations show that intermediate agent roles affect performance materially, but they do not establish that adding more agents always improves balanced classification.

## Overall progression

The notebooks form a methodological progression:

1. **Classical feature engineering:** TF-IDF and Word2Vec baselines.
2. **Supervised representation learning:** fine-tuned BERT-family encoders.
3. **Instruction-based inference:** zero-shot, few-shot, and reasoning-oriented prompts.
4. **Structured agentic reasoning:** decomposed analysis, ablations, and prompt optimization.

The main empirical conclusion is that supervised fine-tuning is the most reliable approach in these experiments, with BERT producing the highest reported validation accuracy and macro F1. TF-IDF remains a competitive, inexpensive baseline. Generative prompting and agentic decomposition provide useful interpretability and experimentation opportunities, but their results are more sensitive to model size, prompt format, output parsing, and evaluation subset size.

## Reproducibility notes

The notebooks are designed for GPU-enabled Kaggle-style environments and download models/datasets at runtime. They use seed `42`, but exact results can still vary with library versions, CUDA behavior, model revisions, and generation settings. The reported scores are validation results from the notebooks, not a unified benchmark on one identical evaluation protocol; the prompting and D3 experiments use smaller evaluation subsets than the supervised baselines.
