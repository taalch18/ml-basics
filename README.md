# ML Basics From Scratch

Minimal implementations of core machine learning concepts, written from scratch first and then mirrored using standard Python libraries.

This repository is built on a simple belief:  
**understanding comes before abstraction**.  
Instead of starting with buzzwords and high-level APIs, each idea here begins with a clear, explicit implementation that shows what is actually happening.


## What this repo contains

- Core ML concepts implemented **from scratch**
- Side-by-side **library equivalents** for comparison
- Small, readable, self-contained scripts
- Focus on reasoning, not performance or scale

This is a fundamentals repo, not a production framework.


## Repository structure

ml-basics-from-scratch/
│
├── linear_models.py # Linear & logistic regression
├── tree_models.py # Decision trees
├── neural_networks.py # MLPs and backpropagation
├── optimization.py # SGD, Momentum, Adam
├── embeddings.py # PCA, similarity, embeddings
├── sequence_models.py # Vanilla RNNs
├── rag_basics.py # RAG fundamentals
│
├── utils.py
└── README.md


Each file follows the same pattern:
1. Short concept explanation  
2. From-scratch implementation  
3. Minimal library version  
4. Small runnable example  

No nested folders. No hidden abstractions.


## Libraries used

Libraries are introduced only after the scratch version exists.

- `numpy`
- `scikit-learn`
- `torch`
- Lightweight embedding and retrieval tools where needed

The goal is not to avoid libraries, but to **understand what they abstract away**.

## How to run

Each file is self-contained.

```bash
python linear_models.py
python neural_networks.py
python rag_basics.py
```

No configuration files. No setup rituals.No configuration files. No setup rituals.


## What this repo is not

- Not a collection of copy-pasted projects
- Not a benchmark or SOTA showcase
- Not optimized for production or large-scale training

Those come later. This repo is about foundations.


## Final note

Most ML failures are not caused by missing libraries, but by missing understanding.

This repository is an attempt to build that understanding properly, starting from scratch and earning each abstraction along the way.
