# Deep Multimodal Fusion for Clinical Trial Outcome Prediction

## Introduction
Many clinical trials fail in the pharmaceutical industry, which causes significant financial losses and delays in life-saving treatments. Existing prediction models are not effective because they depend only on structured data or cannot handle the strong imbalance between successful and failed trials. In this study, we present a Deep Multimodal Fusion Architecture to predict clinical trial outcomes as a high-risk task. Our model combines structured trial data with semantic information from trial protocols, focusing on the early identification of failed trials to reduce costly failures.

## Novelty
Our study provides three important technical breakthroughs to solve the shortcomings of current baselines and satisfy the demanding requirements of this domain:

* **Advanced Text Representation (BioClinical-ModernBERT):** Long-range dependencies in intricate eligibility requirements are frequently difficult for standard BioBERT to capture. In order to extract deeper, context-aware embeddings that outperform historical BERT models, we employ **BioClinical-ModernBERT**, a cutting-edge model pre-trained on a vast mixture of biomedical literature.

* **Strategic Imbalance Handling:** The dataset presents a critical 1:6 failure-to-success ratio. While traditional methods use SMOTE, our experiments revealed that synthetic oversampling introduces noise in this high-dimensional space. We instead employ a strict **Random Downsampling** strategy on the majority class, which proved essential for recovering the minority failure signal and maximizing Recall.

* **Intermediate Multimodal Fusion:** We propose utilizing an Intermediate Fusion Neural Network instead of evaluating text or information separately. This architecture preserves the distinct prediction power of both modalities by processing text embeddings (by Transformer) and structured features (via MLP) in parallel before combining them at a deep latent layer.
