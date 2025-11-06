[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/hgNAtOO3)

# Predicting Clinical Trial Success through Multimodal Data Fusion.

## Abstract  

Clinical trial failure is a major obstacle to medical progress, resulting in significant financial losses and delays in patients' access to novel treatments.

Our project aims to use the Clinical Trials dataset to build a comprehensive classification model utilizing the strengths of both structured metadata, as well as semantic text embeddings in a combined multimodal data fusion, with the goal of predicting the success of clinical trials. Existing prediction models often lack context, analysing unstructured material superficially or relying solely on structured metadata. The main innovation of our method is its **Multimodal Fusion Architecture**, which explicitly combines **deep semantic embeddings** from the eligibility criteria and summary text with the trial's **structured features** (such as sponsor and phase). This combination offers a comprehensive understanding of the scientific justification and trial mechanics. By outperforming a structured-data baseline model in predictive performance, we demonstrate this benefit and provide insights that can be used to improve trial design in the future.

## Novelty  

Our project aims to address a significant gap in the current literature on clinical trial prediction, which often focuses on specific data modalities without providing a holistic view. While existing work has explored **unimodal prediction** from structured data [4], **statistical inference** (PPI) [1], and **sequential modeling** (SPOT) [2], these approaches do not fully leverage the rich semantic content of the trial protocols. Similarly, existing **multimodal models** [3] often focus on drug properties rather than trial design.

Our primary contribution is a novel **Multimodal Fusion Architecture** that focuses on the trial’s design and scientific rationale. Our specific contributions are:

*   **A Novel Feature Set**
    We introduce a feature set that systematically fuses low-dimensional structured metadata (phase, sponsor) with high-dimensional semantic embeddings generated from the trial's `brief_summary` and `eligibility_criteria`. This provides a more comprehensive representation of the trial's operational and scientific context.

*   **A Robust Methodological Benchmark**
    We provide a clear, data-driven comparison of multiple modeling approaches: a structured-only baseline, an embeddings-only baseline, and our proposed final multimodal model. This allows us to rigorously quantify the predictive value added by the NLP features.

*   **Actionable Insights Beyond Prediction**
    Our project aims to go beyond building a classifier. By analyzing model results and feature importances, we aim to deliver actionable insights into the key factors and patterns that correlate with clinical trial success or failure.

## Methods

Our methods for P2 are divided into four main stages: Exploratory Data Analysis (EDA), Data Preparation, Baseline Modeling, and an experimental test of a proposed solution.

**1. Exploratory Data Analysis (EDA)**
We began with a comprehensive EDA on the full dataset to understand its characteristics. This analysis revealed several key properties that guided our approach:
*   **Target Imbalance:** The `overall_status` target variable is severely imbalanced, with "Completed" trials significantly outnumbering "Terminated," "Withdrawn," and "Suspended" trials.
*   **Feature Skewness:** Numerical features like `enrollment_count` and the age columns were found to be heavily right-skewed and contained significant outliers.
*   **Data Quality:** We identified quirks in the data, such as the `phases` column containing a meaningful "NA" (Not Applicable) category and the `minimum_age` feature behaving more like a categorical variable than a continuous one.
*   **Embeddings Generation:** We discovered that the pre-computed embeddings were not included in the default dataset and needed to be generated manually. Due to computational constraints, this was done on a 140,000-row subset using a `thomas-sounack/BioClinical-ModernBERT-base` sentence transformer, and the results were saved for our proof-of-concept models.

**2. Data Preparation and Pipeline**

To ensure a robust and reproducible workflow, we developed a preprocessing pipeline with a strict train/test split to prevent data leakage. All modeling was performed on an 80/20 split of our enhanced 140k-row subset.

*   **Missing Values:** We handled missing data by dropping rows for features with minimal missingness (e.g., `sex`) and imputing with the median for numerical features with moderate missingness (e.g., `minimum_age`).

*   **Feature Encoding:** Categorical features were converted into a numerical format using One-Hot Encoding.

*   **Feature Scaling:** Numerical features were standardized using a `StandardScaler` to have a mean of 0 and a standard deviation of 1.

**3. Baseline Modeling**

We established two distinct baseline models to isolate the predictive power of different data modalities:

*   **Baseline 1 (Structured Data):** A `RandomForestClassifier` was trained on the preprocessed structured metadata. This model served as our primary benchmark and achieved a weighted F1-score of 0.89, but showed poor recall (0.48) on the minority "Failure" class.

*   **Baseline 2 (Embeddings):** We first trained a `LogisticRegression` model on the 1536-dimensional concatenated text embeddings, which performed very poorly (Failure F1-score: 0.06). A subsequent `RandomForestClassifier` failed completely, collapsing into a "lazy classifier" that only predicted the majority class (Failure F1-score: 0.00).

**4. Downsampling Experiment**

To test our hypothesis that class imbalance was the primary issue, we conducted a final experiment. We trained a new `RandomForestClassifier` on a randomly downsampled version of the embeddings training data. This model showed a dramatic improvement in identifying the minority class, increasing the **recall for "Failure" from 0.00 to 0.63. This result provides strong evidence that addressing the data distribution is a critical step for our final P3 model.

## Proposed Timeline

* **Phase 1 (Now until Nov 7): P2 Milestone Finalization**
    * Complete EDA and Baseline Model.
    * Finalize **README.md** (All sections, including Methods and Timeline).
    * Ensure the `main.ipynb` notebook runs error-free.
* **Phase 2 (Nov 8 - Dec 5): Deep Multimodal Model Development**
    * Develop the **Deep Multimodal Fusion Architecture**.
    * Integrate structured features with deep text embeddings.
    * Run optimization and hyperparameter tuning.
* **Phase 3 (Dec 6 - Dec 19): Final Report & Review**
    * Write the **6-page NeurIPS-style Final Report**.
    * Final code review and documentation.
    * Project submission by **December 19th**.

# Organization within the Team

Even though roles are defined, group members all help each other where needed. 

* **Project Coordinator (Anders):** Manages the GitHub repository (branches, commits), monitors deadlines, and ensures consistency between the final README and the Notebook. Responsible for creating .gitignore, requirements.txt and other needed files. Makes sure that everything is structured correctly for hand ins.   
* **Lead Analyst / Coder (Esther):** Leads the data pipeline, completes the EDA, builds the **Baseline Model**, and develops the final **Multimodal Fusion Model**. Responsible for ensuring the `main.ipynb` runs smoothly. Is responsible for ensuring that discussions and explanations within the notebook are written and structured correctly. 
* **Lead Writer / Researcher (Saznila):** Leads the writing of the **README.md**, performs the **Literature Review** (Novelty Statement), and is responsible for all **textual descriptions** within the README.
---

## Appendix

### A.1. Repository Organisation
*   `main.ipynb`: The main Jupyter notebook containing all EDA, model training, and analysis for our P2 proof of concept.
*   `utils.py`: A utility script containing our data preprocessing pipeline functions.
*   `README.md`: This file, containing the detailed project proposal.
*   `subset-embedding-training.ipynb`: The Google Colab notebook used to perform the computationally intensive text embedding generation on a GPU.
*   `requirements.txt`: A file listing the Python libraries required to run the code. Use "pip install -r requirements.txt". 
*   `GAI_Declaration.pdf`: The declaration of our use of Generative AI tools in this project, as required by university policy.

### A.2. Project Datasets
*   **Original Dataset:** The raw data was sourced from the Hugging Face Hub: [`louisbrulenaudet/clinical-trials`](https://huggingface.co/datasets/louisbrulenaudet/clinical-trials).
*   **Enhanced Subset Dataset:** The `subset_dataset_with_embeddings.parquet` file, which was generated for this project, is too large for this repository. It can be downloaded from the following link:
    *   [**Download Link for Parquet File**](https://drive.google.com/file/d/11drOEiW5jSJPtK_AksO1mub21Luveubh/view?usp=drive_link)

## References

[1] Fu, D. J., et al. "Prediction-powered inference." *Proceedings of the National Academy of Sciences*, 119.33 (2022): e2120786119. `https://www.pnas.org/doi/10.1073/pnas.2120786119`

[2] Fu, D. J., et al. "SPOT: A tool for selection of patients for trials using real-world data." *arXiv preprint arXiv:2304.05352* (2023). `https://arxiv.org/abs/2304.05352`

[3] Hwang, Changha. "Deep Multimodal Classification Model for Predicting Successes and Failures of Clinical Trials." (2020).

[4] Liu, R., et al. "Systematic review of machine learning applications in clinical trial design." *Contemporary Clinical Trials*, 141 (2024): 107567. `https://www.sciencedirect.com/science/article/pii/S155171442400035X`

[5] Li, J. (2025). Predicting clinical trial completion and success using machine learning and natural language processing (Master’s thesis). The University of Chicago.

