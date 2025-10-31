[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/hgNAtOO3)
## Predicting Clinical Trial Success through Multimodal Data Fusion.
## Abstract  
Clinical trial failure is a major obstacle to medical progress, causing severe financial losses and delays in patients' access to novel treatments.
This project aims to be a Multimodal Data Fusion to predict the success of clinical trials using the Clinical Trials dataset. Existing prediction models often lack context, analyzing unstructured material superficially or relying solely on structured metadata. The main innovation of our method is its **Multimodal Fusion Architecture**, which explicitly combines **deep semantic embeddings** from the eligibility criteria and summary text with the trial's **structured features** (such as sponsor and phase). This combination offers a comprehensive understanding of the scientific justification and trial mechanics. By outperforming a structured-data baseline model in terms of prediction, we demonstrate this benefit and provide insightful information that can be used to improve trial design in the future.
## Novelty  
Our project fills a significant gap in the current literature on clinical trial prediction, which often falls into three categories: **unimodal prediction**, **sequential modeling**, and **statistical inference**. Sequential Predictive Models (like SPOT)** take advantage of temporal relationships, while **Prediction-Powered Inference (PPI)** and **Sequential Predictive Models (like SPOT)** concentrate on maximizing statistical power after a trial has started using structured covariates, none of which fully incorporate the content itself. In addition, current **Deep Multimodal Classification Models** give more attention to the **physicochemical properties of the drug** than to the design of the trial. Our main contribution is the development of a unique and dedicated Multimodal Fusion Architecture. We shift our focus to the **trial’s implementation and underlying scientific reasoning** by integrating structured metadata, such as phase and sponsor details, with unstructured textual data like eligibility criteria and summaries. This integration gives a comprehensive and predictive perspective that helps explain the factors why trials succeed and fail. 

## Proposed Timeline

Our team is focused on submitting the  P2 proposal by November 7th and following a structured timeline to ensure P3 is also successful.

* **Phase 1 (Now until Nov 7): P2 Milestone Finalization**
    * Complete EDA and Baseline Model (Esther/Anders).
    * Finalize **README.md** (All sections, including Methods and Timeline).
    * Ensure the `main.ipynb` notebook runs error-free.
* **Phase 2 (Nov 8 - Dec 5): Deep Multimodal Model Development**
    * Develop the **Deep Multimodal Fusion Architecture**.
    * Integrate structured features with deep text embeddings.
    * Run optimization and hyperparameter tuning.
* **Phase 3 (Dec 6 - Dec 19): Final Report & Review**
    * Write the **6-page NeurIPS-style Final Report** (Saznila/Anders).
    * Final code review and documentation (Esther).
    * Project submission by **December 19th**.
## Organization within the Team

We have established clear roles to ensure accountability and a structured workflow. While we will all contribute across areas, these roles define the final responsibility for each deliverable:

* **Project Coordinator (Anders):** Manages the GitHub repository (branches, commits), monitors deadlines, and ensures consistency between the final README and the Notebook.
* **Lead Analyst / Coder (Esther):** Leads the data pipeline, completes the EDA, builds the **Baseline Model**, and develops the final **Multimodal Fusion Model**. Responsible for ensuring the `main.ipynb` runs smoothly.
* **Lead Writer / Researcher (Saznila):** Leads the writing of the **README.md**, performs the **Literature Review** (Novelty Statement), and is responsible for all **textual descriptions** and **analysis interpretations** within the Jupyter notebook.
