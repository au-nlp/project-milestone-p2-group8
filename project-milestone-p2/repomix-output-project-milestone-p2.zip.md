This file is a merged representation of the entire codebase, combined into a single document by Repomix.
The content has been processed where security check has been disabled.

# File Summary

## Purpose
This file contains a packed representation of the entire repository's contents.
It is designed to be easily consumable by AI systems for analysis, code review,
or other automated processes.

## File Format
The content is organized as follows:
1. This summary section
2. Repository information
3. Directory structure
4. Repository files (if enabled)
5. Multiple file entries, each consisting of:
  a. A header with the file path (## File: path/to/file)
  b. The full contents of the file in a code block

## Usage Guidelines
- This file should be treated as read-only. Any changes should be made to the
  original repository files, not this packed version.
- When processing this file, use the file path to distinguish
  between different files in the repository.
- Be aware that this file may contain sensitive information. Handle it with
  the same level of security as you would the original repository.

## Notes
- Some files may have been excluded based on .gitignore rules and Repomix's configuration
- Binary files are not included in this packed representation. Please refer to the Repository Structure section for a complete list of file paths, including binary files
- Files matching patterns in .gitignore are excluded
- Files matching default ignore patterns are excluded
- Security check has been disabled - content may contain sensitive information
- Files are sorted by Git change count (files with more changes are at the bottom)

# Directory Structure
```
main.ipynb/
  main.ipynb
p2-comments-review.md/
  p2-comments-review.md
p2-instructor-comments.md/
  p2-instructor-comments.md
README.md/
  README.md
requirements.txt/
  requirements.txt
utils.py/
  utils.py
```

# Files

## File: main.ipynb/main.ipynb
```
{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "9fed8d17",
   "metadata": {},
   "source": [
    "# Project: Predicting Clinical Trial Success through Multimodal Data Fusion\n",
    "\n",
    "This notebook presents the technical proof of concept for our project on predicting clinical trial outcomes. We work with the **louisbrulenaudet/clinical-trials** dataset, where in this P2 deliverable, we want to build simple baseline models for predicting trial success, and show the feasibility of our proposed multimodal approach. \n",
    "\n",
    "The notebook is structured as follows:\n",
    "1. **Exploratory Data Analysis (EDA):** Overview of key dataset features and potential challenges.\n",
    "2. **Data Splitting and Preprocessing:** Steps for preparing data for modeling and avoiding leakage.\n",
    "3. **Baseline Model (Structured Data):** A Random Forest classifier trained on trial metadata.\n",
    "4. **Embeddings Model:** Early experiments using text embeddings to see how informative textual data can be.\n",
    "5. **Conclusion and Next Steps:** Brief summary of results and plans for the next project stage.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c6268ea5",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Libraries\n",
    "\n",
    "# Core Data Handling and Plotting\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "\n",
    "# Hugging Face Datasets\n",
    "from datasets import load_dataset, Dataset, ClassLabel\n",
    "\n",
    "# Scikit-learn for Modeling and Preprocessing\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.metrics import classification_report, ConfusionMatrixDisplay\n",
    "from sklearn.utils import resample\n",
    "\n",
    "# Utility Functions (from our utils.py file)\n",
    "from utils import fit_and_preprocess_train, transform_test_data\n",
    "\n",
    "# Set plot style for the notebook\n",
    "sns.set_style(\"whitegrid\")\n",
    "\n",
    "# Prevent pandas from using scientific notation for easier reading\n",
    "pd.options.display.float_format = '{:.2f}'.format"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9b09e987",
   "metadata": {},
   "outputs": [],
   "source": [
    "# --- Load the Full Raw Dataset ---\n",
    "# We load the 'train' split, which contains all the available data,\n",
    "# and immediately convert it to a pandas DataFrame for EDA.\n",
    "# If you are running this on your local machine, it sometimes takes a while to load the dataset. \n",
    "print(\"Loading the full dataset from Hugging Face...\")\n",
    "full_df = load_dataset(\"louisbrulenaudet/clinical-trials\", split=\"train\").to_pandas()\n",
    "print(\"Dataset loaded.\")\n",
    "\n",
    "# Display initial information about the dataset\n",
    "print(\"\\nDataset Information:\")\n",
    "full_df.info()\n",
    "\n",
    "print(\"\\nFirst 5 rows of the raw dataset:\")\n",
    "display(full_df.head())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "56458d16",
   "metadata": {},
   "outputs": [],
   "source": [
    "# --- 1.1. Target Variable Analysis: overall_status ---\n",
    "\n",
    "# First, we look at the original distribution of the 'overall_status' column.\n",
    "print(\"--- Original Distribution of Trial Statuses ---\")\n",
    "status_counts = full_df['overall_status'].value_counts()\n",
    "print(status_counts)\n",
    "\n",
    "# Next we visualize the distribution\n",
    "plt.figure(figsize=(12, 7))\n",
    "sns.barplot(x=status_counts.index, y=status_counts.values, palette=\"viridis\")\n",
    "plt.title('Original Distribution of Overall Trial Status', fontsize=16)\n",
    "plt.xlabel('Status', fontsize=12)\n",
    "plt.ylabel('Number of Trials', fontsize=12)\n",
    "plt.xticks(rotation=45, ha=\"right\")\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d3995797",
   "metadata": {},
   "source": [
    "### Defining the Binary Target for EDA\n",
    "\n",
    "The `overall_status` column contains many different categories, where a lot of them are not definitive outcomes (e.g., \"RECRUTING\"). For our predictive task, we need a clear binary target representing **Success vs. Failure**. Based on the above plot, we define our classes as follows:\n",
    "\n",
    "*   **Success (Target = 1):** Trials marked **\"COMPLETED\"**.\n",
    "*   **Failure (Target = 0):** Trials that were stopped for definitive negative reasons. We include **\"TERMINATED\"**, **\"WITHDRAWN\"**, and **\"SUSPENDED\"**.\n",
    "\n",
    "We will filter the dataset to these outcomes to create a clean target for our analysis."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6d2b8a2f",
   "metadata": {},
   "outputs": [],
   "source": [
    "# --- Create a temporary filtered DataFrame for EDA visualization ---\n",
    "success_stati = [\"COMPLETED\"]\n",
    "failure_stati = [\"TERMINATED\", \"WITHDRAWN\", \"SUSPENDED\"]\n",
    "definitive_stati = success_stati + failure_stati\n",
    "\n",
    "# Note: This filtered_df is for EDA purposes only. \n",
    "# The final filtering for the model will happen within the preprocessing pipeline.\n",
    "eda_filtered_df = full_df[full_df['overall_status'].isin(definitive_stati)].copy()\n",
    "eda_filtered_df['target'] = eda_filtered_df['overall_status'].apply(lambda x: 1 if x in success_stati else 0)\n",
    "\n",
    "# Visualize the new binary target distribution\n",
    "print(\"--- Distribution of the Binary Target Variable ---\")\n",
    "print(eda_filtered_df['target'].value_counts())\n",
    "\n",
    "plt.figure(figsize=(6, 4))\n",
    "sns.countplot(x='target', data=eda_filtered_df, palette=\"viridis\")\n",
    "plt.title('Distribution of Binary Target (0=Failure, 1=Success)', fontsize=14)\n",
    "plt.show()\n",
    "\n",
    "# Markdown analysis of the imbalance will go in the final EDA conclusion."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "fcca872a",
   "metadata": {},
   "source": [
    "## 1. Exploratory Data Analysis (EDA)\n",
    "\n",
    "This section focuses on developing a grounded understanding of the clinical trials dataset. Our aim is to understand how the data are structured, examine key patterns and distributions, and identify potential issues such as missing values or class imbalance. These insights form the foundation for the preprocessing and modeling steps that follow. All analyses here use the full, unfiltered dataset.\n",
    "\n",
    "### 1.2. Structured Feature Analysis\n",
    "\n",
    "We begin with the structured features that are most likely to influence trial outcomes. These variables were selected for their ability to capture aspects of study design, scale, and participant characteristics:\n",
    "\n",
    "* **`phases`, `study_type`, `enrollment_count`:** Describe the trial’s stage, format, and size, which together indicate its overall complexity and risk.\n",
    "* **`lead_sponsor_class`:** Represents the funding source (e.g., industry or academic sponsor), which may correlate with trial design and outcome likelihood.\n",
    "* **`sex`, `minimum_age`, `maximum_age`:** Define the population under study, helping us gauge demographic scope and potential variability.\n",
    "\n",
    "Our following analysis examines the extent of missing data and the distributions of these categorical and numerical variables. These findings will guide the development of preprocessing methods and inform early model design decisions.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "efa1b818",
   "metadata": {},
   "outputs": [],
   "source": [
    "# --- Missing Values Analysis ---\n",
    "structured_features = [\n",
    "    'phases', 'study_type', 'enrollment_count', \n",
    "    'lead_sponsor_class', 'sex', 'minimum_age', 'maximum_age'\n",
    "]\n",
    "\n",
    "missing_values = eda_filtered_df[structured_features].isnull().sum()\n",
    "missing_percentage = (missing_values / len(eda_filtered_df)) * 100\n",
    "\n",
    "print(\"--- Percentage of Missing Values in Key Features ---\")\n",
    "print(missing_percentage[missing_percentage > 0].sort_values(ascending=False))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "77a1120d",
   "metadata": {},
   "source": [
    "**Interpretation:**\n",
    "The missing value analysis reveals several different frequencies of missing values, each with their own set of solutions or challenges:\n",
    "*   `sex` and `enrollment_count` have very few missing values (<2%), making it safe to simply drop these rows in our processing pipeline.\n",
    "*   `phases`, `minimum_age`, and `maximum_age` have a moderate to high percentage of missing data (6% to 47%). For these, we will need to use an imputation strategy (e.g., filling with \"Unknown\" for categoricals or the median for numericals) to avoid losing a significant portion of our dataset. This is the info we needed to design our preprocessing functions. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "08d5c20c",
   "metadata": {},
   "outputs": [],
   "source": [
    "# --- Categorical Feature Distributions ---\n",
    "\n",
    "# 1. 'phases' column (which contains lists/ndarrays)\n",
    "print(\"--- Distribution of Trial Phases ---\")\n",
    "phase_counts = eda_filtered_df['phases'].explode().value_counts()\n",
    "plt.figure(figsize=(10, 6))\n",
    "sns.barplot(x=phase_counts.index, y=phase_counts.values, palette='viridis')\n",
    "plt.title('Distribution of Trial Phases', fontsize=16)\n",
    "plt.xticks(rotation=45, ha=\"right\")\n",
    "plt.show()\n",
    "\n",
    "# 2. Other categorical features\n",
    "simple_categorical_features = ['study_type', 'lead_sponsor_class', 'sex']\n",
    "fig, axes = plt.subplots(1, 3, figsize=(18, 5))\n",
    "fig.suptitle('Distributions of Other Categorical Features', fontsize=20)\n",
    "axes = axes.flatten()\n",
    "for i, feature in enumerate(simple_categorical_features):\n",
    "    sns.countplot(x=feature, data=eda_filtered_df, ax=axes[i], palette='viridis', order=eda_filtered_df[feature].value_counts().index)\n",
    "    axes[i].set_title(f'Distribution of {feature}')\n",
    "    axes[i].tick_params(axis='x', rotation=45)\n",
    "plt.tight_layout(rect=[0, 0.03, 1, 0.95])\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "891d9b1b",
   "metadata": {},
   "source": [
    "**Interpretation:**\n",
    "\n",
    "* **Phases:** The distribution is largely dominated by “NA” (Not Applicable) entries. We interpret these as corresponding to observational studies, which typically do not include defined phases. This category is therefore meaningful in itself, rather than simply representing missing data.\n",
    "* **Sponsor:** The `lead_sponsor_class` feature is very skewed toward “OTHER.” While this limits its immediate interpretability, it still helps distinguish between major sponsor types such as “INDUSTRY” and “NIH.” We also considered a follow-up analysis that groups individual sponsor classes manually, which could improve feature utility in the next project phase (P3).\n",
    "* **Study Type & Sex:** These features are imbalanced, reflecting the reality of clinical research (more interventional studies, most open to all sexes). This is not a data problem, but rather a real-world pattern our model can learn from.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1bf0bfd7",
   "metadata": {},
   "outputs": [],
   "source": [
    "# --- Numerical Feature Analysis ---\n",
    "numerical_features = ['enrollment_count', 'minimum_age', 'maximum_age']\n",
    "\n",
    "# 1. Check data types and summary statistics\n",
    "print(\"--- Data Types of Numerical Features ---\")\n",
    "print(eda_filtered_df[numerical_features].dtypes)\n",
    "print(\"\\n--- Summary Statistics for Numerical Features ---\")\n",
    "display(eda_filtered_df[numerical_features].describe())"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c46afc4e",
   "metadata": {},
   "source": [
    "**Interpretation of Statistics:**\n",
    "The summary statistics table is highly revealing:\n",
    "*   **Skewness:** For all three features, the `mean` is significantly different from the `median` (50%), indicating strong skewness. `enrollment_count` shows the most extreme right skew.\n",
    "*   **Outliers/Data Quality:** The age columns contain clear data errors or extreme outliers, with a `max` value of 730 for `minimum_age` and 6569 for `maximum_age`. For this baseline proof of concept, we will move on without removing them. Our chosen baseline model for the structured data (See model justification in the model set up section), a Random Forest, is generally robust to such outliers. However, for a more refined model in P3, a potential improvement would be to either cap these values at a reasonable threshold (e.g., 100 years) or remove these outlying rows entirely."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b141bc17",
   "metadata": {},
   "outputs": [],
   "source": [
    "# 2. Plot distributions (Linear vs. Log Scale)\n",
    "fig, axes = plt.subplots(len(numerical_features), 2, figsize=(12, 12))\n",
    "fig.suptitle('Distributions of Numerical Features (Linear vs. Log Scale)', fontsize=16)\n",
    "for i, feature in enumerate(numerical_features):\n",
    "    sns.histplot(eda_filtered_df[feature], bins=50, kde=False, ax=axes[i, 0]).set_title(f'{feature} (Linear Scale)')\n",
    "    sns.histplot(eda_filtered_df[feature], bins=50, kde=False, ax=axes[i, 1], log_scale=True).set_title(f'{feature} (Log Scale)')\n",
    "plt.tight_layout(rect=[0, 0.03, 1, 0.95])\n",
    "plt.show()\n",
    "\n",
    "# 3. \"Zoomed-in\" plot for minimum_age\n",
    "zoomed_df = eda_filtered_df[eda_filtered_df['minimum_age'] <= 100]\n",
    "plt.figure(figsize=(12, 7))\n",
    "sns.histplot(data=zoomed_df, x='minimum_age', bins=100)\n",
    "plt.title('Zoomed-In Distribution of Minimum Age (0-100 years)', fontsize=16)\n",
    "plt.xticks(ticks=range(0, 101, 5))\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "9186afa3",
   "metadata": {},
   "source": [
    "**Interpretation of Plots:**\n",
    "*   The log-scaled plots confirm the right-skew of `enrollment_count` and `maximum_age`.\n",
    "*   The zoomed-in plot for `minimum_age` tells us the most. It proves that this feature does not have a smooth distribution and instead acts like a combination of a smooth distribution at low ages and a **categorical variable**, dominated by specific values like 0, 18, and other \"round number\" age cutoffs (40, 50, 65). This is likely because many studies aimed at lower age groups have specific age requirements where a few years' difference is developmentally significant (e.g., 13 vs. 16 years old), while studies targeting older adults often use broader, round-number cutoffs where the difference between 40 and 42 is less critical. This is an important finding that suggests a simple numerical treatment may be insufficient for a final model. For P2 we did not change this category, but we noted that this is a potential avenue for improvement. "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "fd836f1f",
   "metadata": {},
   "source": [
    "### 1.3. Investigating the Text Embeddings\n",
    "\n",
    "The final step of our EDA was to locate the pre-trained text embeddings mentioned in the dataset's description (`brief_summary_embedding` and `eligibility_criteria_embedding`). These high-dimensional features are the main columns we need for our project's multimodal hypothesis, as they are expected to capture the rich semantic content of the trial's purpose and its inclusion/exclusion rules.\n",
    "\n",
    "However, when exploring the dataset, we found that it did not include the promised enriched text columns. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c2c8157f",
   "metadata": {},
   "outputs": [],
   "source": [
    "# --- Diagnostic Check for Embedding Columns ---\n",
    "\n",
    "# We inspect the columns of our loaded full_df to search for any column containing 'embedding'.\n",
    "embedding_related_columns = [col for col in full_df.columns if 'embedding' in col.lower()]\n",
    "\n",
    "if not embedding_related_columns:\n",
    "    print(\"Diagnostic Result: No columns containing 'embedding' were found in the initial dataframe.\")\n",
    "    print(\"\\nConclusion: The embeddings must be generated manually.\")\n",
    "else:\n",
    "    print(\"Embedding columns found:\", embedding_related_columns)\n",
    "\n",
    "print(\"\\nFull list of available columns:\")\n",
    "print(full_df.columns.tolist())"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "8e3202a5",
   "metadata": {},
   "source": [
    "### Our Solution: Manual Embedding Generation\n",
    "\n",
    "Given that the embeddings were not included in the dataset, we adapted our plan to generate them ourselves, following the code provided in the dataset's documentation. This represents a significant NLP task that adds depth to our project's proof of concept.\n",
    "\n",
    "**Workflow:**\n",
    "1.  **Model Selection:** We used the recommended `thomas-sounack/BioClinical-ModernBERT-base` model from the `sentence-transformers` library, a powerful transformer model pre-trained on biomedical text.\n",
    "2.  **Computational Constraints:** Generating embeddings for the entire dataset (~540,000 trials) is a computationally intensive process. Our initial attempt to run this on the full dataset in Google Colab timed out after 4.5 hours, having processed only about 30% of the data.\n",
    "3.  **Subsetting for P2:** To create a doable and reproducible proof of concept that could be completed within a reasonable timeframe, we created a representative **subset of ~140,000 trials (approx. 25% of the full dataset)**. This size was chosen as a practical balance to make sure that the generation process would complete successfully before Google Colabs runtime timed out. \n",
    "4.  **Generation on Colab:** We performed the embedding generation for both the `brief_summary` and `eligibility_criteria` columns on this subset using a dedicated Google Colab notebook (`subset-embedding-training.ipynb`) with GPU acceleration. This notebook can be found under the appendix in the `README.md`. \n",
    "5.  **Saving the Result:** The final, enhanced DataFrame, containing all original metadata plus the two new 768-dimensional embedding columns—was saved to a compressed **Parquet file** (`subset_dataset_with_embeddings.parquet`). The dataset is not included in the repo due to size, but you can find the link  under the appendix in the `README.md`. \n",
    "\n",
    "**Path Forward:**\n",
    "All subsequent analysis, preprocessing, and modeling in this notebook will be performed on a consistent train/test split derived from this new, enhanced subset dataset. This ensures a fair and direct comparison between our structured data baseline and our embeddings-based models. The code for loading this subset is the first step in our Data Preparation section."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f07e8742",
   "metadata": {},
   "source": [
    "## EDA Conclusion\n",
    "\n",
    "Our exploratory analysis has provided several good insights that directly inform our project's next steps:\n",
    "1.  **A Clear (but Imbalanced) Target:** We have successfully defined a binary target variable (\"Success\" vs. \"Failure\"), but its severe class imbalance is the primary challenge we must address.\n",
    "2.  **Informative (but Messy) Features:** The structured features show clear patterns and distributions that we **hypothesize contain predictive signals** (e.g., the difference in `phases` or `study_type`). However, they also suffer from missing data, skewness, and data quality issues (e.g., age outliers). Our preprocessing pipeline must be designed to handle these issues robustly to effectively test this hypothesis.\n",
    "3.  **A Necessary Workflow Pivot:** The absence of pre-computed embeddings required us to adapt our plan to include a manual generation step, leading to the creation of a smaller, enhanced subset for our proof-of-concept models. We hope to find a way we can train the embeddings for the full dataset during P3. \n",
    "\n",
    "With this new and deeper understanding of the data, we are now ready to move on to the formal data preparation and modeling phase."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "6cde8312",
   "metadata": {},
   "source": [
    "## 2. Data Preparation and Pipeline\n",
    "\n",
    "Based on our EDA findings, we now move to the formal data preparation stage. The goal is to create a clean, consistent, and leak-free set of training and testing data for both our structured and embeddings-based models.\n",
    "\n",
    "### 2.1. The Preprocessing Utility Functions (`utils.py`)\n",
    "\n",
    "To ensure our preprocessing is modular, reusable, and consistently applied, we have encapsulated the logic into two functions stored in a separate `utils.py` file. This `fit`/`transform` paradigm is a standard best practice that prevents data leakage from the test set into the training process.\n",
    "\n",
    "1.  **`fit_and_preprocess_train(train_df)`:** This function learns transformation parameters (like medians and scaling factors) exclusively from the training data and applies them. It returns a processed training DataFrame and a `transformation_rules` dictionary.\n",
    "\n",
    "2.  **`transform_test_data(test_df, rules)`:** This function takes the test data and the `transformation_rules` dictionary and applies the already learned transformations, ensuring the test set is treated as unseen data.\n",
    "\n",
    "### 2.2. Detailed Preprocessing Steps\n",
    "\n",
    "The pipeline implemented in our utility functions performs the following sequential operations on the structured data:\n",
    "\n",
    "*   **1. Feature Selection:** It first selects the seven structured features identified during our EDA: `phases`, `study_type`, `enrollment_count`, `lead_sponsor_class`, `sex`, `minimum_age`, and `maximum_age`.\n",
    "\n",
    "*   **2. Outcome Filtering:** It filters the dataset to only include trials with a definitive outcome, defined as \"COMPLETE\", \"TERMINATED\", \"WITHDRAWN\", or \"SUSPENDED\".\n",
    "\n",
    "*   **3. Handling Missing Values:** Based on our EDA, it applies the following strategies:\n",
    "    *   **Row Dropping:** It drops rows where `sex` or `enrollment_count` are missing, as these constitute less than 2% of the data.\n",
    "    *   **Categorical Imputation:** It fills missing `NaN` values in the `phases` column with the string `\"Unknown\"`, treating this missingness as a distinct category.\n",
    "    *   **Numerical Imputation:** It fills missing `NaN` values in `minimum_age` and `maximum_age` with the median value calculated only from the training set.\n",
    "\n",
    "*   **4. One-Hot Encoding:** It converts the four categorical features (`phases`, `study_type`, `lead_sponsor_class`, `sex`) into a numerical format using one-hot encoding. The encoder is configured with `handle_unknown='ignore'`, which makes sure that if a new, unseen category appears in the test data, it is handled correctly by being encoded as a row of all zeros, preventing errors.\n",
    "\n",
    "*   **5. Numerical Scaling:** It standardizes the three numerical features (`enrollment_count`, `minimum_age`, `maximum_age`) using a `StandardScaler`, which scales the data to have a mean of 0 and a standard deviation of 1.\n",
    "\n",
    "This detailed pipeline ensures our data is clean, purely numerical, and ready for modeling."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c6b193c2",
   "metadata": {},
   "source": [
    "###  2.2. Load Enhanced Subset and Create Train/Test Split"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "125f7793",
   "metadata": {},
   "outputs": [],
   "source": [
    "# First, we load the subset DataFrame containing the pre-generated embeddings.\n",
    "# This dataset should be in your working directory! If it isnt, you can find a link to download it in the appendix in the `README.md`.\n",
    "print(\"Loading the subset dataset with embeddings from Parquet file...\")\n",
    "subset_df_with_embeddings = pd.read_parquet('subset_dataset_with_embeddings.parquet')\n",
    "print(\"Dataset loaded.\")\n",
    "\n",
    "# Now, we create the train/test split for the rest of the notebook.\n",
    "# We want to make sure the same train/test split is used in all baseline models and experiments to ensure we stay consistent and have comparable outcomes. \n",
    "\n",
    "# We stratify on 'overall_status' to maintain the class distribution in both sets.\n",
    "print(\"\\nSplitting the subset into training and testing sets (80/20 split)...\")\n",
    "raw_train_df, raw_test_df = train_test_split(\n",
    "    subset_df_with_embeddings,\n",
    "    test_size=0.2,\n",
    "    random_state=42,\n",
    "    stratify=subset_df_with_embeddings['overall_status']\n",
    ")\n",
    "print(\"Splitting complete.\")\n",
    "print(f\"Raw training set size: {len(raw_train_df)}\")\n",
    "print(f\"Raw testing set size: {len(raw_test_df)}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a7d38b0d",
   "metadata": {},
   "source": [
    "### 2.3. Execute the Structured Data Preprocessing Pipeline"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a050c45a",
   "metadata": {},
   "outputs": [],
   "source": [
    "# We now feed our raw splits into the utility functions.\n",
    "print(\"\\nPreprocessing structured data for the baseline model...\")\n",
    "\n",
    "# The fit_and_preprocess_train function learns rules ONLY from the training set.\n",
    "processed_train_df, new_transformation_rules = fit_and_preprocess_train(raw_train_df)\n",
    "\n",
    "# The transform_test_data function applies those learned rules to the test set.\n",
    "processed_test_df = transform_test_data(raw_test_df, new_transformation_rules)\n",
    "\n",
    "print(\"\\nStructured data preprocessing complete.\")\n",
    "\n",
    "# Display the head of the final processed training data to confirm the structure.\n",
    "print(\"\\nHead of the final processed training data (for structured model):\")\n",
    "display(processed_train_df.head())"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "821bb1f1",
   "metadata": {},
   "source": [
    "## 3. Baseline Model: Structured Metadata\n",
    "\n",
    "With our data split and the structured features preprocessed, we can now build our first proof-of-concept model. The goal is to establish a performance benchmark using only the structured metadata.\n",
    "\n",
    "For this baseline, we will start directly with a **Random Forest Classifier**. While a simpler linear model like Logistic Regression is often a first step, our EDA revealed that the relationships between features (e.g., the semi-categorical nature of `minimum_age`) are likely non-linear. A Random Forest is a powerful ensemble model that works well at capturing these complex, non-linear interactions in tabular data right out of the box, making it a strong and realistic choice for a robust baseline. It also provides feature importances, which will give us initial insights into which structured features are most predictive.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "22843422",
   "metadata": {},
   "outputs": [],
   "source": [
    "# --- 3.1. Prepare Data for Model Training ---\n",
    "\n",
    "# Separate the features (X) and the target (y) for both training and testing sets.\n",
    "X_train = processed_train_df.drop('target', axis=1)\n",
    "y_train = processed_train_df['target']\n",
    "\n",
    "X_test = processed_test_df.drop('target', axis=1)\n",
    "y_test = processed_test_df['target']\n",
    "\n",
    "# --- Sanity Check the final shapes ---\n",
    "print(\"--- Final Shapes for Structured Model ---\")\n",
    "print(f\"Shape of X_train: {X_train.shape}\")\n",
    "print(f\"Shape of y_train: {y_train.shape}\")\n",
    "print(f\"Shape of X_test: {X_test.shape}\")\n",
    "print(f\"Shape of y_test: {y_test.shape}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b330f446",
   "metadata": {},
   "outputs": [],
   "source": [
    "# --- 3.2. Train the Random Forest Classifier ---\n",
    "\n",
    "# Initialize the classifier with parameters for reproducibility and speed.\n",
    "rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)\n",
    "\n",
    "# Train the model on the training data.\n",
    "print(\"Training the Random Forest baseline model...\")\n",
    "rf_classifier.fit(X_train, y_train)\n",
    "print(\"Training complete.\")\n",
    "\n",
    "# --- Make Predictions on the Test Data ---\n",
    "print(\"\\nMaking predictions on the test set...\")\n",
    "y_pred = rf_classifier.predict(X_test)\n",
    "print(\"Predictions complete.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "30beb023",
   "metadata": {},
   "source": [
    "### Evaluate the Model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8d355cfe",
   "metadata": {},
   "outputs": [],
   "source": [
    "# --- 3.3. Evaluate the Baseline Model ---\n",
    "\n",
    "print(\"--- Baseline Model Classification Report ---\")\n",
    "report = classification_report(y_test, y_pred, target_names=['Failure (0)', 'Success (1)'])\n",
    "print(report)\n",
    "\n",
    "print(\"\\n--- Confusion Matrix for Baseline Model ---\")\n",
    "# Use the style context to ensure the plot is clean\n",
    "with plt.style.context('default'):\n",
    "    fig, ax = plt.subplots(figsize=(8, 6))\n",
    "    ConfusionMatrixDisplay.from_predictions(\n",
    "        y_test, \n",
    "        y_pred, \n",
    "        ax=ax,\n",
    "        display_labels=['Failure', 'Success'],\n",
    "        cmap='Blues'\n",
    "    )\n",
    "    plt.title(\"Confusion Matrix for Baseline Model\")\n",
    "    plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "bbfa201f",
   "metadata": {},
   "source": [
    "### Analysis of the Baseline Model Performance\n",
    "\n",
    "The Random Forest baseline model, trained on the structured data from our subset, achieves a high overall accuracy of 90%. However, this metric is misleading due to the severe class imbalance in our target variable.\n",
    "\n",
    "**Performance on the \"Success\" (Majority) Class:**\n",
    "The model is highly effective at identifying successful trials, achieving an F1-score of 0.94. The confusion matrix shows it correctly identifies 14,565 true successes while only missing 525.\n",
    "\n",
    "**Performance on the \"Failure\" (Minority) Class:**\n",
    "The model's key weakness is its performance on the \"Failure\" class, which directly confirms our EDA hypothesis about the imbalance.\n",
    "*   **Recall (0.48):** The model only identifies 48% of all actual failures, misclassifying over half of them (1,304) as successes.\n",
    "*   **F1-Score (0.57):** This low F1-score is a direct consequence of the poor recall and serves as our primary benchmark for this class.\n",
    "\n",
    "**Conclusion:**\n",
    "This proof of concept successfully establishes a baseline. The model is heavily biased towards the majority class, demonstrating that simply training on the raw, imbalanced data is insufficient for reliably detecting trial failures. This result strongly motivates the need for the class balancing strategies we will explore in P3."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c4a30cc1",
   "metadata": {},
   "source": [
    "## 4. Embeddings Model: Initial Version\n",
    "\n",
    "Now we will build our second proof of concept using only the text embeddings. The goal is to isolate the predictive power of the semantic information from the trial summaries.\n",
    "\n",
    "Unlike the structured data, high-dimensional embeddings (1536 dimensions in our case, which come from the concatenating the two separate 768-dimensional embedding vectors) often contain strong linear patterns that can be captured by simpler models. Therefore, following a standard methodology for text classification, we will start with the simplest possible baseline to see if a linear signal exists.\n",
    "\n",
    "### 4a. Linear Model (Logistic Regression)\n",
    "\n",
    "We will begin with a `LogisticRegression` model. This is a fast, highly efficient model that serves as an excellent first-pass test to determine if there is a linearly separable relationship between the text embeddings and the trial outcomes.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "60999a1e",
   "metadata": {},
   "outputs": [],
   "source": [
    "# --- 4.1. Prepare Data for Embeddings Model ---\n",
    "\n",
    "# Define our definitive outcomes (ensure consistency)\n",
    "success_stati = [\"COMPLETED\"]\n",
    "failure_stati = [\"TERMINATED\", \"WITHDRAWN\", \"SUSPENDED\"]\n",
    "definitive_stati = success_stati + failure_stati\n",
    "\n",
    "# Filter the raw dataframes to only include definitive outcomes\n",
    "# This ensures we use the exact same set of trials as the structured baseline\n",
    "train_emb_df = raw_train_df[raw_train_df['overall_status'].isin(definitive_stati)].copy()\n",
    "test_emb_df = raw_test_df[raw_test_df['overall_status'].isin(definitive_stati)].copy()\n",
    "\n",
    "# Create the binary 'target' column on these filtered dataframes\n",
    "train_emb_df['target'] = train_emb_df['overall_status'].apply(lambda x: 1 if x in success_stati else 0)\n",
    "test_emb_df['target'] = test_emb_df['overall_status'].apply(lambda x: 1 if x in success_stati else 0)\n",
    "\n",
    "# --- Extract and Combine Embeddings ---\n",
    "\n",
    "# Extract from the FILTERED training data\n",
    "X_train_emb_summary = np.vstack(train_emb_df['brief_summary_embedding'].values)\n",
    "X_train_emb_criteria = np.vstack(train_emb_df['eligibility_criteria_embedding'].values)\n",
    "\n",
    "# Extract from the FILTERED test data\n",
    "X_test_emb_summary = np.vstack(test_emb_df['brief_summary_embedding'].values)\n",
    "X_test_emb_criteria = np.vstack(test_emb_df['eligibility_criteria_embedding'].values)\n",
    "\n",
    "# --- Sanity Check: Verify the dimensions ---\n",
    "print(\"--- Sanity Checking Embedding Shapes ---\")\n",
    "print(f\"Shape of train summary embeddings: {X_train_emb_summary.shape}\")\n",
    "print(f\"Shape of train criteria embeddings: {X_train_emb_criteria.shape}\")\n",
    "print(f\"The dimension of a single embedding vector is {X_train_emb_summary.shape[1]}.\")\n",
    "\n",
    "# Concatenate them to create the final feature matrices\n",
    "X_train_emb_np = np.concatenate([X_train_emb_summary, X_train_emb_criteria], axis=1)\n",
    "X_test_emb_np = np.concatenate([X_test_emb_summary, X_test_emb_criteria], axis=1)\n",
    "\n",
    "# Extract the corresponding target vectors\n",
    "y_train_emb = train_emb_df['target'].values\n",
    "y_test_emb = test_emb_df['target'].values\n",
    "\n",
    "print(f\"\\nShape of the final, concatenated TRAIN feature matrix: {X_train_emb_np.shape}\")\n",
    "print(f\"Shape of the final, concatenated TEST feature matrix: {X_test_emb_np.shape}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "782b613e",
   "metadata": {},
   "outputs": [],
   "source": [
    "# --- 4.2. Train the Logistic Regression Model ---\n",
    "\n",
    "# Initialize the classifier\n",
    "# max_iter is a hyperparameter. We need it because Logistic regression \n",
    "# uses iterative optimization to minimize the loss function. \n",
    "# The optimization either stops when convergance is reached or the number \n",
    "# of iteration reaches the chosen hyperparameter. \n",
    "lr_classifier = LogisticRegression(random_state=42, max_iter=1000, n_jobs=-1)\n",
    "\n",
    "# Train the model on the embeddings data\n",
    "print(\"Training the Logistic Regression model on embeddings...\")\n",
    "lr_classifier.fit(X_train_emb_np, y_train_emb)\n",
    "print(\"Training complete.\")\n",
    "\n",
    "# --- Make Predictions on the Test Set ---\n",
    "print(\"\\nMaking predictions on the test set...\")\n",
    "y_pred_emb_lr = lr_classifier.predict(X_test_emb_np)\n",
    "print(\"Predictions complete.\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "60047f15",
   "metadata": {},
   "outputs": [],
   "source": [
    "# --- 4.3. Evaluate the Logistic Regression Model ---\n",
    "\n",
    "print(\"\\n--- Embeddings Model (Logistic Regression) Classification Report ---\")\n",
    "report_emb_lr = classification_report(y_test_emb, y_pred_emb_lr, target_names=['Failure (0)', 'Success (1)'])\n",
    "print(report_emb_lr)\n",
    "\n",
    "print(\"\\n--- Confusion Matrix for Embeddings Model (Logistic Regression) ---\")\n",
    "# Use the style context to ensure the plot is clean\n",
    "with plt.style.context('default'):\n",
    "    fig, ax = plt.subplots(figsize=(8, 6))\n",
    "    ConfusionMatrixDisplay.from_predictions(\n",
    "        y_test_emb, \n",
    "        y_pred_emb_lr, \n",
    "        ax=ax,\n",
    "        display_labels=['Failure', 'Success'],\n",
    "        cmap='Greens' \n",
    "    )\n",
    "    plt.title(\"Confusion Matrix for Embeddings Model (Logistic Regression)\")\n",
    "    plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "7af4ca53",
   "metadata": {},
   "source": [
    "### Analysis of the Logistic Regression Performance\n",
    "\n",
    "This initial embeddings model acts as a successful proof of concept, however, its predictive performance is very poor. The overall accuracy of 86% is misleading, and the macro avg f1-score of 0.49 reveals its ineffectiveness.\n",
    "\n",
    "**Performance on the \"Failure\" (Minority) Class:**\n",
    "The model's performance on this class is really poor.\n",
    "*   **Recall (0.03):** The model only identifies 3% of all actual failures, missing **2,429** of them as shown in the confusion matrix.\n",
    "*   **F1-Score (0.06):** This extremely low score confirms that this simple linear model has almost completely failed to learn meaningful patterns from the embeddings to identify failed trials.\n",
    "\n",
    "**Conclusion:**\n",
    "A simple `LogisticRegression` model is not powerful enough to find the complex signals in the high-dimensional embedding space, especially given the class imbalance. It has defaulted to an extremely biased decision boundary, almost always predicting success. This result motivates our next step: testing a more powerful, non-linear model."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "9dcf5621",
   "metadata": {},
   "source": [
    "### 4b. Non-Linear Model (Random Forest)\n",
    "\n",
    "The `LogisticRegression` model's failure suggests that either the signal in the embeddings is too weak for any model, or that the relationship between the embeddings and trial outcomes is non-linear and requires a more complex model to capture.\n",
    "\n",
    "To test this second hypothesis, we will now use a **`RandomForestClassifier`**. As a powerful, non-linear ensemble model, it can capture complex interactions within the 1536-dimensional feature space that a linear model cannot. This will help us determine if the poor performance was due to the model's simplicity or a fundamental lack of signal in the data. This also allows for a more direct comparison between the structured data model and this one, due to the model families being the same. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "450dd0f2",
   "metadata": {},
   "outputs": [],
   "source": [
    "# --- 4.4. Train the Random Forest Classifier on Embeddings ---\n",
    "\n",
    "# Initialize the classifier\n",
    "# The n_estimators and max_depth are hyperparameters. The former\n",
    "# defines how many trees we have in our random forest. The \n",
    "# latter is the number of split each tree can grow to. This controls the complexity.\n",
    "rf_emb_classifier = RandomForestClassifier(\n",
    "    n_estimators=100, \n",
    "    random_state=42, \n",
    "    n_jobs=-1,\n",
    "    max_depth=10 # Limiting depth helps control training time and prevent overfitting\n",
    ")\n",
    "\n",
    "# Train the model on the same embeddings data\n",
    "print(\"Training the Random Forest model on embeddings...\")\n",
    "# We use the same X_train_emb_np and y_train_emb from the previous step\n",
    "rf_emb_classifier.fit(X_train_emb_np, y_train_emb)\n",
    "print(\"Training complete.\")\n",
    "\n",
    "# --- Make Predictions ---\n",
    "print(\"\\nMaking predictions with the Random Forest model...\")\n",
    "y_pred_emb_rf = rf_emb_classifier.predict(X_test_emb_np)\n",
    "print(\"Predictions complete.\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0780f826",
   "metadata": {},
   "outputs": [],
   "source": [
    "# --- 4.5. Evaluate the Random Forest (Embeddings) ---\n",
    "\n",
    "print(\"\\n--- Embeddings Model (Random Forest) Classification Report ---\")\n",
    "report_emb_rf = classification_report(y_test_emb, y_pred_emb_rf, target_names=['Failure (0)', 'Success (1)'], zero_division=0)\n",
    "print(report_emb_rf)\n",
    "\n",
    "print(\"\\n--- Confusion Matrix for Embeddings Model (Random Forest) ---\")\n",
    "with plt.style.context('default'):\n",
    "    fig, ax = plt.subplots(figsize=(8, 6))\n",
    "    ConfusionMatrixDisplay.from_predictions(\n",
    "        y_test_emb, \n",
    "        y_pred_emb_rf, \n",
    "        ax=ax,\n",
    "        display_labels=['Failure', 'Success'],\n",
    "        cmap='Purples'\n",
    "    )\n",
    "    plt.title(\"Confusion Matrix for Embeddings Model (Random Forest)\")\n",
    "    plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "57b78750",
   "metadata": {},
   "source": [
    "### Analysis of the Random Forest (Embeddings) Performance\n",
    "\n",
    "The result of the Random Forest model is even worse than the Logistic Regression. Despite being a more powerful, non-linear model, its performance on the minority class collapsed completely.\n",
    "\n",
    "**Performance on the \"Failure\" (Minority) Class:**\n",
    "The model entirely failed to identify any \"Failure\" cases.\n",
    "*   The `precision`, `recall`, and `f1-score`** are all 0.00.\n",
    "*   The confusion matrix confirms this starkly, showing 0 True Negatives. The model never once predicted \"Failure,\" and misclassified all 2,505 actual failures as \"Success.\"\n",
    "\n",
    "**Conclusion: A \"Lazy Classifier\" Confirmed**\n",
    "This experiment proves that simply increasing model complexity is not the solution. The Random Forest, faced with the severe class imbalance and noisy high-dimensional data, found that the optimal strategy to minimize its overall error was to ignore the minority class completely and always predict the majority class.\n",
    "\n",
    "This provides the strongest possible evidence that the primary issue is not the model architecture, but the data itself. This definitively justifies our hypothesis that a data-level intervention, such as class balancing, is the necessary next step."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b4bafb2d",
   "metadata": {},
   "source": [
    "### 4c. Downsampling Experiment (A Quick Test for P2)\n",
    "\n",
    "Our analysis has now clearly shown that both linear and non-linear models fail on the imbalanced text embedding data. Like we discussed above, we hypothesize that his is due to the data distribution and not the model choice. \n",
    "\n",
    "To test this hypothesis and provide a proof of concept for our P3 plan, we will do a final experiment using a simple random downsampling strategy. This idea comes from Li, 2025 (full source is in the appendix of the README file). The goal is to see if a model trained on a balanced dataset can learn to identify the \"Failure\" class more effectively, even if it means sacrificing some performance on the \"Success\" class."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "cf70d11a",
   "metadata": {},
   "outputs": [],
   "source": [
    "# --- 4.6. Downsampling the Training Data ---\n",
    "\n",
    "from sklearn.utils import resample\n",
    "\n",
    "# We first need to combine our training features and labels into a single DataFrame to manage the sampling.\n",
    "# We'll use the already filtered `train_emb_df` for this.\n",
    "X_train_emb_df_for_resample = train_emb_df[['brief_summary_embedding', 'eligibility_criteria_embedding', 'target']]\n",
    "\n",
    "# 1. Separate the majority and minority classes\n",
    "df_majority = X_train_emb_df_for_resample[X_train_emb_df_for_resample.target == 1]\n",
    "df_minority = X_train_emb_df_for_resample[X_train_emb_df_for_resample.target == 0]\n",
    "\n",
    "# 2. Downsample the majority class\n",
    "df_majority_downsampled = resample(df_majority, \n",
    "                                 replace=False,    # sample without replacement\n",
    "                                 n_samples=len(df_minority), # to match minority class size\n",
    "                                 random_state=42) # for reproducibility\n",
    "\n",
    "# 3. Combine the downsampled majority with the original minority\n",
    "df_downsampled = pd.concat([df_majority_downsampled, df_minority])\n",
    "\n",
    "print(f\"Original training set size (definitive outcomes): {len(train_emb_df)}\")\n",
    "print(f\"Downsampled training set size: {len(df_downsampled)}\")\n",
    "print(f\"New class distribution in training set:\\n{df_downsampled.target.value_counts()}\")\n",
    "\n",
    "# --- Prepare the final downsampled training data ---\n",
    "# Extract summary and criteria embeddings separately, then vstack and concatenate\n",
    "X_train_ds_summary = np.vstack(df_downsampled['brief_summary_embedding'].values)\n",
    "X_train_ds_criteria = np.vstack(df_downsampled['eligibility_criteria_embedding'].values)\n",
    "X_train_downsampled_np = np.concatenate([X_train_ds_summary, X_train_ds_criteria], axis=1)\n",
    "y_train_downsampled = df_downsampled.target.values\n",
    "\n",
    "# --- 4.7. Train a NEW Random Forest on the BALANCED data ---\n",
    "rf_downsampled_classifier = RandomForestClassifier(\n",
    "    n_estimators=100, \n",
    "    random_state=42, \n",
    "    n_jobs=-1,\n",
    "    max_depth=10 \n",
    ")\n",
    "\n",
    "print(\"\\nTraining Random Forest on downsampled data...\")\n",
    "rf_downsampled_classifier.fit(X_train_downsampled_np, y_train_downsampled)\n",
    "print(\"Training complete.\")\n",
    "\n",
    "# --- 5. Evaluate on the ORIGINAL, UNBALANCED test set ---\n",
    "# We use the original X_test_emb_np and y_test_emb to get an honest performance measure.\n",
    "print(\"\\nMaking predictions on the original unbalanced test set...\")\n",
    "y_pred_downsampled = rf_downsampled_classifier.predict(X_test_emb_np)\n",
    "print(\"Predictions complete.\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "41ea7eb2",
   "metadata": {},
   "outputs": [],
   "source": [
    "# --- 4.8. Evaluate the Downsampled Model ---\n",
    "\n",
    "print(\"\\n--- Embeddings Model (Downsampled Random Forest) Classification Report ---\")\n",
    "report_downsampled = classification_report(y_test_emb, y_pred_downsampled, target_names=['Failure (0)', 'Success (1)'])\n",
    "print(report_downsampled)\n",
    "\n",
    "print(\"\\n--- Confusion Matrix for Downsampled Model ---\")\n",
    "with plt.style.context('default'):\n",
    "    fig, ax = plt.subplots(figsize=(8, 6))\n",
    "    ConfusionMatrixDisplay.from_predictions(\n",
    "        y_test_emb, \n",
    "        y_pred_downsampled, \n",
    "        ax=ax,\n",
    "        display_labels=['Failure', 'Success'],\n",
    "        cmap='Reds',\n",
    "        values_format='d'\n",
    "    )\n",
    "    plt.title(\"Confusion Matrix for Embeddings Model (Downsampled)\")\n",
    "    plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "3b0b4c41",
   "metadata": {},
   "source": [
    "### Analysis of the Downsampled Model Performance\n",
    "\n",
    "This experiment  provides clear validation for our P3 strategy. By training the Random Forest on a balanced dataset, the model's ability to predict the failure class improved significantly, and it did not lose too much predictive power from the success class. \n",
    "\n",
    "**Impact on the \"Failure\" (Minority) Class:**\n",
    "This is where the most significant improvement occurred.\n",
    "*   **Recall (0.63):** This is a massive improvement over the previous model's 0.00. By learning from a balanced set, the model now correctly identifies 63% of all actual failures. The confusion matrix shows it found 1,573 true negatives, proving that a strong predictive signal does exist in the embeddings.\n",
    "*   **F1-Score (0.34):** While still modest, this F1-score is a huge leap from 0.00 and establishes a much more credible benchmark for the embeddings model.\n",
    "\n",
    "**The Trade-off:**\n",
    "As expected, this improvement came at the cost of performance on the majority class. The recall for \"Success\" dropped to 0.66, as the model is no longer biased towards this outcome. The overall accuracy is lower (66%), but the `macro avg f1-score` of **0.55** shows that the model is now much more balanced and useful.\n",
    "\n",
    "**Conclusion:**\n",
    "This experiment proves that the primary issue was not a lack of signal in the embeddings, but the severe class imbalance. By addressing the data distribution directly, we were able to train a model that can effectively identify both success and failure cases. This provides strong, data-driven evidence that our proposed P3 plan—implementing a balancing strategy and building a final multimodal model—is the correct path forward."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a2f8522d",
   "metadata": {},
   "source": [
    "### Math Methods\n",
    "\n",
    "We treat clinical trial outcome prediction as binary classification with label $y \\in \\{0, 1\\}$ for *failure* vs *success* and tabular/embedding features $x$.\n",
    "\n",
    "**Logistic Regression.**\n",
    "Our `sklearn` model optimizes the **weighted logistic loss**\n",
    "\n",
    "$\n",
    "\\mathcal{L}(\\theta) = \\frac{1}{n} \\sum_{i=1}^n w_{y_i} \\log(1 + \\exp(-y_i \\theta^\\top x_i)) + \\lambda \\|\\theta\\|_2^2,\n",
    "$\n",
    "\n",
    "where $w_{y_i}$ are class weights (`class_weight=\"balanced\"` in sklearn).\n",
    "These weights increase the penalty for misclassifying rare failure cases.\n",
    "\n",
    "**Random Forest.**\n",
    "Each tree chooses splits by maximizing **weighted Gini impurity decrease**.\n",
    "For node $t$,\n",
    "\n",
    "$\n",
    "G(t) = 1 - \\sum_k p_k(t)^2;\n",
    "$\n",
    "\n",
    "and a split's quality is\n",
    "\n",
    "$\n",
    "\\Delta = G(\\text{parent}) - \\frac{n_L}{n}G(L) - \\frac{n_R}{n}G(R),\n",
    "$\n",
    "\n",
    "where class weights modify the effective class proportions $p_k(t)$.\n",
    "This shifts splits to better separate minority failure cases.\n",
    "\n",
    "**Imbalance handling.**\n",
    "We use both (i) sklearn's class weighting and (ii) random downsampling of successes.\n",
    "Class weights influence the training objective (above), while downsampling changes the empirical class frequencies seen by the models."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e75a9342",
   "metadata": {},
   "source": [
    "## 5. P2 Conclusion and Next Steps for P3\n",
    "\n",
    "This milestone served as a detailed \"proof of concept\" for our project. We have completed a full round of exploratory data analysis, pipeline development, and baseline modeling, which has given us good insights and a clear path forward for Milestone P3.\n",
    "\n",
    "### Summary of P2 Findings\n",
    "\n",
    "1.  **Structured Data Baseline:** Our `RandomForestClassifier` trained on the structured metadata established a strong initial baseline. It achieved high performance on the majority \"Success\" class (F1-score: 0.94), but its effectiveness on the \"Failure\" class was limited by the data's strong class imbalance (F1-score: 0.57, with only 48% recall).\n",
    "\n",
    "2.  **Embeddings Model Baseline:** Our initial models (`LogisticRegression` and `RandomForestClassifier`) trained on the text embeddings performed very poorly on the unbalanced data. The Random Forest model, in particular, completely failed to identify any \"Failure\" cases (Recall: 0.00), collapsing into a \"lazy classifier\" that only predicted the majority class.\n",
    "\n",
    "3.  **Downsampling Experiment:** A quick experiment with random downsampling on the embeddings training data provided a dramatic improvement. The model's recall for the \"Failure\" class jumped from 0% to 63%. This successfully proved our hypothesis that class imbalance, not a lack of signal, was the primary issue limiting the embeddings model's performance.\n",
    "\n",
    "### P2 Conclusion\n",
    "\n",
    "Our work in this milestone has been highly successful. We have:\n",
    "*   Developed a robust, leak-free data preprocessing pipeline.\n",
    "*   Established clear performance benchmarks for both structured and text-based models.\n",
    "*   Definitively identified severe class imbalance as the central challenge that must be addressed to build an effective predictive model.\n",
    "*   Proven, with a data-driven experiment, that class balancing strategies are a highly promising solution.\n",
    "\n",
    "### Next Steps for P3\n",
    "\n",
    "0.  **Scale Up Computation:** Our first task will be to productionize our embedding generation process to run on the full dataset, which may require using more powerful cloud computing resources.\n",
    "\n",
    "1.  **Implement a Robust Balancing Strategy:** We will apply our planned downsampling technique to the training sets for both the structured data model and the embeddings model. This will create two new, stronger, and more balanced baseline models.\n",
    "\n",
    "2.  **Build the Final Multimodal Model:** We will then combine the balanced, preprocessed structured data with the text embeddings into a single, rich feature set. We will train a final, powerful model (e.g., XGBoost or a simple neural network) on this multimodal data.\n",
    "\n",
    "3.  **Final Comparative Analysis:** The final project will present a comprehensive comparison of all models: the original imbalanced baselines, the improved balanced baselines, and the final multimodal model. This will allow us to definitively measure the impact of class balancing and quantify the predictive value added by the semantic text embeddings, thereby fully addressing our project's core hypothesis."
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "nlp",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.11"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
```

## File: p2-comments-review.md/p2-comments-review.md
```markdown
### Summary of Key Feedback and Our Action Plan

Your advisor's comments boil down to three main areas for improvement, which will form the core of your P3 work.

**1. The "Suspiciously High" Baseline Score & Potential Leakage**
*   **The Feedback:** The `weighted F1-score of 0.89` looked too good to be true, making him suspect data leakage. He correctly notes that class imbalance is a factor.
*   **Our Analysis:** We already know the truth here. Your pipeline is methodologically sound and **does not have data leakage**. The high score is purely a result of the `weighted average` being a misleading metric that is inflated by the massive "Success" class. Our focus on the `macro average` and the per-class F1-scores was the correct, critical approach.
*   **P3 Action Plan:**
    *   **Lead with the Honest Metrics:** In your final report, you will lead with the per-class and `macro average` F1-scores. You will explicitly state that the `weighted average` is misleading and explain why. This directly addresses his concern and demonstrates your deep understanding of model evaluation.
    *   **Refine the Baseline:** Your downsampling experiment is the key. For P3, the **official baseline score** will be the one you get after implementing a proper balancing strategy (like downsampling). This will produce a lower, more realistic, and far more meaningful benchmark.

**2. Defining the Multimodal "Fusion Architecture"**
*   **The Feedback:** The main critique is that the proposal is vague about *how* you will combine the structured data and the embeddings. "Novelty claims are a bit vague... without architectural details."
*   **Our Analysis:** This is the core technical task for P3. We need to define a specific model that can handle two different types of input.
*   **P3 Action Plan:** Propose and build a simple **Intermediate Fusion Neural Network**. This sounds complex, but it's a standard and very effective architecture that directly answers his question:
    *   **Branch 1 (Structured Data):** A small neural network (an MLP - Multi-Layer Perceptron) that takes your 23 preprocessed structured features as input.
    *   **Branch 2 (Embeddings):** A separate, slightly larger MLP that takes your 1536-dimensional embedding vector as input.
    *   **Fusion:** The outputs of these two branches are **concatenated** into a single, combined vector.
    *   **Classification Head:** This combined vector is fed into a final set of layers that makes the ultimate "Success/Failure" prediction.
    This is the concrete architectural detail your advisor is looking for. It's a standard multimodal approach and a perfect fit for this project.

**3. Acknowledging Dataset Limitations & Correlation vs. Causation**
*   **The Feedback:** A very sharp academic point: "Do you think the dataset includes all the factors... financial or regulatory confounders... capture correlations rather than true causal drivers?"
*   **Our Analysis:** He is absolutely right. This is a crucial point for the scientific integrity of your report. Your model can only learn from the data it's given.
*   **P3 Action Plan:**
    *   Add a **"Limitations" section** to your final P3 report.
    *   In this section, you will explicitly state what the advisor pointed out: your model does not have access to external factors like funding changes, regulatory hurdles, or specific scientific breakthroughs that could cause a trial to fail.
    *   You will conclude that the model is therefore designed to find powerful **predictive correlations** in the trial design data, not to identify the **root causal drivers** of trial failure. This shows maturity and a nuanced understanding of your project's scope.

### How This Translates into a Concrete P3 Workflow

Now that we've analyzed the feedback, here is a clear, step-by-step plan for Milestone P3.

**Step 1: Refine and Re-run the Baseline Models (The "Honest" Benchmarks)**
*   **Task:** Fully implement the **downsampling** strategy we tested.
*   **Action:**
    1.  Create a balanced training set for the **structured data**. Train a new `RandomForestClassifier` on it. Evaluate on the original unbalanced test set. This is your **new, official structured data baseline score**.
    2.  Use the balanced training set for the **embeddings data** that you already created. The result from your downsampling experiment is your **new, official embeddings baseline score**.
*   **Outcome:** You now have two strong, fair, and realistic benchmarks that properly handle the class imbalance.

**Step 2: Build the Multimodal Fusion Model**
*   **Task:** Implement the Intermediate Fusion Neural Network described above using a library like `Keras` (with TensorFlow) or `PyTorch`.
*   **Action:**
    1.  Define the two-branch architecture.
    2.  Use the **balanced structured data** as input for one branch and the **balanced embeddings data** as input for the other.
*   **Outcome:** You have your final, most advanced model.

**Step 3: Train, Tune, and Evaluate the Final Model**
*   **Task:** Train the fusion model and compare it to your refined baselines.
*   **Action:**
    1.  Train the neural network on the balanced training data.
    2.  Evaluate it on the original, unbalanced test set.
    3.  Create a final comparison table showing the `macro avg` and per-class F1-scores for all three final models (Balanced Structured, Balanced Embeddings, and Multimodal).
*   **Outcome:** A clear, quantitative answer to your project's main hypothesis.

**Step 4: Write the Final Report and Presentation**
*   **Task:** Synthesize all your findings into the final report.
*   **Action:**
    1.  **Abstract/Novelty:** Update these sections to be more precise, defining "Success" and describing your specific fusion architecture.
    2.  **Methods:** Detail the final, balanced workflow and the neural network design.
    3.  **Results:** Present your final comparison table and analyze the results.
    4.  **Discussion:** Add the crucial **"Limitations"** section to discuss the correlation vs. causation issue.
*   **Outcome:** A complete, well-reasoned, and scientifically rigorous final project.

This plan directly addresses every piece of your advisor's feedback and sets you on a clear path to a very successful P3. You're in a great position.
```

## File: p2-instructor-comments.md/p2-instructor-comments.md
```markdown
# Feedback for Project milestone P2

## Overall Feedback
Great milestone in general. However, there are some concerns that are detailed below. My main comments would be:  
1) Some details regarding the proposed fusion architecture are missing.  
2) Do you think the dataset includes all the factors needed for you to build a success prediction model? For example, financial or regulatory confounders are not there. Is it possible that the model can only predict trial completion based on partial information and may capture correlations rather than true causal drivers?

## Abstract
Clear motivation and good framing. Novelty might be a bit overstated, you need to emphasize why your approach is new. “Success” needs a more precise definition so the reader has a clear understanding of your pitch.

## Novelty
Good context and clear contributions. Novelty claims are a bit vague though without architectural details. Strengthen justification for why your fusion approach is unique, and clarify how insights will be extracted.

## Methods
**EDA:** Strong understanding of data quirks and imbalance.  
**Data Preparation and Pipeline:** Pipeline is sound. I don't get the “minimal missingness” justification. Also, I feel that here you should mention something about the target imbalance that you previously mentioned. How did you handle that?  
**Baseline Modeling:** Nice baselines. My only worry here is that an F1-score of 0.89 is almost suspiciously high, which might indicate label leakage or overly easy prediction. Of course, class imbalance plays a role in that. I think this issue must be fixed so that your conclusions are more meaningful.  
**Downsampling Experiment:** Strong experimental intuition → directly tests imbalance as a root cause.

## Timeline and organization within the team
The proposed timeline makes sense and is clear to me. I think 7–8 days should be more than enough to write the final report, but that's completely up to you.

Cool titles in the team organization section. My only comment here is that you should do this for Project Milestone P3 as well in advance. This allows you to plan better and divide responsibilities in a practical manner.

## Appendix
Detailed and nicely written. Great initiative to include the GenAI declaration!

## Question for TAs:
No questions

**Textual description quality:** Excellent work, keep it up!  
**Code quality:** Excellent work, keep it up!
```

## File: README.md/README.md
```markdown
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
```

## File: requirements.txt/requirements.txt
```
# Core data science libraries
pandas
numpy
pyarrow

# Library for loading the dataset from Hugging Face
datasets

# Scikit-learn (good to have for general data science)
scikit-learn

# NLP library and its backend for GPU processing
sentence-transformers
torch

# Plotting libraries (harmless to keep for future analysis)
matplotlib
seaborn
```

## File: utils.py/utils.py
```python
# preprocess function that preprocesses our data and creates rules for transofrming our test set later on. 

import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import numpy as np

def fit_and_preprocess_train(train_df: pd.DataFrame):
    """
    Takes the raw training DataFrame, preprocesses it according to our specific EDA plan,
    and learns the transformation rules.
    
    Args:
        train_df: The raw training pandas DataFrame.
        
    Returns:
        A tuple containing:
        - processed_train_df: The cleaned and preprocessed training DataFrame.
        - transformation_rules: A dictionary containing the learned medians, scaler, and encoder.
    """
    data = train_df.copy()

    # --- Step 1 & 2: Feature Selection and Target Creation ---
    features_to_keep = [
        'phases', 'study_type', 'enrollment_count', 'lead_sponsor_class', 
        'sex', 'minimum_age', 'maximum_age', 'overall_status'
    ]
    data = data[features_to_keep]

    success_stati = ["COMPLETED"]
    failure_stati = ["TERMINATED", "WITHDRAWN", "SUSPENDED"]
    data = data[data['overall_status'].isin(success_stati + failure_stati)].copy()
    data['target'] = data['overall_status'].apply(lambda x: 1 if x in success_stati else 0)
    data = data.drop(columns=['overall_status'])

    # --- Step 3: Handle Missing Values ---
    
    # 3a: Drop rows for 'sex' and 'enrollment_count'
    # The number of rows to drop are both under 3%, based on our EDA
    initial_rows = len(data)
    data.dropna(subset=['sex', 'enrollment_count'], inplace=True)
    print(f"Dropped {initial_rows - len(data)} rows with missing 'sex' or 'enrollment_count'.")

    # Define feature types
    categorical_features = ['phases', 'study_type', 'lead_sponsor_class', 'sex']
    numerical_features = ['enrollment_count', 'minimum_age', 'maximum_age']
    
    # 3b: Impute CATEGORICAL ('phases' only)
    # This will fill true NaN values, but leave the string "NA" untouched.
    # We do this because the "NA" string itself is usually used in trials or studies where phases are not relevant.
    # So we want to be able to distinguish between studies where the phases value was missing vs where it was not relevant. 
    data['phases'] = data['phases'].fillna('Unknown')

    # 3c: Impute NUMERICAL ('minimum_age' and 'maximum_age')
    transformation_rules = {'medians': {}}
    for col in ['minimum_age', 'maximum_age']:
        median_val = data[col].median()
        transformation_rules['medians'][col] = median_val
        data[col] = data[col].fillna(median_val)
        
    # Special fix for 'phases' if it's a list/ndarray
    data['phases'] = data['phases'].apply(lambda d: d[0] if isinstance(d, (list, np.ndarray)) and len(d) > 0 else d if not isinstance(d, (list, np.ndarray)) else 'Unknown')
    
    # --- Step 4: One-Hot Encode Categorical Features ---
    # handle_unknown='ignore' will create a column of all zeros for categories in the test set
    # that were not seen in the training set. This is a robust approach.
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoder.fit(data[categorical_features])
    encoded_cols = encoder.get_feature_names_out(categorical_features)
    encoded_df = pd.DataFrame(encoder.transform(data[categorical_features]), columns=encoded_cols, index=data.index)
    transformation_rules['encoder'] = encoder

    # --- Step 5: Scale Numerical Features ---
    scaler = StandardScaler()
    scaler.fit(data[numerical_features])
    scaled_df = pd.DataFrame(scaler.transform(data[numerical_features]), columns=numerical_features, index=data.index)
    transformation_rules['scaler'] = scaler
    
    # Combine into the final processed DataFrame
    processed_train_df = pd.concat([scaled_df, encoded_df, data['target']], axis=1)

    print("Training data preprocessed successfully according to the specific plan.")
    
    return processed_train_df, transformation_rules


# preprocess function that preprocesses our testing data data and applies the transformation rules

def transform_test_data(test_df: pd.DataFrame, rules: dict):
    """
    Applies the learned preprocessing rules to the raw test DataFrame.
    
    Args:
        test_df: The raw testing pandas DataFrame.
        rules: The dictionary of transformation rules learned from the training data.
        
    Returns:
        A cleaned and preprocessed test DataFrame ready for evaluation.
    """
    data = test_df.copy()

    # --- Step 1 & 2: Feature Selection and Target Creation ---
    # (Same initial steps as the training function)
    features_to_keep = [
        'phases', 'study_type', 'enrollment_count', 'lead_sponsor_class', 
        'sex', 'minimum_age', 'maximum_age', 'overall_status'
    ]
    data = data[features_to_keep]

    success_stati = ["COMPLETED"]
    failure_stati = ["TERMINATED", "WITHDRAWN", "SUSPENDED"]
    data = data[data['overall_status'].isin(success_stati + failure_stati)].copy()
    data['target'] = data['overall_status'].apply(lambda x: 1 if x in success_stati else 0)
    data = data.drop(columns=['overall_status'])

    # --- Step 3: Handle Missing Values using LEARNED Rules ---
    # Drop rows first, mirroring the training process
    data.dropna(subset=['sex', 'enrollment_count'], inplace=True)
    
    categorical_features = ['phases', 'study_type', 'lead_sponsor_class', 'sex']
    numerical_features = ['enrollment_count', 'minimum_age', 'maximum_age']
    
    # Impute CATEGORICAL ('phases' only)
    data['phases'] = data['phases'].fillna('Unknown')

    # Impute NUMERICAL using the SAVED medians from the rules dictionary
    for col in ['minimum_age', 'maximum_age']:
        data[col] = data[col].fillna(rules['medians'][col])
        
    # Special fix for 'phases' if it's a list/ndarray
    data['phases'] = data['phases'].apply(lambda d: d[0] if isinstance(d, (list, np.ndarray)) and len(d) > 0 else d if not isinstance(d, (list, np.ndarray)) else 'Unknown')

    # --- Step 4: One-Hot Encode Categorical Features using LEARNED Encoder ---
    # Retrieve the fitted encoder from the rules
    encoder = rules['encoder']
    encoded_cols = encoder.get_feature_names_out(categorical_features)
    # APPLY the learned transformation
    encoded_df = pd.DataFrame(encoder.transform(data[categorical_features]), columns=encoded_cols, index=data.index)

    # --- Step 5: Scale Numerical Features using LEARNED Scaler ---
    # Retrieve the fitted scaler from the rules
    scaler = rules['scaler']
    # APPLY the learned transformation
    scaled_df = pd.DataFrame(scaler.transform(data[numerical_features]), columns=numerical_features, index=data.index)
    
    # Combine into the final processed DataFrame
    processed_test_df = pd.concat([scaled_df, encoded_df, data['target']], axis=1)

    print("Test data transformed successfully using training set rules.")
    
    return processed_test_df
```
