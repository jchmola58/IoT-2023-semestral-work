# IoT-2023: Detection of Malicious IoT Network Traffic

## Project Overview

This project focuses on the analysis and classification of IoT network traffic using the **IoT-2023 dataset**.

The main goal is to investigate whether network traffic features can be used to distinguish between:

* **BENIGN** traffic
* **MALICIOUS** traffic

The project will combine exploratory data analysis (EDA), statistical hypothesis testing, machine learning, model evaluation, and feature importance analysis.

An important part of the project will also be investigating **data leakage and generalization**, especially in the context of network traffic datasets.

---

# Dataset

The original dataset contains approximately **45 million network traffic records** distributed across 63 CSV files.

For this project, approximately **5% of each CSV file** is sampled, resulting in approximately:

* **2.25 million observations**
* **39 features**
* **1 target variable (`Label`)**
* **34 original traffic classes**

The original labels include classes such as:

* BENIGN
* DDOS
* DOS
* MIRAI
* RECON
* VULNERABILITYSCAN
* DNS_SPOOFING
* MITM-ARPSPOOFING
* SQLINJECTION
* XSS
* COMMANDINJECTION
* BACKDOOR_MALWARE
* and others

For the main classification task, the original labels will eventually be aggregated into:

```text
BENIGN
MALICIOUS
```

The original multiclass `Label` will be preserved for exploratory analysis.

---

# Features

The dataset contains several groups of network traffic features.

## Network and protocol features

* `Header_Length`
* `Protocol Type`
* `TCP`
* `UDP`
* `ICMP`
* `IGMP`
* `IPv`
* `LLC`

## Application protocol features

* `HTTP`
* `HTTPS`
* `DNS`
* `Telnet`
* `SMTP`
* `SSH`
* `IRC`
* `DHCP`
* `ARP`

## TCP flag features

* `fin_flag_number`
* `syn_flag_number`
* `rst_flag_number`
* `psh_flag_number`
* `ack_flag_number`
* `ece_flag_number`
* `cwr_flag_number`

## Packet/traffic statistics

* `Tot sum`
* `Min`
* `Max`
* `AVG`
* `Std`
* `Tot size`
* `Variance`

## Timing and traffic features

* `Rate`
* `IAT`
* `Number`

## Count features

* `ack_count`
* `syn_count`
* `fin_count`
* `rst_count`

---

# Project Goals

The project will investigate the following main questions:

1. Can network traffic features distinguish between benign and malicious IoT traffic?
2. Which network characteristics are most strongly associated with malicious traffic?
3. How do different machine learning models perform on this classification task?
4. How does the choice of train/test splitting strategy affect the estimated model performance?
5. Can the model generalize to traffic originating from different source files/scenarios?

---

# Exploratory Data Analysis

The first stage of the project will focus on understanding the dataset before applying machine learning.

The EDA will include:

* Dataset dimensions
* Data types
* Missing values
* Duplicate observations
* Class distribution
* Feature cardinality
* Descriptive statistics
* Distribution of numerical features
* Distribution of protocol features
* Distribution of TCP flags
* Correlation analysis
* Identification of highly correlated features
* Identification of outliers
* Comparison of features between benign and malicious traffic

Special attention will be given to highly skewed variables such as traffic rates, packet sizes, and timing-related features.

---

# Statistical Hypothesis Testing

Based on the results of the EDA, several hypotheses will be formulated and tested.

Potential research questions include:

### Hypothesis 1 — Traffic Rate

Does network traffic rate (`Rate`) differ significantly between benign and malicious traffic?

### Hypothesis 2 — Packet/Traffic Size

Do packet and traffic size characteristics differ significantly between benign and malicious traffic?

Relevant features may include:

* `Tot size`
* `AVG`
* `Std`
* `Variance`

### Hypothesis 3 — Protocol Usage

Is protocol usage associated with whether network traffic is benign or malicious?

### Hypothesis 4 — TCP Flags

Does the distribution of TCP flags differ between benign and malicious traffic?

The final hypotheses will be selected and refined after completing the EDA.

Statistical significance will not be evaluated using p-values alone. Where appropriate, **effect sizes** will also be reported.

---

# Data Preprocessing

Before machine learning, the dataset will be prepared using an appropriate preprocessing pipeline.

Potential preprocessing steps include:

* Handling missing values
* Checking and handling duplicate observations
* Investigating outliers
* Encoding categorical variables if necessary
* Feature scaling where required
* Creating the binary target variable
* Removing features that could cause data leakage
* Splitting the data into training and testing sets

Preprocessing decisions will be based on the results of the EDA.

---

# Data Leakage

Data leakage will be treated as an important part of the project.

Since the dataset contains network traffic originating from multiple source files/scenarios, a simple random train/test split may potentially result in very similar traffic patterns appearing in both training and testing data.

Therefore, different evaluation strategies will be investigated.

Potential approaches include:

* Random train/test split
* Group-aware train/test split based on the original source file
* Cross-validation where appropriate

The goal is to determine whether high model performance is caused by genuine generalization or by similarities between training and testing observations.

---

# Machine Learning

Machine learning will be performed only after the EDA and statistical analysis are completed.

The primary task will be binary classification:

```text
BENIGN → 0
MALICIOUS → 1
```

Potential baseline and machine learning models include:

* Logistic Regression
* Decision Tree
* Random Forest
* Gradient Boosting / XGBoost

The final selection of models may change depending on the results of the exploratory analysis.

---

# Model Evaluation

Models will be evaluated using multiple metrics rather than accuracy alone.

Planned metrics include:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* PR-AUC
* Confusion Matrix

Special attention will be given to **recall for the malicious class**, since failing to detect malicious traffic can be more problematic than incorrectly classifying benign traffic as malicious.

---

# Feature Importance and Explainability

After selecting the best-performing models, feature importance will be investigated.

Potential approaches include:

* Tree-based feature importance
* Permutation importance
* SHAP values

The goal is to determine which network traffic characteristics contribute most to the classification of malicious traffic.

---

# Feature Ablation

If time permits, different groups of features will be evaluated separately.

Potential feature groups include:

* Protocol features
* TCP flag features
* Traffic statistics
* Timing features
* Packet size features

The goal is to investigate whether the model relies primarily on a specific type of network information.

---

# Expected Results

The project aims to determine:

* Whether benign and malicious IoT traffic can be reliably distinguished using the available features
* Which features show the strongest statistical differences
* Which machine learning models perform best
* Whether random train/test splitting produces overly optimistic results
* How well the models generalize to previously unseen traffic scenarios
* Which features are most important for detecting malicious traffic

---

# Project Structure

```text
IoT-2023/
│
├── data/
│   └── ...
│
├── notebooks/
│   ├── 01_data_loading.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_hypothesis_testing.ipynb
│   ├── 04_preprocessing.ipynb
│   ├── 05_machine_learning.ipynb
│   └── 06_model_interpretation.ipynb
│
├── src/
│   └── ...
│
├── figures/
│   └── ...
│
├── README.md
└── requirements.txt
```

---

# Current Status

* [x] Dataset selected
* [x] 5% sample from each source CSV created
* [x] Dataset merged into a single DataFrame
* [x] Initial dataset inspection
* [x] Missing value inspection
* [x] Feature cardinality inspection
* [ ] Detailed EDA
* [ ] Final research questions
* [ ] Final hypotheses
* [ ] Statistical hypothesis testing
* [ ] Data preprocessing
* [ ] Leakage analysis
* [ ] Baseline model
* [ ] Machine learning models
* [ ] Model evaluation
* [ ] Feature importance
* [ ] Final analysis
* [ ] Conclusions

---

# Technologies

The project will primarily use Python and the following libraries:

* Python
* pandas
* NumPy
* Matplotlib
* Seaborn
* SciPy
* scikit-learn
* SHAP
* XGBoost (if needed)

---

# Main Research Question

> **Can network traffic features be used to reliably distinguish benign from malicious IoT network traffic, and how does the evaluation strategy affect the estimated performance of the resulting machine learning models?**
