# IR Project: Replicating the LightGBM Paper

This repository contains the code and documentation for the Information Retrieval project conducted as part of the UNIPI course. Our goal is to replicate the experimental findings of the LightGBM paper by applying the model on three widely used datasets: **Allstate_claim**, **Flight Delay**, and **LETOR**. This replication study helps verify whether the trends reported in the original paper hold consistently across different datasets and hardware environments.

---

## Overview

The project involves:
- **Data Preprocessing:** Cleaning and preparing the datasets according to the paper’s guidelines.
- **Model Training:** Implementing and training the LightGBM model on each dataset.
- **Comparison Models:** While our main objective was testing LightGBM, we also implemented the comparison models described in the paper, including:
  - **XGBoost with Histogram Splitting**
  - **XGBoost with Exact Splitting**
  - **LightGBM Baseline** (without GOSS or EFB enhancements)
- **Evaluation:** Comparing the performance metrics and trends with those reported in the original study.
- **Analysis:** Verifying that the observed trends remain consistent even when executed on a machine with different hardware specifications.

**Hardware Specifications:**
- **Our Machine:** 128GB RAM, 16 cores
- **Original Study's Machine:** 256GB RAM, 24 cores (E5-2670 v3 CPUs)

Despite the differences in hardware, our results show very similar trends to those in the original paper, thereby reinforcing the robustness of the model’s performance.

---

## Datasets

- **Allstate_claim:** Insurance claim data used for predictive modeling.
- **Flight Delay:** Data containing flight delay records.
- **LETOR:** A benchmark dataset for learning to rank in information retrieval.

Each dataset has been processed and utilized to closely mirror the experimental setup of the LightGBM paper.

---
