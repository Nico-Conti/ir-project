# IR Project: Replicating the LightGBM Paper

This repository contains the code and documentation for the Information Retrieval project conducted as part of the UNIPI course. Our goal is to replicate the experimental findings of the LightGBM paper by applying the model on three widely used datasets: **Allstate_claim**, **Flight Delay**, and **LETOR**. This replication study helps verify whether the trends reported in the original paper hold consistently across different datasets and hardware environments.

---
<br><br>
<p align = "center">
  <img src = "images/Stemma_unipi.svg.png" width="150" height="150">
</p>

<p align = "center">
  Computer Science Department
  <br>
  A project for
  <br>
  the Information Retrieval course
  <br>
  in the CS Master's program at the University of Pisa.
</p>

## Authors
* **Francesco Alizzi**        - [FrancescoAlizzi](https://github.com/FrancescoAlizzi)
* **Nico Conti** - [NicoConti](https://github.com/Nico-Conti)



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

## 📖 Reference
If you use this project or LightGBM in your work, please cite the original LightGBM paper:

> **Ke, Guolin, et al.**  
> *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*  
> Advances in Neural Information Processing Systems (**NeurIPS**), 2017.  
> 📄 [Read the Paper](https://proceedings.neurips.cc/paper_files/paper/2017/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf)

```bibtex
@inproceedings{ke2017lightgbm,
  title={LightGBM: A Highly Efficient Gradient Boosting Decision Tree},
  author={Ke, Guolin and Meng, Qi and Finley, Thomas and Wang, Taifeng and Chen, Wei and Ma, Weidong and Ye, Qiwei and Liu, Tie-Yan},
  booktitle={Advances in Neural Information Processing Systems},
  volume={30},
  year={2017}
}
```

