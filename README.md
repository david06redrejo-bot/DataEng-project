# Advanced Data Engineering Project

## Overview
This repository contains the advanced data engineering project structure, tailored to synthesize diverse data archetypes including structured matrices, sparse datasets, unstructured media, and relational graphs. 

## Project Structure
```
PROJECT/
├── data/
│   ├── raw/             # Raw data downloaded from Kaggle or other sources
│   └── processed/       # Cleaned and processed data ready for modeling
├── notebooks/           # Jupyter notebooks for exploratory data analysis and modeling
│   └── 01_exploratory_data_analysis.ipynb
├── scripts/             # Python scripts for data ingestion and processing
│   └── download_kaggle.py  # Script for downloading direct Kaggle links
├── reports/             # Generated analysis, theoretical notes (Latex/Markdown)
├── context.md           # The Technical Design Report
├── README.md            # The top-level project overview
└── requirements.txt     # The required dependencies to run the project
```

## Setup and Prerequisites

### 1. Requirements
Ensure Python 3 is installed. To test and run the code, use the following `pip` command to install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Ingesting Data from Kaggle
To automatically download the desired dataset for this project, you can use the `download_kaggle.py` script.
**Note:** You must have the `kaggle` Python package installed and your `kaggle.json` credentials placed in `~/.kaggle/kaggle.json` (or `%USERPROFILE%\.kaggle\kaggle.json` on Windows).

```bash
python scripts/download_kaggle.py "https://www.kaggle.com/datasets/user/dataset"
```
The data will be fetched, unzipped, and stored in `data/raw/`.

## Author
Developed for the Advanced Data Engineering course.