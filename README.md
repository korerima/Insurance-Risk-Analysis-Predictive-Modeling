# Insurance-Risk-Analysis-Predictive-Modeling

This project focuses on analyzing historical car insurance data to uncover risk patterns, optimize premium pricing, and identify low-risk customer segments for AlphaCare Insurance Solutions (ACIS).


## 📁 Project Structure

insurance-risk-analytics/
├── datasets/
│ └── MachineLearningRating_v3.csv # Tracked by DVC
├── Note books
│ └── 01_eda.ipynb # Exploratory Data Analysis
├── requirements.txt # Environment dependencies
├── .dvc/ # DVC tracking folder
├── .gitignore
├── README.md
└── ...


---

## 📦 Setup Instructions

**Clone the repo:**
```bash
git clone https://github.com/korerima/Insurance-Risk-Analysis-Predictive-Modeling.git
cd Insurance-Risk-Analysis-Predictive-Modeling
```

**Create a virtual environment and activate it**
```bash
python -m venv venv
venv\Scripts\activate  # On Windows
```

**Install dependencies**
```bash
pip install -r requirements.txt
```

**Rehydrate data with DVC**
```bash
pip install dvc
dvc pull
```
