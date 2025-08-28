# Machine Learning Classification & Regression Projects

A collection of machine learning projects demonstrating SVM and KNN algorithms for both classification and regression tasks.

## 📊 Projects Overview

### 1. Wine Cultivar Classification
- **Algorithm:** Support Vector Classifier (SVC) vs K-Nearest Neighbors (KNN)
- **Dataset:** Sklearn Wine Recognition Dataset (178 samples, 13 features)
- **Task:** Classify wine cultivars based on chemical composition
- **Best Result:** 98.6% accuracy with SVC (RBF kernel)

### 2. California Housing Price Prediction
- **Algorithm:** Support Vector Regression (SVR) vs KNN Regression
- **Dataset:** Sklearn California Housing Dataset (20,640 samples, 8 features)
- **Task:** Predict house prices based on location and demographics
- **Best Result:** RMSE $47,500 with SVR (tuned hyperparameters)

## 🔧 Technologies Used

- **Programming Language:** Python 3.8+
- **Libraries:** scikit-learn, pandas, numpy, matplotlib, seaborn
- **Algorithms:** SVM (SVC/SVR), K-Nearest Neighbors
- **Techniques:** Cross-validation, Grid Search, Feature Scaling

## 📁 Repository Structure

```
├── notebooks/
│   ├── wine_classification.ipynb    # SVC vs KNN classification
│   ├── housing_regression.ipynb     # SVR vs KNN regression
├── data/                            # Dataset files (if external)
├── src/                             # Utility functions (optional)
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

## 🚀 Getting Started

### Prerequisites
```bash
pip install -r requirements.txt
```

### Running the Projects
1. Clone this repository:
   ```bash
   git clone https://github.com/Aqib-02/Ml-Projects.git
   cd ml-projects
   ```

2. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn jupyter
   ```

3. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

4. Open and run the notebooks in the `notebooks/` folder

## 📈 Key Results

| Project | Algorithm | Metric | Score |
|---------|-----------|--------|-------|
| Wine Classification | SVC (RBF) | Accuracy | 98.6% |
| Wine Classification | KNN (k=5) | Accuracy | 97.2% |
| Housing Regression | SVR (RBF) | RMSE | $47.5k |
| Housing Regression | KNN (k=7) | RMSE | $49.2k |

## 🎯 Key Insights

### Classification (Wine Dataset)
- **SVC with RBF kernel** performed best with proper hyperparameter tuning
- **Feature scaling** was crucial for both algorithms
- **Cross-validation** showed consistent performance across folds

### Regression (Housing Dataset)
- **SVR** handled outliers better than KNN
- **Geographic features** (latitude/longitude) were particularly important for KNN
- **Hyperparameter tuning** improved RMSE by 15%

## 🔍 Algorithm Comparison

### When to Use SVC/SVR:
- Non-linear relationships in data
- High-dimensional feature spaces
- Need for robust performance with outliers

### When to Use KNN:
- Local patterns are important
- Interpretability is required
- Simple baseline model needed

## 📚 Learning Outcomes

- Implemented and compared SVM and KNN algorithms
- Performed hyperparameter tuning using Grid Search
- Applied proper evaluation techniques (cross-validation, train/test splits)
- Understood the importance of feature scaling
- Analyzed model performance using multiple metrics

## 🛠️ Future Improvements

- [ ] Feature engineering and selection
- [ ] Ensemble methods combining SVC/SVR and KNN
- [ ] Advanced hyperparameter optimization (RandomizedSearchCV)
- [ ] Model interpretability analysis
- [ ] Deployment using Flask/Streamlit

## 👨‍💻 About

This project demonstrates practical machine learning skills including:
- Data preprocessing and exploration
- Algorithm implementation and comparison
- Model evaluation and validation
- Result interpretation and business insights

**Contact:** aqibuddin02@gmail.com | **LinkedIn:** linkedin.com/in/mohammedaqib0717 | 
---

*Note: Both datasets are available directly through scikit-learn, making this project easily reproducible.*
