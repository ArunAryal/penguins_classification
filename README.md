# 🐧 Penguin Species Classification

A machine learning project that classifies penguin species (**Adelie**, **Chinstrap**, **Gentoo**) based on physical measurements and habitat data. Built with a scikit-learn Pipeline and deployed as an interactive web app using Streamlit.

---

## 📌 Project Overview

This project walks through the full ML workflow — from exploratory data analysis to model deployment — using the [Palmer Penguins dataset](https://allisonhorst.github.io/palmerpenguins/). A K-Nearest Neighbors (KNN) classifier is trained inside a scikit-learn Pipeline that handles preprocessing, encoding, and scaling automatically.

---

## 📁 Project Structure

```
penguins_classification/
│
├── penguin_classification.ipynb   # Main notebook (EDA, training, evaluation)
├── deploy.py                      # Streamlit web app for live predictions
├── pipe.joblib                    # Saved trained pipeline (not committed)
├── penguins.csv                   # Dataset
├── analysis.html                  # Sweetviz EDA report (not committed)
├── barPlot.png                    # Species distribution bar chart
├── requirements.txt               # Pinned dependencies
└── confusion_matrix_png.png       # Confusion matrix visualization
```

---

## 🔍 Dataset

The dataset contains **344 penguin records** with the following features:

| Feature          | Description                        |
|------------------|------------------------------------|
| `island`         | Island where penguin was found     |
| `bill_length_mm` | Length of bill (mm)                |
| `bill_depth_mm`  | Depth of bill (mm)                 |
| `flipper_length_mm` | Flipper length (mm)             |
| `body_mass_g`    | Body mass (grams)                  |
| `sex`            | Sex of the penguin                 |
| `species`        | Target — Adelie / Chinstrap / Gentoo |

> 11 rows (~3.2%) with missing values were dropped during preprocessing.

---

## ⚙️ ML Pipeline

The scikit-learn Pipeline consists of three stages:

1. **OneHotEncoding** — Encodes categorical features (`island`, `sex`) using `ColumnTransformer`
2. **StandardScaler** — Scales numerical features to zero mean and unit variance
3. **KNeighborsClassifier** — Classifies species; optimal `k` selected via GridSearchCV (tested: 3, 5, 7, 9, 11)

---

## 📊 Model Evaluation

- **Cross-Validation Accuracy** (5-fold): evaluated on training set
- **Metrics**: Accuracy Score, Classification Report, Confusion Matrix

The confusion matrix is saved as `confusion_matrix_png.png`.

---

## 🚀 Streamlit Web App

The app (`deploy.py`) loads the saved pipeline and takes user input to predict penguin species in real time.

### Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/ArunAryal/penguins_classification.git
cd penguins_classification

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the app
streamlit run deploy.py
```

### App Inputs

| Input               | Type     | Options                          |
|---------------------|----------|----------------------------------|
| Island              | Dropdown | Torgersen, Biscoe, Dream         |
| Bill Length (mm)    | Number   | —                                |
| Bill Depth (mm)     | Number   | —                                |
| Flipper Length (mm) | Number   | —                                |
| Body Mass (g)       | Number   | —                                |
| Sex                 | Dropdown | male, female                     |

> The app validates that all numeric fields are filled before predicting. Empty fields will show a warning instead of returning a bad prediction.

---

## 🛠️ Tech Stack

- **Python** — Core language
- **Pandas & NumPy** — Data manipulation
- **Scikit-learn** — ML pipeline, KNN, GridSearchCV
- **Matplotlib** — Visualizations
- **Sweetviz** — Automated EDA report
- **Joblib** — Model serialization
- **Streamlit** — Web app deployment

---

## 📦 Requirements

```
pandas==1.5.3
numpy==1.24.4
scikit-learn==1.8.0
matplotlib==3.10.0
sweetviz==2.3.1
joblib==1.5.2
streamlit==1.52.2
```

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).