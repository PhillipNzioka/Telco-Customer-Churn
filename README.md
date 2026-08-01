# Telco Customer Churn Prediction

A machine learning project that predicts whether a telecommunications customer is likely to churn based on demographic information, subscribed services, billing information, and contract details.

The project covers the complete machine learning lifecycle including:

- Data cleaning and preprocessing
- Exploratory Data Analysis (EDA)
- Feature engineering
- Model training and evaluation
- Model serialization
- REST API deployment using FastAPI

---

## Project Structure

```text
telco-churn-prediction/
│
├── api/
│   └── main.py                # FastAPI application
│
├── data/
│   └── telco_churn.csv        # Dataset
│
├── models/
│   ├── ml_pipeline.joblib     # Trained machine learning pipeline
│   └── target_labels.joblib   # LabelEncoder for target decoding
│
├── notebooks/
│   └── project.ipynb          # Data analysis and model development
│
├── README.md
└── .gitignore
```

---

# Project Overview

Customer churn is one of the most important business metrics for subscription-based businesses. Predicting which customers are likely to leave enables organizations to take proactive retention measures.

This project builds a complete end-to-end churn prediction solution using Scikit-learn and deploys the trained model through a REST API using FastAPI.

---

# Dataset

The project uses the Telco Customer Churn dataset containing customer information such as:

- Gender
- Senior Citizen status
- Partner
- Dependents
- Tenure
- Phone Service
- Multiple Lines
- Internet Service
- Online Security
- Online Backup
- Device Protection
- Tech Support
- Streaming TV
- Streaming Movies
- Contract Type
- Paperless Billing
- Payment Method
- Monthly Charges
- Total Charges

Target Variable

```
Churn
```

Possible values:

- Yes
- No

---

# Machine Learning Workflow

## 1. Import Libraries

The project utilizes:

- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib

---

## 2. Load Dataset

The dataset is loaded into a Pandas DataFrame for preprocessing and analysis.

---

## 3. Exploratory Data Analysis (EDA)

Initial exploration includes:

- Dataset inspection
- Feature distributions
- Missing value identification
- Target class distribution
- Data types

---

## 4. Data Cleaning

The notebook performs several preprocessing tasks including:

### Standardizing column names

- Convert column names to lowercase
- Remove leading/trailing whitespace

### Type conversion

`TotalCharges`

Converted from string to floating-point values.

### Missing values

Rows containing missing values are cleaned before model training.

---

## 5. Feature Engineering

Features are separated into:

### Numerical Features

- tenure
- monthlycharges
- totalcharges

### Categorical Features

- gender
- seniorcitizen
- partner
- dependents
- phoneservice
- multiplelines
- internetservice
- onlinesecurity
- onlinebackup
- deviceprotection
- techsupport
- streamingtv
- streamingmovies
- contract
- paperlessbilling
- paymentmethod

---

## 6. Train-Test Split

The dataset is split into training and testing sets using:

- 80% Training
- 20% Testing

Stratified sampling is used to preserve the class distribution.

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

---

## 7. Data Preprocessing Pipeline

The project uses Scikit-learn Pipelines for reproducible preprocessing.

### Numerical Pipeline

- StandardScaler

### Categorical Pipeline

- OneHotEncoder
- handle_unknown='ignore'

The pipelines are combined using a ColumnTransformer.

---

# Models Evaluated

Three classification algorithms were trained and evaluated.

## Logistic Regression

```python
LogisticRegression(max_iter=1000)
```

---

## K-Nearest Neighbors

```python
KNeighborsClassifier(
    n_neighbors=7,
    metric='euclidean'
)
```

---

## Random Forest

```python
RandomForestClassifier(
    n_estimators=1000,
    max_depth=10
)
```

---

# Model Evaluation

Each model is evaluated using:

- Classification Report
- Precision
- Recall
- F1-Score

After comparison, the best-performing model is selected for deployment.

The final production pipeline consists of:

```
Preprocessing
        ↓
Machine Learning Model
```

This allows raw customer data to be passed directly into the model without requiring manual preprocessing.

---

# Model Serialization

The trained model is saved using Joblib.

```python
joblib.dump(full_pipeline,
            "../models/ml_pipeline.joblib")
```

The target encoder is also saved:

```python
joblib.dump(le,
            "../models/target_labels.joblib")
```

---

# FastAPI Deployment

The project exposes the trained model as a REST API.

## Health Check

```
GET /health
```

Response

```json
{
    "status": "API is running successfully"
}
```

---

## Prediction Endpoint

```
POST /predict
```

Example Request

```json
{
    "gender":"Female",
    "seniorcitizen":0,
    "partner":"Yes",
    "dependents":"No",
    "tenure":12,
    "phoneservice":"Yes",
    "multiplelines":"No",
    "internetservice":"Fiber optic",
    "onlinesecurity":"No",
    "onlinebackup":"Yes",
    "deviceprotection":"No",
    "techsupport":"No",
    "streamingtv":"Yes",
    "streamingmovies":"Yes",
    "contract":"Month-to-month",
    "paperlessbilling":"Yes",
    "paymentmethod":"Electronic check",
    "monthlycharges":79.85,
    "totalcharges":920.45
}
```

Example Response

```json
{
    "prediction":"Yes"
}
```

---

# Running the Project

## Clone Repository

```bash
git clone https://github.com/yourusername/Telco-Customer-Churn.git
```

---

## Create Virtual Environment

Windows

```bash
python -m venv venv
```

Activate

```bash
venv\Scripts\activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Train the Model

Open the notebook:

```
notebooks/project.ipynb
```

Run all cells to:

- Clean the dataset
- Train the models
- Save the pipeline
- Save the LabelEncoder

Generated files:

```
models/
    ml_pipeline.joblib
    target_labels.joblib
```

---

## Run the API

Navigate to the API folder.

```bash
cd api
```

Run:

```bash
uvicorn main:app --reload
```

---

Open the Swagger documentation:

```
http://127.0.0.1:8000/docs
```

---

# Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib
- FastAPI
- Pydantic
- Uvicorn

---

# Future Improvements

- Hyperparameter tuning using GridSearchCV
- Cross-validation
- Feature importance visualization
- Model monitoring
- Docker containerization
- CI/CD pipeline
- Cloud deployment (Azure, AWS, or Google Cloud)
- Interactive frontend using Streamlit or React

---

