# 🔐 Network Security ML Pipeline (End-to-End Production Project)

## 📌 Overview

This project is a **complete end-to-end Machine Learning pipeline** designed to detect malicious network activity (e.g., cyber attacks) using structured network data.

It follows **industry-standard architecture**, covering the entire lifecycle:

* Data Ingestion
* Data Validation
* Data Transformation
* Model Training
* Model Evaluation
* Model Deployment (FastAPI)
* Cloud Integration (S3 ready)

---

## 🧠 Problem Statement

In modern systems, detecting malicious network traffic is critical.

This project aims to:

> **Classify network data into normal or malicious activity using Machine Learning**

---

## 🏗️ Project Architecture

```
Data Source (MongoDB / CSV)
        ↓
Data Ingestion
        ↓
Data Validation
        ↓
Data Transformation
        ↓
Model Trainer
        ↓
Artifacts (model.pkl, preprocessor.pkl)
        ↓
FastAPI (Prediction API)
        ↓
Frontend (HTML Table)
        ↓
Deployment Ready (AWS / S3)
```

---

## 📂 Project Structure

```
networksecurity/
│
├── components/
│   ├── data_ingestion.py
│   ├── data_validation.py
│   ├── data_transformation.py
│   ├── model_trainer.py
│
├── pipeline/
│   ├── training_pipeline.py
│   ├── batch_prediction.py
│
├── entity/
│   ├── config_entity.py
│   ├── artifact_entity.py
│
├── utils/
├── logging/
├── exception/
├── cloud/
│   └── s3_syncer.py
```

---

## ⚙️ ML Pipeline Explanation

### 🔹 1. Data Ingestion

* Fetches data from MongoDB / local source
* Stores raw data into artifact directory

### 🔹 2. Data Validation

* Schema validation (columns, data types)
* Missing value checks
* Drift detection
* Ensures data consistency

### 🔹 3. Data Transformation

* Handles missing values using **KNN Imputer**
* Feature engineering
* Converts data into NumPy arrays
* Saves preprocessing object (`preprocessor.pkl`)

### 🔹 4. Model Trainer

* Trains multiple ML models:

  * Random Forest
  * Decision Tree
  * Gradient Boosting
  * AdaBoost
* Hyperparameter tuning (GridSearchCV)
* Selects best model
* Saves trained model (`model.pkl`)

---

## 📦 Artifacts Generated

| Artifact         | Description                  |
| ---------------- | ---------------------------- |
| train.npy        | Transformed training data    |
| test.npy         | Transformed testing data     |
| preprocessor.pkl | Data transformation pipeline |
| model.pkl        | Final trained ML model       |

---

## 🔌 FastAPI Integration

### Endpoints:

#### 🔹 Train Model

```
GET /train
```

#### 🔹 Predict

```
POST /predict
```

* Upload CSV file
* Returns prediction results
* Displays output in HTML table

---

## 🌐 Deployment Ready

This project is designed for deployment with:

* AWS S3 (artifact storage)
* FastAPI backend
* Docker support (Dockerfile included)

---

## ☁️ Cloud Integration

* S3 sync functionality implemented
* Upload:

  * Artifacts
  * Final models
* Ready for production scaling

---

## 🧪 Tech Stack

* Python
* Scikit-learn
* Pandas, NumPy
* FastAPI
* MongoDB
* AWS S3
* MLflow (tracking)
* Docker

---

## ▶️ How to Run

### 1. Clone Repo

```
git clone <your-repo-url>
cd project
```

### 2. Install Dependencies

```
pip install -r requirements.txt
```

### 3. Run Application

```
python app.py
```

### 4. Open Swagger UI

```
http://127.0.0.1:8000/docs
```

---

## 📊 Features

✔ Modular ML pipeline
✔ Scalable architecture
✔ Production-ready structure
✔ Logging & exception handling
✔ API-based prediction
✔ Cloud-ready (S3 integration)

---

## 🔥 Key Highlights

* Industry-standard folder structure
* End-to-end ML pipeline (not just model)
* Fully deployable project
* Real-world use case (cybersecurity)

---

## 👨‍💻 Author

**Rohit Kumar**

---

## 📌 Future Improvements

* Add Deep Learning models
* Real-time streaming prediction
* Frontend dashboard (React)
* CI/CD pipeline

---

## ⭐ If you like this project

Give it a ⭐ on GitHub!
