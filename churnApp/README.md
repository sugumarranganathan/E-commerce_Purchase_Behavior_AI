# 🛒 E-commerce Purchase Behavior AI (Django + Machine Learning)

## 📌 Project Overview

This project is a Django-based web application integrated with a Machine Learning model to predict customer purchase behavior (Churn / Not Churn).

---

## 🚀 1. Project Setup

### Create Project

```
django-admin startproject fashionChurn
cd fashionChurn
```

### Create App

```
python manage.py startapp churnApp
```

---

## ⚙️ 2. Settings Configuration

* Added `churnApp` in INSTALLED_APPS
* Configured templates and static files

---

## 🗄️ 3. Database Setup

```
python manage.py migrate
```

* SQLite database used (`db.sqlite3`)

---

## 🧾 4. Forms (User Input)

User provides input like:

* Age
* Gender
* Purchase history

---

## 🧠 5. Machine Learning Model

* Model trained using `train_model.py`
* Saved as:

  * `fashion_churn_rf_model.pkl`
  * Label encoders `.pkl`

---

## 🌐 6. Frontend (HTML Page)

* Input form created using HTML
* Users enter data and submit

---

## 🔁 7. Views (Backend Logic)

* Receives user input (POST method)
* Loads ML model
* Predicts output
* Sends result to frontend

---

## 🔄 8. GET & POST Methods

* GET → Load page
* POST → Submit form data

---

## ⚡ 9. Processing Flow

1. User enters data
2. Data sent to backend
3. Data processed using encoders
4. ML model predicts result
5. Output displayed

---

## 📊 10. Output Page

Displays prediction:

* Purchase / Not Purchase
* Churn / Not Churn

---

## 🔗 11. URL Configuration

### App URLs

```
path('', views.predict)
```

### Project URLs

```
path('', include('churnApp.urls'))
```

---

## ☁️ 12. Deployment

* GitHub used for version control
* Render used for deployment (`render.yaml`)

---

## 🔥 Final Workflow

User → Input Form → Backend Processing → ML Prediction → Output Page

---

## 🧾 Technologies Used

* Python
* Django
* Machine Learning (Random Forest)
* HTML/CSS
* SQLite

---

## ✅ Conclusion

This project demonstrates how Machine Learning can be integrated with Django to build a real-time prediction web application.

---
