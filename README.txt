
🌸 Iris Flower Classification with PSO-Optimized Random Forest

This project aims to classify Iris flower species using a **Random Forest Classifier**, and improve the model’s performance using **Particle Swarm Optimization (PSO)** for hyperparameter tuning.

---

Project Structure

- **app.py**  
  A **Streamlit** web interface where users can input flower measurements (sepal and petal lengths and widths) and get real-time species predictions.

- **iris.ipynb**  
  A **Jupyter Notebook** that contains the core logic of the model. It includes:
  - Data loading and preprocessing  
  - Model training using Random Forest  
  - Optimization using **PSO**  
  - Evaluation with accuracy, confusion matrix, and classification report

- **README.txt**  
  This file contains a summary of the project, structure, and results before and after optimization.

---

##  Performance Comparison

###  Before Optimization (Default Random Forest)

Accuracy: 0.7333333333333333333
Confusion Matrix:
 [[10  0  0]
 [ 0  8  1]
 [ 0  7  4]]
Classification Report:
               precision    recall  f1-score   support

           0       1.00      1.00      1.00        10
           1       0.53      0.89      0.67         9
           2       0.80      0.36      0.50        11

    accuracy                           0.73        30
   macro avg       0.78      0.75      0.72        30
weighted avg       0.79      0.73      0.72        30

###  After PSO Optimization (Tuned Random Forest)

- **Accuracy**: `1.00`

- **Confusion Matrix**:
  [[10  0  0]
   [ 0  9  0]
   [ 0  0 11]]

- **Classification Report**:
                precision    recall  f1-score   support

        setosa       1.00      1.00      1.00        10
    versicolor       1.00      1.00      1.00         9
     virginica       1.00      1.00      1.00        11

      accuracy                           1.00        30
     macro avg       1.00      1.00      1.00        30
  weighted avg       1.00      1.00      1.00        30

---

##  Optimization Details

We applied **Particle Swarm Optimization (PSO)** to tune the following hyperparameters of the Random Forest:

- `n_estimators` (number of trees)  
- `max_depth` (maximum depth of trees)

PSO was selected for its global optimization ability and simplicity. Even though the dataset is small and achieved perfect accuracy even before optimization, this experiment showcases how PSO can automate hyperparameter search effectively.

---

##  Technologies Used

- Python
- scikit-learn
- Streamlit
- PSO (custom implementation)
- pandas, NumPy, matplotlib

---

##  Author

Ali Mostafa – AI Engineer, New Ismailia National University

MIT License
Copyright (c) 2025 Aly Mostafa

