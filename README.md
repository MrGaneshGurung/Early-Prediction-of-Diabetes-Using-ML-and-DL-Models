# Diabetes Prediction Using Machine Learning and Deep Learning

This project implements a predictive system for *diabetes classification* using machine learning (ML) and deep learning (DL) models. It demonstrates an *end-to-end pipeline* covering data preprocessing, feature scaling, class imbalance handling, model training, evaluation, and interpretability.

---

##  Dataset
- *PIMA Indians Diabetes Dataset* (768 records, 8 clinical features)  
- Target variable: 0 (non-diabetic) or 1 (diabetic)

---

##  Preprocessing
- Handling missing/implausible values (zeros in features like glucose, BMI, insulin)  
- Median imputation  
- Feature standardisation (zero mean, unit variance)  
- Class balancing using *SMOTE*

---

## Models Implemented
1. Logistic Regression  
2. Support Vector Machine (SVM)  
3. Random Forest  
4. XGBoost  
5. Neural Network (MLP)

*Hyperparameter tuning* was performed using GridSearchCV for optimal performance.

---

## Evaluation Metrics
- Accuracy  
- Precision  
- Recall  
- F1-score  
- AUC-ROC  

*XGBoost* achieved the best performance in terms of accuracy, F1-score, and AUC.

---

## Explainability
- *LIME* was used to interpret model predictions  
- Key features identified: *Glucose, BMI, Insulin*  
- Ensures predictions are clinically meaningful and transparent

---

## Key Highlights
- SMOTE improved recall for diabetic cases, reducing false negatives  
- Ensemble methods (Random Forest, XGBoost) outperformed traditional ML and Neural Networks on this dataset  
- LIME enhanced trust and interpretability for potential clinical deployment

---

## Technologies Used
- Python, Pandas, NumPy, Matplotlib, Seaborn  
- Scikit-learn, XGBoost, Keras (TensorFlow backend)

---

## Applications
- Early detection of diabetes in routine health screenings  
- Automated flagging system for high-risk
