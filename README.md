# CSC311-Introduction-to-Machine-Learning-2025Fall-Project
CSC311 Introduction to Machine Learning Project 2025 Fall

## Logistic Regression Pipeline

### Iterations

- **Iter1_knn**
    From Template Code  
    - **Training accuracy:** 0.727  
    - **Test accuracy:** 0.611  

- **Iter2_logreg**  
    Apply logistic regression on baseline  
    - **Training accuracy (LogReg):** 0.628  
    - **Test accuracy (LogReg):** 0.710  

- **Iter3_logreg**  
    Previous + Split by `student_id` to avoid data leakage  
    - **Training accuracy (LogReg, student-wise split):** 0.673  
    - **Test accuracy (LogReg, student-wise split):** 0.619  

- **Iter4_logreg**  
    Previous + Use 3 for missing ratings (avoid loss of data)  
    - **Training accuracy (LogReg, student-wise split, imputed):** 0.639  
    - **Test accuracy (LogReg, student-wise split, imputed):** 0.663

- **Iter5_logreg**  
    Previous + Consider all features (all ratings & bag-of-words)  
    - Maximum features for text: 2000  
    - **Training accuracy (full features):** 0.984  
    - **Test accuracy (full features):** 0.635

- **Iter6_logreg**  
    Previous + Hyperparameter tuning for `C` (best `C=0.1`)  
    - Maximum features for text: 3000  
    - **Training accuracy:** 0.958  
    - **Test accuracy:** 0.683