# detecting_heart_disease
Using machine learning to detect heart disease, maximizing recall with accuracy as secondary metric. <br>
Project completed for practice and portfolio.

**Models Used**
* Linear Regression
* Support Vector Machine
* Random Forest
* Gradient Boosting
* Stacked Models
    * Tuned each of the above four types of models on an enhanced dataset that included the predictions of the four original models

**Results**
* A gradient boosting classifier using stacked data and a threshold low enough to have perfect recall on the training data had the highest accuracy of those models with 1.0 recall on the test data.
* The random forest classifier had the highest AUC but suffered more in accuracy when maximizing recall.

Used dataset from UCI Machine Learning Repository: <br>
Janosi,Andras, Steinbrunn,William, Pfisterer,Matthias, and Detrano,Robert. (1988). Heart Disease. UCI Machine Learning Repository. https://doi.org/10.24432/C52P4X.
