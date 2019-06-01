# XGBoost Classification with Python and Scikit-Learn


===============================================================================


## Table of Contents


I have divided this project into various topics which are listed below:-


1.	Introduction to XGBoost algorithm

2.	XGBoost algorithm intuition

3.	Boosting algorithms

4.	Advantages and disadvantages of XGBoost

5.	XGBoost parameters

6.	Parameter tuning with XGBoost

7.	Detect and control overfitting

8.	Handle imbalanced dataset with XGBoost

9.	k-fold Cross Validation using XGBoost

10.	Boosting trees with XGBoost

11.	Feature selection with XGBoost

12.	The problem statement

13.	Results and conclusion

14.	References

===============================================================================


## 1. Introduction to XGBoost algorithm


**XGBoost** stands for **Extreme Gradient Boosting**.  XGBoost is a powerful machine learning algorithm that is dominating the world of applied machine learning and Kaggle competitions. It is an implementation of gradient boosted trees designed for speed and accuracy.

XGBoost (Extreme Gradient Boosting) is an advanced implementation of the gradient boosting algorithm. It has proved to be a highly effective machine learning algorithm extensively used in machine learning competitions. XGBoost has high predictive power and is almost 10 times faster than other gradient boosting techniques. It also includes a variety of regularization parameters which reduces overfitting and improves overall performance. Hence, it is also known as **regularized boosting** technique.

XGBoost was developed by Tianqi Chen in C++ but also enables interfaces for Python, R and Julia. Initially, he started XGBoost as a research project as part of the Distributed (Deep) Machine Learning Community. It became popular in the ML competitions after its use in the winning solution of the Higgs Machine Learning Challenge. 


===============================================================================


## 3. Advantages and disadvantages of XGBoost algorithm


XGBoost is very popular in machine learning competitions. Its advantages are listed below:-


1.	XGBoost is widely used for its computational scalability. It was originally written in C++ and hence is comparatively faster than other ensemble classifiers.
2.	It can handle missing values and is robust to outliers.
3.	It does not require feature scaling and can deal with irrelevant inputs.
4.	It can handle mixed predictors (quantitative and qualitative).
5.	The core XGBoost algorithm is parallelizable. So it can harness the power of multi-core computers. 
6.	It has shown greater accuracy than other machine learning algorithms on variety of machine learning datasets.
7.	XGBoost has parameters for cross-validation, regularization, user-defined objective functions, missing values, tree parameters and scikit-learn compatible API.


The disadvantages of XGBoost are as follows:-


1.	It can’t extract the linear combination of features.
2.	It has small predictive power and hence high variance.
