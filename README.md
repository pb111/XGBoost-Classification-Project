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

## 2. XGBoost algorithm intuition

XGBoost (Extreme Gradient Boosting) belongs to a family of boosting algorithms. It uses the gradient boosting (GBM) framework at its core. So, first of all we should know about gradient boosting.

### Gradient boosting 
Gradient boosting is a supervised machine learning algorithm, which tries to predict a target variable by combining the estimates of a set of simpler, weaker models. 


===============================================================================

## 3. Boosting algorithms

The term **boosting** refers to a family of algorithms which converts weak learners to strong learners. Boosting is a sequential process, in which each subsequent model attempts to correct the errors of the previous model. The succeeding models are dependent 
on the previous model. The idea of boosting can be explained as follows:-


-	A subset is created from the original dataset where all the data points are given equal weights. A base model is created on this subset. 
-	This model is used to make predictions on the whole dataset. Errors are then calculated using the actual values and predicted values. The observations which are incorrectly predicted are given higher weights. 
-	Then another model is created which tries to correct the errors from the previous model. Using this model, predictions are made on the dataset.
-	This process is repeated and multiple models are created. Each model corrects the errors of the previous model.
-	The final model which is the strong learner is the weighted mean of all the models (weak learners).


Thus, the boosting algorithm combines a number of weak learners to form a strong learner. The individual models would not perform well on the entire dataset, but they work well for some part of the dataset. Thus, each model actually boosts the performance of the ensemble.


Boosting is shown in the figure below:-

# D-Boosting

===============================================================================


## 4. Advantages and disadvantages of XGBoost algorithm


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
2.	It has great predictive power and hence high variance.


===============================================================================

## 5. XGBoost parameters

Now, we know about the XGBoost model. But, we should also know about the parameters that XGBoost provides. There are three types of parameters that we must set before running XGBoost. These parameters are as follows:-

## General parameters

These parameters relate to which booster we are doing boosting. The common ones are tree or linear model.

## Booster parameters

It depends on which booster we have chosen for boosting.

## Learning task parameters

These parameters decide on the learning scenario. For example, regression tasks may use different parameters than ranking tasks. 

## Command line parameters

In addition there are command line parameters which relate to behaviour of CLI version of XGBoost.


The most important parameters that we should know about are as follows:-

**learning_rate** - It gives us the step size shrinkage which is used to prevent overfitting. Its range is [0,1].

**max_depth** - It determines how deeply each tree is allowed to grow during any boosting round.

**subsample** - It determines the percentage of samples used per tree. Low value of subsample can lead to underfitting.

**colsample_bytree** - It determines the percentage of features used per tree. High value of it can lead to overfitting.

**n_estimators** - It is the number of trees we want to build.

**objective** - It determines the loss function to be used in the process. For example, `reg:linear` for regression problems, `reg:logistic` for classification problems with only decision, `binary:logistic` for classification problems with probability.


XGBoost also supports regularization parameters to penalize models as they become more complex and reduce them to simple models. 
These regularization parameters are as follows:-

**gamma** - It controls whether a given node will split based on the expected reduction in loss after the split. A higher value 
leads to fewer splits. It is supported only for tree-based learners.

**alpha** - It gives us the `L1` regularization on leaf weights. A large value of it leads to more regularization.

**lambda** - It gives us the `L2` regularization on leaf weights and is smoother than `L1` regularization.

Though we are using trees as our base learners, we can also use XGBoost’s relatively less popular linear base learners and one other tree learner known as `dart`. We have to set the `booster` parameter to either `gbtree` (default), `gblinear` or `dart`.


===============================================================================


## 6. Parameter tuning with XGBoost


**Bias-Variance Tradeoff** is one of the most important concepts in machine learning. When the machine learning model has more depth, 
it becomes more complicated. In that case, the model has better ability to fit the training data. The resulting model is a less biased model. But, such a complicated model requires more data for training.

Most of the parameters in XGBoost are about bias-variance trade off. The best model should balance the model complexity with its predictive power. This can be used to strike a balance between complicated and simple models.


===============================================================================


 ## 7. Detect and control overfitting
 
We can detect overfitting by looking at training and test accuracy. When there is high training accuracy and low test accuracy, 
it is likely the case of overfitting.

There are two ways by which we can control overfitting in XGBoost. These are given below:-

-	The first way is to directly control model complexity.

      o	This includes controlling the parameters `max_depth`, `min_child_weight` and `gamma`.
    
-	The second way is to add randomness to make training robust to noise.

    o	This includes controlling the parameters `subsample` and `colsample_bytree`.
    
    o	We can also reduce stepsize by controlling the parameter `eta`. We have to increase `num_round` parameter when we do so.


===============================================================================


## 8. Handle imbalanced dataset with XGBoost

For common machine learning problems, such as ads clickthrough, the dataset is extremely imbalanced. This can affect the 
training of XGBoost model, and there are two ways to improve it.


-	If we care only about the overall performance metric (AUC) of the prediction.

    o	We should balance the positive and negative weights via `scale_pos_weight` parameter.
    
    o	Use AUC for evaluation.
    
-	If we care about predicting the right probability.

    o	In such a case, we cannot rebalance the dataset.
    
    o	We should set parameter `max_delta_step` to a finite number (say 1) to help convergence.


===============================================================================

## 9. k-fold Cross Validation using XGBoost

To build more robust models with XGBoost, we must do k-fold cross validation. In this way, we ensure that the original training 
dataset is used for both training and validation. Also, each entry is used for validation just once. XGBoost supports k-fold cross validation using the `cv()` method. In this method, we will specify several parameters which are as follows:-


**nfolds** - This parameter specifies the number of cross-validation sets we want to build.

**num_boost_round** - It denotes the number of trees we build.

**metrics** - It is the performance evaluation metrics to be considered during CV.

**as_pandas** - It is used to return the results in a pandas DataFrame.

**early_stopping_rounds** - This parameter stops training of the model early if the hold-out metric does not improve 
for a given number of rounds.

**seed** - This parameter is used for reproducibility of results.

We can use these parameters to build a k-fold cross-validation model by calling `XGBoost’s CV()` method.


===============================================================================


## 10. Boosting trees with XGBoost

We can visualize individual trees from the fully boosted model that XGBoost creates using the entire dataset. XGBoost has a **plot_tree()** function which can be used to visualize the individual trees. Once we train a model using XGBoost, we can pass 
it to the **plot_tree()** function along with the number of trees we want to plot using the **num_trees** argument.
These plots provide insight into how the model arrive at its final decisions and what splits it made to arrive at those decisions.

===============================================================================

## 11. Feature selection with XGBoost

XGBoost provides a way to examine the importance of each feature in the original dataset within the model. 

It involves counting the number of times each feature is split on across all boosting trees in the model. Then we visualize the 
result as a bar graph, with the features ordered according to how many times they appear. XGBoost has a **plot_importance()** 
function that helps us to achieve this task. 

Then we can visualize the features that has been given the highest important score among all the features. Thus XGBoost provides 
us a way to do feature selection.

===============================================================================


## 12. The problem statement


===============================================================================


## 13. Results and conclusion

===============================================================================


## 14. References


The work done in this project is inspired from following books and websites:-

1.	Hands on Machine Learning with Scikit-Learn and Tensorflow by Aurélién Géron

2.	Introduction to Machine Learning with Python by Andreas C. Müller and Sarah Guido

3.	Udemy course – Machine Learning – A Z by Kirill Eremenko and Hadelin de Ponteves

4.	https://en.wikipedia.org/wiki/XGBoost

5.	https://xgboost.readthedocs.io/en/latest/

6.	https://xgboost.readthedocs.io/en/latest//parameter.html

7.	https://xgboost.readthedocs.io/en/latest/tutorials/param_tuning.html

8.	https://www.datacamp.com/community/tutorials/xgboost-in-python

9.	https://acadgild.com/blog/xgboost-python

10.	https://machinelearningmastery.com/gentle-introduction-xgboost-applied-machine-learning/ 

11.	https://www.analyticsvidhya.com/blog/2018/09/an-end-to-end-guide-to-understand-the-math-behind-xgboost/















