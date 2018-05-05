# Predictive Analysis of spending money on healthy food
Predictive Analysis for likelihood of spending more money on healthy food for Young-People survey dataset

Dataset used for the project: https://www.kaggle.com/miroslavsabo/young-people-survey/
It’s a survey performed with young people on diverse topics including demographics. The data file (responses.csv) consists of 1010 rows and 150 columns (139 integer and 11 categorical) and it contain missing values; and continuous and discrete values.
Goal: Analysis and prediction of likelihood of a person to “pay more money for good, quality or healthy food”.

Machine Learning solutions used: 
1. Random Forest Classifier
2. SVM with different kernels (linear and rbf)
3. Gaussian Naïve Bayes

Reasoning for using Random Forest Classifier: Random Forest Classifier is ensemble algorithm. It creates a set of decision trees from randomly selected subset of training set. It then aggregates the votes from different decision trees to decide the final class. Aggregation of votes from different classifiers makes this classifier more generalized and thus it performs better on test data set. 

Reasoning for using SVM as classifier: Since our dataset is small and SVM is good at dealing with small data set as only support vectors are used to construct the separating hyperplane. Also, it’s easy to implement using Scikit-learn libraries which provides the flexibility to use different kernels with different params.

Reasoning for using Naïve Bayes classifier: It is easy and fast to predict class of test data set. Also, it performs well in multi class prediction.

Data preprocessing:

1. Missing values replacement with most frequent values: Missing values in the data creates issue while vectorizing the data to retrieve feature matrix. Its necessary to remove these missing values. If we simply delete those examples from the data, we might incur in heavy loss of the data (especially for data with small data size) which are important to train the model.

2. Data discretization of continuous values: Since almost all features in our data have discretized values expect few ones. We should discretize these features with continuous values as these can mislead the network in understanding the data.

3. Finding the correlation between target feature and rest all features: Since our data has large number of features and many of the features are not relevant with our target feature. Considering all the features will result in feature redundancy and feature explosion while training the model. Hence, its necessary to find the correlation between features to select the best features to create the feature matrix for training the model.

Machine learning techniques: 

1. Splitting strategy for train and test set: The fundamental goal of machine learning is to generalize the model beyond the data used for training the model. Evaluation of model is estimated on the quality of its pattern generalization for data the model has not been trained on, that is unknown future instances of data for prediction. Since we need to check the accuracy of the model before it is applied to predict future instances that doesn’t have target values, we need to use some of the data for which we know the target values. Model evaluation with same data on which it is trained on is meaningless as it will simply learn those data and predict its target values. We need to generalize the model for which we need to keep some data in safe, unseen from the model while training. This data which is kept aside can be used to evaluate the efficiency of model for predicting future instances as we have target values for this data and these are not known to the model. Hence by using this splitting strategy, we can achieve generalized model which can do well on future data instances.
In this project, 15% of the data was kept for test set which was kept unseen from the model till final evaluation and 85% of the data again got split in training and development to cross validate and tune the hyper-parameters. 

2. Used cross validation to tune the hyper-parameters: Tuning the hyper parameters is very important step in machine learning. Hyperparameter optimization is the problem of choosing a set of optimal hyperparameters for a learning algorithm. These are not learnt during training of the model. These needs to passed to the algorithm. Deciding on the best parameter for given dataset and model is difficult. Hence, we do hyper-parameter tuning to achieve a better model.

3. Ensemble method: Ensemble modeling is a powerful way to improve the performance of the model. It combines several base models to produce one optimal predictive model. These are meta-algorithms as it combines several machine learning techniques into one predictive model in order to decrease variance, bias and improve predictions. This project uses Random Forest Classifier which is an ensemble algorithm as one of the classifier to get the optimal predictive model.

4. Balancing all the labeled classes in the data by oversampling the data: Imbalanced classes are bias in the data which can mislead the model while training. Its important to properly adjust metrics to adjust for the goal. Its necessary otherwise algorithm may end up optimizing for a meaningless metric in the context of given problem. There are various techniques that can be used to mitigate class imbalance. One is sampling which is simply to balance them, either by oversampling the instances of minority of class or under sampling the instances of majority class. Since our dataset is small, under sampling would not be a good approach in this case. Oversampling the data can fix the imbalance in the data without any loss of the data.

Success evaluation: Based on precision, recall and overall accuracy. 

References: 
http://scikit-learn.org/stable/auto_examples/model_selection/plot_validation_curve.html#sphx-glr-auto-examples-model-selection-plot-validation-curve-py, 
https://www.kaggle.com/jkokatjuhha/we-are-from-our-childhood,   
http://chaspari.engr.tamu.edu/wp-content/uploads/sites/147/2018/01/2_9-1.pdf
