# instacart_market_basket_analysis

**Key Findings:** The tuned Light GBM model resulted in the highest F1-Score and the engineered up_reorder_ratio feature was the most important.  This feature represents the ratio of the number of times a user purchased a specific product over the number of orders placed since they first ordered it. 

**Table of Contents**

•	Business Problem

•	Data Source & Description

•	Exploratory Data Analysis

•	Modelling Approach 

•	Results

**Business Problem**

Instacart is a grocery delivery and pick-up service in the U.S. and Canada. Through a website and/or mobile app, customers can place their grocery orders from various participating retailers.  

In 2017, Instacart released relational datasets describing customers' order patterns. These included over 3.4 million orders with 50,000 products to choose from for approximately 200,000 users. The objective is to predict which products will be in the user's next order. 


**Data Source & Description**

The datasets can be found on Kaggle using this link: https://www.kaggle.com/c/instacart-market-basket-analysis/data

A description of the datasets are as follows:

•	Departments: Grocery stores are split into various sections called departments. In this dataset, there are 21 different departments each with their own corresponding department id. Examples of these departments include frozen, bakery, produce, alcohol, pets, etc. 

•	Aisles: Departments are divided into aisles that are organized for ease of shoppers to find their desired products. There are 134 aisles each with their own corresponding aisle id. Examples of the aisles include specialty cheeses, energy granola bars, instant foods, etc. 

•	Order_products_prior: This dataset contains 3 to 99 orders from the sample of Instacart users. Information provided includes the products that were purchased in each order, the placement in the cart and whether the product was reordered. 

•	Order_products_train: This dataset contains 1 order for each user. Information provided is similar to the orders_products_prior dataset. 

•	Orders: This dataset specifies which set (prior, train, test) an order belongs. For the Kaggle submission, reordered items were predicted on the test set. 

•	Products: This dataset specifies all the 49,688 items that were sold at the grocery.

**Exploratory Data Analysis**

To better understand users' past purchasing behavior, visualizations were created using the prior_dataset.  

The bar chart below shows the reordered products by department sorted from highest to lowest. Not surprisingly, produce, dairy eggs, snacks, beverages, and frozen are at the top.

![](https://github.com/joanneevangelista/instacart_market_basket_analysis/blob/main/images/Count_By_Department.PNG)

The next graph models the number of orders by hour of the day. As we can see here, most users order later in the morning to early evening. 

![](https://github.com/joanneevangelista/instacart_market_basket_analysis/blob/main/images/Order_Hour_Of_Day.PNG)

The word cloud visualization was created to capture how frequently the past products appear in users' orders. In this case, Bananas are the most popular followed by organic products. 

![](https://github.com/joanneevangelista/instacart_market_basket_analysis/blob/main/images/Word%20Cloud.PNG)

**Modelling Approach**

_Data Preparation_

In order to work with the data and engineer new features, the relational datasets were joined based on identifiers for user, product or order. Refer to the diagram below:

![](https://github.com/joanneevangelista/instacart_market_basket_analysis/blob/main/images/Relational_Map.PNG)

Each of the features in the raw datasets were easy to interpret and there were very little missing values. Only a few data points were missing in the days_since_prior_order feature, however these represent the first order for a specific customer. These missing data points were treated as nulls and did not cause any issues.   

![](https://github.com/joanneevangelista/instacart_market_basket_analysis/blob/main/images/Missing_Values.PNG)

Imbalance of the target class is an important consideration as it can cause the model to over fit by misclassifying the minority class. In this case, there was not much imbalance in the data. For both the prior and training datasets, the proportion of reordered items in comparison to not reordered was just under 60% for both. Thus, it was not critical to implement techniques such as ADASYN and Random Oversampling. Below are visual representations of the count of ‘reordered’ items, the target feature, in the prior and train data.  

![](https://github.com/joanneevangelista/instacart_market_basket_analysis/blob/main/images/Imbalance.PNG)

The orders_products_prior dataset was merged with the orders dataset to have more information to work with when creating new features. Because of the size of the datasets, DataFrames were created for each feature and then merged into one called new_features. By creating features one-by-one, memory storage was saved.

The train set, called new_features_train, was created by filtering the orders dataset for “train” and merged with the new features DataFrame and orders_prodicts_train DataFrame. To save processing power and time, a sample of 50,000 instances in the train set was used to train the various models. Finally the test set, called new_features_test, was created by filtering the orders dataset for “test” and merged it with the new features DataFrame.

_Feature Engineering_

To help the machine learning algorithms learn which products an Instacart user will purchase in their next order, 13 new features were created, each of which fall into one of 3 categories:

![](https://github.com/joanneevangelista/instacart_market_basket_analysis/blob/main/images/Feature_Engineering.PNG)

_Model Selection_

The algorithms used to make the predictions were Random Forest, XGBoost and Light GBM which are decision-tree based models. Although it was more time consuming, stacking various models was considered to see if it would improve the F1-score. 

Eecision-tree based models were selected as opposed to other options because they have faster training and prediction speed than neural networks or support vector machines. Additionally, these models have higher accuracy on unstructured and structured data compared to K nearest neighbors and logistic regression. Lastly, the algorithms are designed to make efficient use of hardware resources which is especially useful when working with big data frames and memory storage is a constraint. 

First, the new_features_train was split into a train and test set. Then baseline models were trained for the chosen algorithms using basic parameters. After training the model, they were run on the test set to derive the model results as well as the new_features_test data (aka prediction dataset) to obtain the Kaggle results. The evaluation metric used to assess the performance was the F1 score. The results below show that the XGBoost baseline model performed the best on the model test set and on the prediction data set. 

![](https://github.com/joanneevangelista/instacart_market_basket_analysis/blob/main/images/Results.PNG)

As evidenced by the discrepancy in results between the model and the Kaggle score, overfitting of the data appears to be a problem. A common issue with decision trees is that they have a tendency to overfit the data. To deal with this, depth of the tree and size of the leaf parameters was tuned which helped the performance of the model and also reduced the overfitting.

_Model Description and Hyper-parameter Tuning_

The GridSearchCV method in Python was used to tune the random forest, XGBoost and Light GBM parameters to identify an optimal combination. A few combinations for the ‘n_estimators’, ‘max_depth’, ‘num_leaves’ and ‘learning_rate” were explored and the final combination of hyperparameters that were used in the model are as follows: 

![](https://github.com/joanneevangelista/instacart_market_basket_analysis/blob/main/images/Hyperparameters.PNG)

The stacking model was comprised of the top 3 performing algorithms: decision tree, XGBoost and gradient boosting. 

_Threshold Selection_

As part of the modelling process, a threshold probability was used to determine whether a product would be reordered. To do this, the training data was further split into a train and test set. The model learned from the train set, then was run on the test set and the F1 score was calculated. The optimal threshold was determined based on which gave the highest F1 score. In this case for Light GBM, the best model, the max threshold is 0.20.

![](https://github.com/joanneevangelista/instacart_market_basket_analysis/blob/main/images/Threshold.png)

_Tech Stack_

Most of the work was performed in Python. The packages used in Python include: pandas, numpy, matplotlib, seaborn, sklearn, lightgbm, RandomForestClassifer, xgboost, scipy, mlxtend.frequent_patterns. 

**Results**

_Model Performance Results_

After hyperparameter tuning, the models with the highest F1 score for each algorithm were used to make the predictions. Before making predictions, the new_features_test data was transformed into the same shape by repeating the data pre-processing steps. 

In order to format the predictions for the final submission, a dictionary was created which was used to map unique keys to values. Order_id was the index and the values were all the products that were predicted to be reordered. The final results of each of model is presented below with Light GBM as the best one with an F1 score of 0.37206

![](https://github.com/joanneevangelista/instacart_market_basket_analysis/blob/main/images/Final_Results.PNG)

As demonstrated in the results, hyperparameter tuning not only resulted in higher F1 scores, it also reduced the overfitting in the data in comparison to the baseline models. Although the scores might appear low, they are relatively comparable to those on the Kaggle Leaderboard. In fact, the highest score was just 0.40972.

_Discussion and Interpretation_

Below is a feature importance chart of the best model which was Light GBM. In this case, the up_reorder_ratio was the most important. It is worth noting that importance scores are calculated by a predictive model that has been fit on the dataset. Thus, depending on the model used and the hyperparameters chosen, the features with the highest importance may be different.

![](https://github.com/joanneevangelista/instacart_market_basket_analysis/blob/main/images/Feature_Importance.PNG)

Another interesting insight is that stacking resulted in the lowest F1 score on the test and Kaggle submissions in comparison to only using one model. Because ensemble machine learning algorithms learn how to optimally combine the predictions from multiple algorithms, intuitively it would be expected to produce the best result. This may be because the best stacking combination was not used or better parameters could have been chosen. 











