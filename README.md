# Real-Estate-Price-Prediction 
* It is a Advanced Problem of Regression which requires advanced techniques of feature engineering, feature selection and extraction, modelling, model evaluation, and Statistics.
 # Goal
 My goal for this project is to build an end to end solution or application that is capable of predicting the house prices better than individuals. Another motivation for this project is to implement similar solution at my workplace and help Investments and Residential team make data driven decisions.


# Resources and Tech used in this project
**Python:** 3.8
* Numpy and Pandas for data cleaning
* Matplotlib for data visualization
* Sklearn for model building
* Google Colaboratory Notebook
* Python flask for http server
* HTML/CSS/Javascript for UI
* AWS EC2 For deployement of web application and flask server

# Explaining Project with Steps
**Data loading and cleaning:**
* Download the data from Kaggle and load into csv file using pandas.
* Started with Droping unwanted columns and Filling na values with mean for numeric      columns and delete(low no of rows with compare to the size of data) rows having na values in categorical columns.
* Some columns are skewed so use lambda function and write some function for tranfsformation.set range values into average etc. 
* Code - https://github.com/mr-robot-hack/Real-Estate-Price-Prediction/blob/master/model_s/Home_price_model.ipynb.py

**Analysing The Data and removing outlier:**
* Analysing data using matplotlib, pandas, numpy and some statistics concepts to remove outliers and unnecessary columns.
* Removal outliers using standard deviation and mean.
* For some features like BHK,Location write function to remove outliers
* Added some features like Price/sqft.
* code - https://github.com/mr-robot-hack/Real-Estate-Price-Prediction/blob/master/model_s/Home_price_model.ipynb.py

**Preparing data for model building:**
* Used One hot encoding for location column using pandas.get_dummies method 
* Split our data into training and testing data set using train_test_split method
* code - https://github.com/mr-robot-hack/Real-Estate-Price-Prediction/blob/master/model_s/Home_price_model.ipynb.py

**Model Building:**
* Train LinearRegression, Lasso, DecisionTreeRegressor model on training dataset used ShuffleSplit and cross_val_score and check scores for each model.
* To find Best fit model for our dataset used GridSearchCV for finding best perameter for our model and check scores.
* For my dataset Linear_regression is the best model with 85% accuracy
* Save model using pickle library
* Model.pickle - https://github.com/mr-robot-hack/Real-Estate-Price-Prediction/blob/master/model_s/model1.pickle
 
 **Flask server and web application:**
 * write a python flask server that uses the saved model to serve http requests.
 * Develop a website built in html, css and javascript that allows user to enter home square ft area, bedrooms etc and it will call python flask server to retrieve the predicted price.
 *  Website code - https://github.com/mr-robot-hack/Real-Estate-Price-Prediction/tree/master/client
 *  flask code - https://github.com/mr-robot-hack/Real-Estate-Price-Prediction/tree/master/server
 
 **Deployment of flask server and web application on AWS:**
 * Deploy machine learning model to production on amazon AWS EC2 instance
 * For AWS EC2 use ubuntu server on which I deploy web application as well as python flask server
 * Using nginx reverse proxy /api requests will be routed to python flask server running on same machine.
 * Reference - https://towardsdatascience.com/simple-way-to-deploy-machine-learning-models-to-cloud-fd58b771fdcf
 

# References
**Data:** https://www.kaggle.com/amitabhajoy/bengaluru-house-price-data
**Project Tutorial:** https://www.youtube.com/watch?v=rdfbcdP75KI&list=PLeo1K3hjS3uu7clOTtwsp94PcHbzqpAdg
**Data Analysis:** https://intellipaat.com/blog/python-for-data-science/



