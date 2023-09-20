# Project Start Date
August 25, 2023

# Project Team Member

<div style="text-align: justify">  
* 1. Dr. Mrs. Jacinta Chioma Odirichukwu [Project Manager/Team Leader, Data Analysis, Machine Learning, IoT and Robotics]
* 2. Simom Peter Chimaobi Odirichukwu    [Member, Health Officer, Health Data Montoring and Evaluation]
  
# Project Aim

This project/paper aims at predicting menstrual cycle length of women based on past historical data. Accurate fertility prediction is crucial. It provides important knowledge for making decisions about family planning. Women can efficiently plan pregnancies, check the health of their reproductive systems, take preventative measures to guarantee overall health with precise knowledge of lengths of their menstrual cycles. 


# Problem Statement
<div style="text-align: justify">  

The variability of menstrual cycles, which can be influenced by a range of factors including stress, changes in weight, and certain medical issues, is one of the major obstacles in designing a Machine Learning algorithm for period tracking. Women seeking to optimize conception require a precise and dependable fertility prediction system capable of identifying their most fertile days with precision. However, current systems rely on simplistic algorithms that fail to encompass the intricate and individualized characteristics of each woman's menstrual cycle. the current system The data may not accurately reflect the general population because they came from a self-selected group of women who were willing to spend money on a commercially available fertility medication. 
The existing research, as the current research review suggests, suggests future research in this field may concentrate on improving developed models through the incorporation of extra data sources, such as menstrual symptoms and hormonal data, and through the execution of more extensive investigations to verify conclusions in the existing system.

</div>

# The Existing System

<div style="text-align: justify"> 

Using data produced by a predetermined model, the existing system used machine learning techniques to forecast the menstrual cycle. This may result in fresh perspectives and understandings in the study of menstrual cycle prediction. Additionally, the results of the current system imply that machine learning models can forecast the menstrual cycle phase reliably and with little error. The findings have significant health implications for women and may be used to guide decisions on individualized reproductive health, including family planning and fertility treatment. Additionally, the algorithms can be trained on individualised data, making it possible to forecast menstrual cycle trends specifically for each user. Since conventional prediction techniques may not be as effective for women with irregular periods, this could be very helpful to them.

</div>

# Algorithms/Techniques of the Existing System
* Model		                    MAE		    MSE		    RMSE

* LSTM 		                    3.4000 	    4.2895 	    2.0711 
* ARIMA                           7.3000      7.7964      2.7922 
* Orthogonal Matching Pursuit     5.3373      41.1000     6.1243 
* Elastic Net                     5.3686      41.1295     6.1588 
* Huber Regressor                 5.5458      43.5776     6.2851 
* Ridge Regression                5.4826      42.5662     6.2523 
* Linear Regression               5.4914      42.7152     6.2604 
* Least Angle Regression          5.4914      42.7152     6.2604 
* Lasso Regression                5.4554      42.2978     6.2402 
* Lasso Least Angle Regression    5.4554      42.2978     6.2402 
* Dummy Regressor                 5.4702      41.5427     6.2063 
* Light Gradient Boosting Machine 5.4702      41.5427     6.2063 
* Bayesian Ridge                  5.5910      44.0523     6.3834 
*K Neighbors Regressor           5.6383      45.9650     6.5151 
* Passive Aggressive Regressor    5.6844      48.3409     6.5683 
* Random Forest Regressor         6.2947      59.3721     7.3739 
* AdaBoost Regressor              6.2856      56.1740     7.3034 
* CatBoost Regressor              7.2368      76.8913     8.4820 
* Gradient Boosting Regressor     7.2034      75.0468     8.3428 
* Decision Tree Regressor         7.1083      75.7250     8.3555 
* Extra Trees Regressor           7.1378      69.3403     8.1291 
* Extreme Gradient Boosting       7.5025      77.0432     8.5549 

# Proposed System Design

## Step 1: Data Collection

Menstrual cycle dataset was collected from kaggle via https://www.kaggle.com/datasets/nikitabisht/menstrual-cycle-data
Based on the existing system suggestion, we created this google form; https://docs.google.com/forms/d/e/1FAIpQLSfAeWexULsDXC5ZmZgLGPDVaE-RAKao4BnUQA-8-XoWRmbx7g/viewform
to gather individualised data basically within Nigeria Environment

## Step 2 : Data Preprocessing



## Step 3: Training the Model



## Step 4: Import the dataset



## Step 5: Splitting the dataset


## Step 6: Split the dataset into training and testing set


## Step 7: Develop the model


## Step 8: Train the model


## Testing the trained model using the X_train set



## Step 9: Evaluating the model using some evaluation parameters.


## Step 10: Pickling the model


# Deploying the model Using Django


# Feedbacks
[01:28, 9/20/2023] +234 803 261 8951: There is a little concern in predicting menstural cycle
[01:30, 9/20/2023] +234 803 261 8951: Menstruations are not precise. And predictions are not precise too. This variability seems to cancel the relevance of the research because what you are attempting to predict is already a variability. Significance of such projects are little
[01:31, 9/20/2023] +234 803 261 8951: Do you have any other idea to mitigate this? Or how precise do you expect the result to be here?
[01:32, 9/20/2023] DR. JACO: You are very right sir
[01:33, 9/20/2023] DR. JACO: Some authors have conducted research, I am trying to validate the said performance based on their future suggestion to contribute to the existing knowledge.
[01:35, 9/20/2023] DR. JACO: Menstrual cycle is not the only imprecise situation. Stock prediction is too and life itself is also fuzzy too
[01:39, 9/20/2023] DR. JACO: Fuzzy logic is computer technique developed to capture imprecised situation
