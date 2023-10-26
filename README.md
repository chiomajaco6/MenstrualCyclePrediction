# Project Start Date
August 25, 2023

# Project Team Member

* Dr. Mrs. Jacinta Chioma Odirichukwu [Project Manager/Team Leader, Data Analysis, Machine Learning, IoT and Robotics]
* Simom Peter Chimaobi Odirichukwu    [Member, Health Officer, Health Data Montoring and Evaluation]
* Okorie Ignatius Chukwunonyerem
[Member, Data Scientist/Data analyst]
* John Ugochukwu Nnoruka [Member, Machine learning Engineer, Django/Python developer, Research Assistant]

  
# Project Aim

This project/paper aims at predicting the ovulation day of women based on past historical data. Accurate fertility prediction is crucial. It provides important knowledge for making decisions about family planning. Women can efficiently plan pregnancies, check the health of their reproductive systems, take preventative measures to guarantee overall health with precise knowledge of ovulation day of their menstrual cycles. 


# Problem Statement

The variability of menstrual cycles, which can be influenced by a range of factors including stress, changes in weight, and certain medical issues, is one of the major obstacles in designing a Machine Learning algorithm for period tracking. Women seeking to optimize conception require a precise and dependable fertility prediction system capable of identifying their most fertile days with precision. 
The existing research, as the current research review suggests, future research in this field may concentrate on improving developed models through the incorporation of extra data sources, such as menstrual symptoms and hormonal data, and through the execution of more extensive investigations to verify conclusions in the existing system.


# The Existing System

<div style="text-align: justify"> 

Using data produced by a predetermined model, the existing system used machine learning techniques to forecast the menstrual cycle. This may result in fresh perspectives and understandings in the study of menstrual cycle prediction. Additionally, the results of the current system imply that machine learning models can forecast the menstrual cycle phase reliably and with little error. The findings have significant health implications for women and may be used to guide decisions on individualized reproductive health, including family planning and fertility treatment. 
The AutoRegressive Integrated Moving Average, Huber Regression, Lasso Regression, Orthogonal Matching Pursuit, and Long Short-Term Memory Network are some of the time series forecasting algorithm approaches that existing system used.
Additionally, the algorithms can be trained on individualised data, making it possible to forecast menstrual cycle trends specifically for each user. Since conventional prediction techniques may not be as effective for women with irregular periods, this could be very helpful to them.

</div>

# Algorithms/Techniques of the Existing System
* Model		                        MAE		        MSE		    RMSE

* LSTM 		                        3.4000 	    4.2895 	    2.0711 
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
* K Neighbors Regressor           5.6383      45.9650     6.5151 
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

## Step 2 : Data Preprocessing
Install and import the necessary libraries.
The EDA done is found in the attached code and the branch called EDA


## Step 3: Load the dataset

The data was loaded as thus;
df = pd.read_csv("C:/Users/hp/Desktop/MLs/DeployModelOvuLength/FedCycleData.csv")

## Step 4: Feature Engineering
The feature engineering was done on the dataset. Out of the 80 columns contained in the dataset, only 12 columns were extracted. These columns include: 
CycleNumber', 'LengthofCycle', 'LengthofLutealPhase',  'TotalNumberofHighDays', 'TotalNumberofPeakDays', 'UnusualBleeding', 'PhasesBleeding', 'IntercourseInFertileWindow', 'Age', 'BMI', 'Method', 'EstimatedDayofOvulation'
The columns were then splitted into feature columns and target columns. The feature columns include: ‘CycleNumber', 'LengthofCycle', 'LengthofLutealPhase',  'TotalNumberofHighDays', 'TotalNumberofPeakDays', 'UnusualBleeding', 'PhasesBleeding', 'IntercourseInFertileWindow', 'Age', 'BMI', 'Method', 
 While the target column is ‘EstimatedDayofOvulation’. The feature columns were used to predict the Ovulation day of a woman.

## Step 5: Split the dataset into training and testing set

Th
## Step 6: Train the model

In training the model, twenty two (22) machine learning algorithm were used.


## Step 7: Testing the trained model
The model was trsted using the test set



## Step 8: Evaluating the model using some evaluation parameters.

The metrics utilized to assess the proposed system's accuracy include Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), R-squared (R2 score), Mean Absolute Percentage Error (MAPE), and Explained Variance Score (EVS).

## Step 9: Iteration/Pickling the model
The researchers experimented with different techniques to determine the model performs better.
Decision having performed better based on R2 score was pickled.

# Deploying the model Using Django
The model was deployed in real time using the decision tree pickled model. Find deployed folder attached

# Proposed System Result 

