# Project Start Date
August 25, 2023

# Project Team Member

* Dr. Mrs. Jacinta Chioma Odirichukwu [Project Manager/Team Leader, Data Analysis, Machine Learning, IoT and Robotics]
* Simom Peter Chimaobi Odirichukwu    [Member, Health Officer, Health Data Montoring and Evaluation]
* Okorie Ignatius Chukwunonyerem [Member, Data Scientist/Data analyst]
* John Ugochukwu Nnoruka [Member, Machine learning Engineer, Django/Python developer, Research Assistant]
*  Nwanchukwu, David Chika [Member, Data Scientist/Data analyst]

  
# Project Aim

This project/paper aims at predicting ovulation day of women based on past historical data. Accurate fertility prediction is crucial. It provides important knowledge for making decisions about family planning. Women can efficiently plan pregnancies, check the health of their reproductive systems, take preventative measures to guarantee overall health with precise knowledge of lengths of their menstrual cycles. 


# Problem Statement

The variability of menstrual cycles, which can be influenced by a range of factors including stress, changes in weight, and certain medical issues, is one of the major obstacles in designing a Machine Learning algorithm for period tracking. Women seeking to optimize conception require a precise and dependable fertility prediction system capable of identifying their most fertile days with precision. 
The existing research,  suggests, future research in this field may concentrate on improving developed models through the incorporation of extra data sources, such as menstrual symptoms and hormonal data, and through the execution of more extensive investigations to verify conclusions in the existing system.


# Generating model artifacts
This model is trained using jupyter notebook
Overview of Jupyter Notebook
Go to command prompt and type: pip install jupyter I believe, you know how to invoke your command propmt from your Windows. Simply type: cmd in the search bar and press enter key. Your command prompt will show up.

Run the jupyter notebook from command prompt by typing: jupyter notebook. Dataset was viewed using Python Pandas, dataframe

The libraries need to be installed using the following commands in Jupyter Notebook: !pip install numpy !pip install pandas !pip install scikit-learn !pip install matplotlib !pip install seaborn

Functions of the libraries used

Pandas: A Python library called Pandas. Data analysis is done using Pandas. Python's Pandas package is used to manipulate data sets. It offers tools for data exploration, cleaning, analysis, and manipulation. Wes McKinney came up with the name "Pandas" in 2008, and it refers to both "Panel Data" and "Python Data Analysis."

Numpy: Large, multi-dimensional arrays and matrices can be supported with the Python package Numpy. Additionally, it offers a vast selection of mathematical functions. Listed here are a few of NumPy's fundamental operations: a. Creating arrays b. Array indexing c. Array slicing d. Data types e. Copy vs. View f. Array shape g. Array reshape

Matplotlib: Matplotlib is easy to use and an amazing visualizing library in Python. It is built on NumPy arrays, designed to work with the broader SciPy stack, and consists of several plots like lines, bars, scatters, histograms, etc.

Seaborn

Seaborn is a Python data visualization library that helps to visualize the statistical relationships between variables in a dataset. Its plotting functions operate on dataframes and arrays containing whole datasets and internally perform the necessary semantic mapping and statistical aggregation to produce informative plots. Seaborn aims to make visualization the central part of exploring and understanding data. It provides dataset-oriented APIs so that we can switch between different visual representations for the same variables for a better understanding of the data.

scikit-learn
Scikit-learn is a Python package that provides a selection of efficient tools for machine learning and statistical modeling. It includes algorithms for classification, regression, clustering, and dimensionality reduction. It is an open-source library that is reusable and easy to interface with other scientific packages. Scikit-learn is used to create and evaluate machine learning models of various types.

tensorflow
TensorFlow is an open-source machine learning framework developed by the Google Brain team. It is one of the most popular and widely used deep learning frameworks. TensorFlow is designed to facilitate the development and deployment of machine learning models, particularly neural networks, for a wide range of applications.

key features and components of TensorFlow:


keras
Keras is an open-source deep learning framework for building and training artificial neural networks. It was originally developed as an independent project but has been integrated into TensorFlow as of TensorFlow 2.0. Keras provides a high-level, user-friendly interface for designing and training neural networks, making it easier for both beginners and experts to work with deep learning models.

Key features of Keras include:
1. Flexible Architecture: TensorFlow provides a flexible and modular architecture for designing and training machine learning models. It allows you to define complex neural network architectures with ease.

2. Numerical Computation: TensorFlow is particularly known for its efficient numerical computation capabilities. It can leverage hardware acceleration (e.g., GPUs and TPUs) for faster training and inference.

3. High-Level APIs: TensorFlow offers high-level APIs like Keras, which allow users to quickly create and train neural networks with minimal code.

4. Low-Level APIs: For more advanced users, TensorFlow provides lower-level APIs that give fine-grained control over model development and training.

5. TensorBoard: TensorFlow includes TensorBoard, a powerful tool for visualizing and monitoring the training process and model performance. It helps with debugging and optimizing models.

6. AutoGraph: TensorFlow's AutoGraph feature can automatically convert Python code into graph operations, making it easier to work with dynamic computation graphs.

7. Distributed Computing: TensorFlow supports distributed computing, allowing you to train models on multiple machines or accelerators simultaneously.

8. SavedModel Format: TensorFlow models can be saved in the SavedModel format, making it easy to deploy models across various platforms and languages.

9. TensorFlow Serving: TensorFlow Serving is a part of TensorFlow that simplifies model deployment and serving for production applications.

10. Community and Ecosystem: TensorFlow has a large and active community, extensive documentation, and a wealth of pre-trained models and resources available.

TensorFlow is widely used in various domains, including computer vision, natural language processing, speech recognition, and reinforcement learning. It's employed by researchers, engineers, and data scientists to develop machine learning models and deep neural networks for a wide range of applications.

TensorFlow has undergone several versions and significant updates, with TensorFlow 2.x being a major release that introduced a more user-friendly and integrated experience, including the tight integration of the Keras high-level API. This has made TensorFlow more accessible to developers and has contributed to its continued popularity.




1.User-Friendly API: Keras offers a simple and intuitive API for building and training neural networks, which abstracts many of the complexities of deep learning. It allows for easy model prototyping and experimentation.

2. Modularity: Models in Keras are constructed as a sequence of layers, making it easy to assemble complex architectures. You can stack layers and connect them with various activation functions and other operations.

3. Wide Range of Applications: Keras supports a variety of neural network architectures, including feedforward, convolutional, recurrent, and more. This makes it suitable for a wide range of tasks, from image classification to natural language processing.

4. Backends: Keras is capable of running on top of different deep learning frameworks, including TensorFlow, Theano, and CNTK. As of TensorFlow 2.0, Keras is tightly integrated with TensorFlow.

5. Extensibility: You can easily create custom layers, loss functions, and metrics in Keras. This makes it suitable for research and experimentation.

6. Visualization Tools: Keras provides tools for visualizing your model's architecture and performance, which can be helpful for debugging and understanding your network.

7. Pre-trained Models: Keras includes pre-trained models for various tasks, such as image classification, which can be fine-tuned for specific applications.

To use Keras, you typically need to install TensorFlow or one of the other supported backends. Then, you can import Keras and start building and training your neural network models.



To install the libraries in jupyter notebook, use the following command
!pip install numpy
!pip install pandas
!pip install scikit-learn
!pip install matplotlib
!pip install seaborn
!pip install tensorflow
!pip install keras


# Import the necessary librabries

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, r2_score
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import HuberRegressor
from sklearn.linear_model import Lars
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoLars
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import OrthogonalMatchingPursuit
import lightgbm as lgb
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.ensemble import AdaBoostRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
import statsmodels.api as sm
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

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

In training the model, twenty-two (22) machine learning algorithms were used. The dataset was split into a training set and a testing set. The ‘X_train’ represent 85% of the training dataset. The ‘X_test’ represent the 15% for the testing the model. ‘y_train’ represent the 85% of the target that was used to train the model. ‘y_test’ represent the 15% of the target dataset that was used to test the model. Thus;
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, random_state = 42)

## Step 6: Train the model

In training the model, twenty two (22) machine learning algorithm were used.


## Step 7: Testing the trained model
The model was tested using the test set


## Step 8: Evaluating the model using some evaluation parameters.

The metrics utilized to assess the proposed system's accuracy include Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), R-squared (R2 score), Mean Absolute Percentage Error (MAPE), and Explained Variance Score (EVS).

## Step 9: Iteration/Pickling the model
The researchers experimented with different techniques to determine the model performs better.
Decision having performed better based on R2 score was pickled.

# Deploying the model Using Django
The model was deployed in real time using the decision tree pickled model. The Decision Tree Regressor  model was deployed using python and django framework to create a real-time predictive app that predicts the ovulation of a woman. Find deployed folder attached



