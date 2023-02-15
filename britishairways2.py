# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 02:14:13 2023

@author: Champo
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_csv("customer_booking.csv", encoding="ISO-8859-1")
print(df.head())

del df['booking_origin']
del df['route']
del df['trip_type']

df["flight_day"].unique()

mapping = {
    "Mon": 1,
    "Tue": 2,
    "Wed": 3,
    "Thu": 4,
    "Fri": 5,
    "Sat": 6,
    "Sun": 7,
}

df["flight_day"] = df["flight_day"].map(mapping)

df["flight_day"].unique()

mapping2 = {
    "Mobile": 1,
    "Internet": 2,
}

df["sales_channel"] = df["sales_channel"].map(mapping2)


mapping3 = {
    "Round Trip": 1,
    "One Way":2,
    "Circle Trip": 3,
}


print(df.describe())


x = df.iloc[:, :-1]
y = df.iloc[:, -1]


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#x=np.arange(0,len(x),1)

# Plot the data points
plt.plot(df['num_passengers'], df['booking_complete'], marker='x', c='r')
plt.title("Customer Booking")
# Set the y-axis label
plt.ylabel('Booking Complete')
# Set the x-axis label
plt.xlabel('num_passengers')
plt.show()

from sklearn.ensemble import RandomForestClassifier



clf = RandomForestClassifier(n_estimators = 100)  

# fit function is used to train the model using the training sets as parameters
clf.fit(x_train, y_train)
  
# performing predictions on the test dataset
y_pred = clf.predict(x_test)
  
# metrics are used to find accuracy or error
from sklearn import metrics  
print()
  
# using metrics module for accuracy calculation
print("ACCURACY OF THE MODEL: ", metrics.accuracy_score(y_test, y_pred))

import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# View confusion matrix for test data and predictions
confusion_matrix(y_test, y_pred)


# Get and reshape confusion matrix data
matrix = confusion_matrix(y_test, y_pred)
matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]

# Build the plot
plt.figure(figsize=(16,7))
sns.set(font_scale=1.4)
sns.heatmap(matrix, annot=True, annot_kws={'size':10},
            cmap=plt.cm.Greens, linewidths=0.2)

# Add labels to the plot
class_names = ['num_passengers', 'sales_channel', 'trip_type', 
               'purchase_lead', 'length_of_stay', 'flight_hour',    
               'flight_day','route','booking_origin','wants_extra_baggage','wants_preferred_seat','wants_in_flight_meals'
               'flight_duration','booking_complete']
tick_marks = np.arange(len(class_names))
tick_marks2 = tick_marks + 0.5
plt.xticks(tick_marks, class_names, rotation=25)
plt.yticks(tick_marks2, class_names, rotation=0)
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix for Random Forest Model')
plt.show()