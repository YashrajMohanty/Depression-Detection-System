# Depression Detection System
## Overview
This is a web app called "ML for Mental Health". It uses machine learning and deep learning models to predict if a person is susceptible to depression using symptoms as input.  
Uses a variety of machine learning models like KNN, naive-bayes algorithm, random forest, logistic regression, SVM, decision tree as well as a feed forward DNN.

## Contents
It consists of the following:
- **Frontend**: An interface to input the required information. It allows the user to answer questions about mental health aspects as well as choose the type of model to run inference on.
- **Backend**: Python-based backend that accepts input, runs inference using the chosen model, and returns the results to the API. Primarily uses sklearn and keras.
- **API**: A flask API that handles requests from the frontend and passes the information to the backend. On receiving results from the backend, returns it to the frontend.


## Usage
Configure server url in the html and js files.  
Launch on a browser with URL: `<server-url>/home`.


## Results/Gallery
Here's a glimpse of the home page.


![Website home page image](github_images/WebHome.png)


Here's where we input the information and receive the results.


![Website input page image 1](github_images/WebDepDet.png)


### Other details
- Hovering over the machine learning section on the home page reveals the link to the form page.
- Clicking on the nav bar icon in the form page returns you to the home page.