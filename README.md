## Intent-Classification-Model-Flask-Heroku

This is a example project to explain how to train Machine Learning (ML) Model for intent classification based on text utterances. Moveover, how to deploy on production optimized machine learning model using Flask API and Heroku cloud.

### Prerequisites
You must have Scikit Learn, NumPy, Pandas and Flask installed.

### Project Structure
This project has two parts :
1. train_nlp_model.py - This script contains the procedure for the training of the machine learning model. It is divided into five parts: Load text utterances data, Natural Language Processing (stop words, lemmatization, stemming, among others), feature extraction using tf-idf, training and development of the ML model, and save final ML model.  

contains four  code for the machine learning our Machine Learning model to predict employee salaries absed on trainign data in 'hiring.csv' file.
2. app.py - This contains Flask APIs that receives employee details through GUI or API calls, computes the precited value based on our model and returns it.
3. request.py - This uses requests module to call APIs already defined in app.py and dispalys the returned value.
4. templates - This folder contains the HTML template to allow user to enter employee detail and displays the predicted employee salary.

### Running the project
NOTE: Ensure that you are in the project home directory. 

1. Create the machine learning model by running below command -
```
python model.py
```
This would create a serialized version of our model into a file model.pkl

2. Run app.py using below command to start Flask API
```
python app.py
```
By default, flask will run on port 5000.

3. Navigate to URL http://localhost:5000

You should be able to view the homepage as below :
![alt text](http://www.thepythonblog.com/wp-content/uploads/2019/02/Homepage.png)

Enter valid numerical values in all 3 input boxes and hit Predict.

If everything goes well, you should  be able to see the predcited salary vaule on the HTML page!
![alt text](http://www.thepythonblog.com/wp-content/uploads/2019/02/Result.png)

4. You can also send direct POST requests to FLask API using Python's inbuilt request module
Run the beow command to send the request with some pre-popuated values -
```
python request.py
```
