## Intent-Classification-Model-Flask-Heroku

This is a example project to explain how to train Machine Learning (ML) Model for intent classification based on text utterances. Moveover, how to deploy on production optimized machine learning model using Flask API and Heroku cloud.

### Prerequisites
You must have Scikit Learn, NumPy, Pandas and Flask installed.

### Project Structure
This project has two parts :
1. train_nlp_model.py - This script contains the procedure for the training of the machine learning model. It is divided into five parts: Load text utterances data, Natural Language Processing (stop words, lemmatization, stemming, among others), feature extraction using tf-idf, training and development of the ML model, and save final ML model.   The training data is located in  'knowledgebase.xlsx' file.
2. app_nlp.py - This contains Flask APIs that receives utterance through GUI or API calls. Here, it is used nlp model and vocabulary saved in train_nlp_model.py 
3. request.py - This uses requests module to call APIs already defined in app_nlp.py and displays the returned value.
4. templates - This folder contains the HTML template to allow user to enter utterance and displays the predicted intent.

### Running the project
NOTE: Ensure that you are in the project home directory. 

1. Train the ML model by running below command -
```
python train_nlp_model.py
```
This would create a serialized version of our ML model, Knowledge base vocabulary, and intent names into the files nlp_model.pkl, knowledgebase_vocabulary.pkl, and intent_names.pkl, respectively.

2. Run app_nlp.py using below command to start Flask API
```
python app_nlp.py
```
By default, flask will run on port 5000.

3. Navigate to URL http://127.0.0.1:5000/ 

4. You can also send direct POST requests to FLask API using Python's inbuilt request module
Run the beow command to send the request with some pre-popuated values -
```
python request.py
```

5. [Create heroku account](https://signup.heroku.com/)
6. [Create heroku app](https://dashboard.heroku.com/new-app)
7. Download and install the [Heroku CLI](https://devcenter.heroku.com/articles/heroku-cli)
8. If you haven't already, log in to your Heroku account and follow the prompts to create a new SSH public key.
```
$ heroku login
```
9. Initialize a git repository in a new or existing directory
```
$ git init
$ heroku git:remote -a {your-app-name}
```
10. Deploy your application
```
$ git add .
$ git commit -am "make it better"
$ git push heroku master
```
