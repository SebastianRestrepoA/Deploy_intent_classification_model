## Intent-Classification-Model-Flask-Heroku

This is a example project to explain how to train Machine Learning (ML) Model for intent classification based on text utterances. Moveover, how to deploy on production optimized machine learning model using Flask API and Heroku cloud.

### Prerequisites
You must have Scikit Learn, NumPy, Pandas and Flask installed.

### Project Structure

1. train_nlp_model.py - This script contains the procedure for the training of the machine learning model. It is divided into five parts: Load text utterances data, feature extraction using tf-idf, training and development of the ML model, and save final ML model.   The training data is located in  'KnowledgeBase.xlsx' file.
2. app_nlp.py - This contains Flask APIs that receives utterance through GUI or API calls. Here, it is used nlp model and vocabulary saved in train_nlp_model.py 
3. request.py - This uses requests module to call APIs already defined in app_nlp.py and displays the returned value.
4. templates - This folder contains the HTML template to allow user to enter utterance and displays the predicted intent.

### Running the project

1. Clone intent classification project in a local directory.
```
git clone https://github.com/SebastianRestrepoA/intent_classification_model_flask_heroku.git
```

2. Create enviroment for run intent classification project using flask and heroku. In your cloned folder run the following commands:
```
virtualenv env
env\Scripts\activate
pip install flask
pip install gunicorn
pip install scikit-learn
pip install request
pip install pandas
pip install nltk
pip install matplotlib
pip install xlrd
```

3. Train the ML model by running below command -
```
python train_nlp_model.py
```
This would create a serialized version of our ML model, Knowledge base vocabulary, and intent names into the files nlp_model.pkl, knowledgebase_vocabulary.pkl, and intent_names.pkl, respectively.

4. Run app_nlp.py using below command to start Flask API
```
python app_nlp.py
```
By default, flask will run on port 5000.

5. Navigate to URL http://127.0.0.1:5000/ 

6. Define which libraries uses our application. 
```
pip freeze > requirements.txt
```
7. If you haven't already, [Create heroku account](https://signup.heroku.com/) or [log in](https://id.heroku.com/login) your heroku account.
8. If you haven't already, [Create heroku app](https://dashboard.heroku.com/new-app)
9. If you haven't already, download and install the [Heroku CLI](https://devcenter.heroku.com/articles/heroku-cli)
10. If you haven't already, log in to your Heroku account and follow the prompts to create a new SSH public key. Ensure that you are in  project local directory. 
```
heroku login
```
11. Initialize a git repository in a new or existing directory
```
git init
heroku git:remote -a {your-app-name}
```
12. Deploy your application
```
git add .
git commit -am "updating scripts"
git push heroku master
```
