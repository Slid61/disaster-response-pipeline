# Disaster Response Pipeline Project
A machine learning project part of Udacity's Data Scientist nanodegree program.
The aim of this project is to analyze short messages and classify them according to various
disaster-relief related categories, with a potential use for this being to forward said messages
to the appropriate response agencies.

This product was written mostly in python, with html, css, and javascript integration using flask.

## Libraries and Modules Used:

### python:

- [sys](https://docs.python.org/3/library/sys.html)
- [re](https://docs.python.org/3/library/re.html?highlight=re#module-re)
- [numpy](https://numpy.org/)
- [pandas](https://pandas.pydata.org/)
- [sqlalchemy](https://www.sqlalchemy.org/)
- [progressbar2](https://pypi.org/project/progressbar2/)
- [nltk](https://www.nltk.org/)
- [scikit-learn](https://scikit-learn.org/stable/)
- [flask](https://flask.palletsprojects.com/en/2.0.x/)
- [plotly](https://plotly.com/)
- [joblib](https://pypi.org/project/joblib/)



## File Structure:
disaster-response-pipeline/  
├── app/  
│   ├── templates/  
│   │   ├── go.html  
│   │   └── master.html  
│   └── run.py  
├── data/  
│   ├── disaster_categories.csv  
│   ├── disaster_messages.csv  
│   ├── DisasterResponse.db  
│   └── process_data.py  
├── models/  
│   └── train_classifier.py  
└── README.md  

### How to run:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run the ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run the ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
