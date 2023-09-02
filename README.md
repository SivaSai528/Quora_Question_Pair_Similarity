# Quora_Question_Pair_Similarity

QUORA QUESTION PAIR SIMILARITY with MLOPS

**ABSTRACT**

Over millions of people visit Quora every month, so it's no surprise that many
people ask similarly worded questions. Multiple questions with the same intent
can cause seekers to spend more time finding the best answer to their question,
and make writers feel they need to answer multiple versions of the same
question. Quora values canonical questions because they provide a better
experience to active seekers and writers, and offer more value to both of these
groups in the long term. The main aim of the project is to predict whether a pair
of questions are similar or not by employing several methodologies of natural
language processing.

**PROBLEM STATEMENT**

Identify which questions asked on Quora are duplicates of questions that have already been asked.
Domain: Social Network 
Prerequisites: Python programming language, Machine Learning, NLP and MLFlows

**DATASET DETAILS**

Data set:
https://github.com/Koorimikiran369/Quora-Question-Pairing/blob/main/train.csv.zip
Data Overview
Data will be in a file Train.csv
Train.csv contains 5 columns : qid1, qid2, question1, question2, is_duplicate
Number of rows in Train.csv = 404,290

**HOW TO SETUP/ RUN**

1. Create a Python Environment
2. Install all Requirments and Activate Environment
3. Run all the Python Files on by one to Get all the DataSets
4. If you want to run the application you can easily run the application by running App.py with help of model
5. if you want the data sets first download the given data set from the link and run all the python files one by one based on number ranking in python files

**Approach of the Project**

1.Preprocessing
2.Token Based Features Extraction
3.Length Based Features Extraction
4.Fuzzy Features Extraction
5.TFIDF 
6.Word2Vec 
7.XGBoost With Hyperparameter Tuning
8.Logloss and Confusion Matrix
9.Analysis with MLflows
