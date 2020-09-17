#  STATE TWITTER TROLL DETECTION USING TRANSFORMERS 

![](https://cdn-images-1.medium.com/max/1600/1*O4DSBYKmCHdeG0B5i5BPbw.png)

With the 2020 US election around the corner, the issue of electoral interference by state actors via social media and other online means is back in the spotlight. Can a fine tuned transformer model do a better job of detecting these state troll tweets than "classic" machine learning approaches? This is what we'll try to assess in this series of notebooks 

# REPO STRUCTURE:
## 1. DATA FOLDER

* 5 CSV files for notebooks in this series. Note that raw troll tweet files from Twitter are not included here.


## 2. NOTEBOOKS FOLDER

* Notebooks 1.0 - 1.2: Data collection, cleaning and preparation. Optional if you just want to experiment with the final dataset.

* Notebooks 2.0 - 2.1: Fine tuning distilbert with custom dataset and detailed testing with unseen validation dataset, as well as a fresh dataset with state troll tweets from Iran.

* Notebook 3.0: Create and test optimised logistic regression model against datasets used to assess fine tuned Distilbert model.


## 3. APP FOLDER

* app.py + folders for "static" and "template: simple app for use on a local machine to demonstrate how a state troll tweet detector can be used in deployment. Unfortunately free hosting accounts can't accomodate the disk size required for pytorch and the fine tuned model, so I've not deployed this online. 


## 4. TROLL_DETECT FOLDER

* Fine tuned Distilbert model from Colab notebook2.0. Too big for Github, download [here](https://www.dropbox.com/sh/90h7ymog2oi5yn7/AACTuxmMTcso6aMxSmSiD8AVa) from Dropbox instead.


## 5. PKL FOLDER

* Pickled logistic regression model from notebook3.0 

---
