[![](https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=darkgreen)](https://www.python.org)  [![](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=TensorFlow&logoColor=white)](https://www.tensorflow.org) [![](https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/stable/) [![](https://img.shields.io/badge/SciPy-654FF0?style=for-the-badge&logo=SciPy&logoColor=white)](https://www.scipy.org) [![](https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org) [![](https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)  [![](https://img.shields.io/badge/Plotly-239120?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com) [![](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=Keras&logoColor=white)](https://keras.io) [![](https://img.shields.io/badge/conda-342B029.svg?&style=for-the-badge&logo=anaconda&logoColor=white)](https://www.anaconda.com)



# Twitter-Sentiment-Analysis
It is a Natural Language Processing Problem where Sentiment Analysis is done by Classifying the Positive tweets from negative tweets by machine learning models for classification, text mining, text analysis, data analysis and data visualization


## Introduction
This dataset contains 1,600,000 tweets extracted using the twitter api . The tweets have been annotated (0 = negative, 4 = positive) and they can be used to detect sentiment .

Natural Language Processing (NLP) is a hotbed of research in data science these days and one of the most common applications of NLP is sentiment analysis. From opinion polls to creating entire marketing strategies, this domain has completely reshaped the way businesses work, which is why this is an area every data scientist must be familiar with.

Thousands of text documents can be processed for sentiment (and other features including named entities, topics, themes, etc.) in seconds, compared to the hours it would take a team of people to manually complete the same task.

We will do so by following a sequence of steps needed to solve a general sentiment analysis problem. We will start with preprocessing and cleaning of the raw text of the tweets. Then we will explore the cleaned text and try to get some intuition about the context of the tweets. After that, we will extract numerical features from the data and finally use these feature sets to train models and identify the sentiments of the tweets.

This is one of the most interesting challenges in NLP so I’m very excited to take this journey with you!

## Machine Learning and Data Science
Our approach involves utilizing machine learning techniques and text extraction to predict the sentiment of a given text, determining whether it is positive or negative. Initially, we will analyze the text and examine the various words present within it. Once we have a comprehensive understanding of the text, we will proceed with the machine learning analysis, employing deep neural networks. The output from this analysis will be utilized in subsequent machine learning operations to generate predictions regarding the sentiment of the text, specifically determining whether it is positive or negative.

## Natural Language Processing (NLP)
We would be using the __natural language processing__ that is required when doing the machine learning analysis. Performing the natural language processing ensures that the words that are present are converted into mathematical vectors that are used for different machine learning models for prediction. Once the mathematical vectors are converted into different vectors, they are given for the machine learning models for prediction respectively. Therefore, with the features that are present in the text along with some newly created features, the machine learning, and deep learning models would be using those techniques and ensures that they are getting the best outputs respectively. 

## Vectorizers 
It is important to use vectorizers that are important for machine learning. Therefore, a given text which is in the form of a string is converted into a vectorial representation which is what is being used by machine learning models for prediction. Below are some of the vectorizers that were used in the process of converting a text into a mathematical representation. 

* [__Count Vectorizer__](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html)
* [__Tfidf Vectorizer__](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
