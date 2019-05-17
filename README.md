# Problem Statement

I am interested to see if I can create a fraud detection model that limits the instances where I am not catching instances of fraud. This is important because financial institutions are often subject to financial penalties when they allow fraudulent charges to go through. If I were able to create a model that detected fraudulent charges before they went through and declined these charges this would save credit card companies massive amounts of money.

# Notebooks

The discovery notebook cleans and analyzes my data and looks to find insights on any trends I had in my data. The modelling notebook balances my classes and applies frequentist models to predict instances of fraud. The Bayesian notebook balances my classes and applies a Bayesian logistic regression model to predict instances of fraud.

- [Discovery](./discovery.ipynb)

- [Modelling Notebook](./modelling_notebook.pynb)

- [Bayesian Notebook](./pymc_notebook)

# Datasets

For this project I used data a dataset from Kaggle titled “Credit Card Fraud Detection” and can be found and downloaded at https://www.kaggle.com/mlg-ulb/creditcardfraud. There is a column titled “Class” which has a value of one if the charge was fraudulent and 0 if the charge was not fraudulent. There is a column labeled Amount which is the amount of the charge and the column labeled Time is the difference between the transaction and the first transaction in the dataset. The remaining 28 columns are PCA’d columns which retain the information from the data but allow the information to remain confidential. This makes recommendations hard considering that I am not able to columns mean.  


# Executive Summary

I started my exploratory analysis looking at my different classes to see if they were even or not. My fraud class has only 492 observations while my non-fraud class has 284,315 observations so there is a large class imbalance. I then decided to graph my variables to see if there was any difference between the charges labeled as fraud and the charges labeled as non-fraud. When looking at the distribution for fraud amounts I saw that there were spikes in observations at 1, 0 and 99.99 and no spikes elsewhere. The non-fraud graph is still skewed to the and has an initial spike but has far more variance in the observations.

![Amount Distribution for fraud and non-fraud charges](../data/amount.png)

The PCA’d columns are hard to gather insight about given that I don’t know what they represent but I did analyze them none the less. I found that the columns that had low correlation to the class variable had very similar overlapping distribution curves for the fraud and the non-fraud classes. However if you look at columns such as V17 that has the strongest correlation to class you can see that the distribution plots are completely different for the fraud and the non-fraud class. The non-fraud class is centralized around zero and its values are centralized around zero. The fraud class is centralized closer to -10 and its values are distributed far more ranging from -30 to 10. What was interesting was when I plotted this same plot but for another highly correlated variable, V14 I got very similar results.

I was also interested to see if there were any clear clusters of fraud charges in scatterplots between my variables, which would help in identifying fraud instances. I plotted the V7 variable against amount and was able to see a slight difference between the fraud and the non-fraud charges. The fraud charges had values along the y values where x equaled zero while most of the values for non-fraud were where y equaled zero, however the non-fraud charges still overlapped with most of the fraud charges. I tried plotting V5 against V6 and while the scatterplot looked very different I had similar issues. 

# Modelling

To train my models I randomly selected 50,000 rows and used them to train my model. This is because since I had so many data points running models on all my data or even 50 percent of my data would take a very long time. Since my classes were highly unbalanced I decided to use SMOTE which is way to balance my classes so that my models did not predict not fraud every single time. SMOTE works by underrepresenting my majority class which is not fraud and overrepresenting my minority class which is fraud. I changed my data so that I had the same amount of fraud cases in my sample dataset and did not change my original dataset.

I wanted to run many models to make sure the model I was choosing for production worked the best. I have displayed a plot below that has all the models and the recall and accuracy scores. For this project I wanted to reduce false negatives so I wanted the highest recall score. Since I had multiple models with 1.0 recall scores I chose to pick the model that had the highest accuracy score of models that had perfect recall scores.

I also ran a Bayesian logistic regression to see if having a distribution of coefficients would give me better predictability. Unfortunately, the Bayesian logistic regression had worse scores with an average recall score of .795.

# Conclusions

My best model was the random forest model which was able to get a perfect recall score and an accuracy score .9994. I found out that boosting methods and support vector machines work well but do not outperform a random forest in this case. Since interpretability is not important in this case I do not need to worry about random forest’s interpretability. I found that I unsupervised methods such as k-means did not work well for my data. I also found that Bayesian logistic regression models did not perform better than their frequentist counterparts. I was able to create a credit card detection model that caught all instances of fraud, in fact my model caught all instances of fraud.

# Going Forward  
Going forward I would hope to get information on the what the PCA’d columns mean so I could find out what features most impact fraud. I also would like to increase the predictability of my Bayesian model as this model gives me more insight to the distribution of my coefficients. Lastly I would like to use flask to implement this model in real life.
