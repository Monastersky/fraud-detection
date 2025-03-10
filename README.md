# Problem Statement

I am interested to see if I can create a fraud detection model that limits the instances where I am not catching instances of fraud. This is important because financial institutions are often subject to financial penalties when they allow fraudulent charges to go through. If I were able to create a model that detected fraudulent charges before they went through and declined these charges this would save credit card companies massive amounts of money.

# Notebooks

The discovery notebook cleans and analyzes my data and looks to find insights on any trends I had in my data. The modeling notebook balances my classes and applies a variety of models to predict instances of fraud. The Bayesian notebook balances my classes and applies a Bayesian logistic regression model to predict instances of fraud. The live widget demo allows someone to make live predictions on whether a charge is fraudalent or not.

- [Discovery](./discovery.ipynb)

- [Modeling Notebook](./modelling_notebook.ipynb)

- [Bayesian Notebook](./pymc_model.ipynb)

- [Live Widget Demo](./credit_risk_widget.ipynb)

# Datasets

For this project I used data a dataset from Kaggle titled “Credit Card Fraud Detection” and can be found and downloaded at https://www.kaggle.com/mlg-ulb/creditcardfraud. In this dataset there is a total of 284,807 observations. There is a column titled “Class” which has a value of 1 if the charge was fraudulent and 0 if the charge was not fraudulent. There is a column labeled Amount, which is the amount of the charge and the column labeled Time, which is the difference between the transaction and the first transaction in the dataset. The remaining 28 columns are PCA’d columns from a previous dataset so that I would not be given confidential information. This makes recommendations hard considering that I do not know what these columns mean.  


# Executive Summary

I started my exploratory analysis looking at my different classes to see if they were even or not. My fraud class has only 492 observations while my non-fraud class has 284,315 observations so there is a large class imbalance. I then decided to graph my variables to see if there was any difference between the charges labeled as fraud and the charges labeled as non-fraud. When looking at the distribution for fraud amounts I saw that there were spikes in observations at 1, 0 and 99.99 and no spikes elsewhere. The non-fraud graph is still skewed to the and has an initial spike but has far more variance in the observations.

![Amount Distribution for fraud and non-fraud charges](plots/amount.png)

The PCA’d columns are hard to gather insight about given that I don’t know what they represent but I did analyze them none the less. I found that the columns that had low correlation to the class variable had very similar overlapping distribution curves for the fraud and the non-fraud classes. However if you look at columns such as V17, which has the strongest correlation to class you can see that the distribution plots are completely different for the fraud and the non-fraud class. The non-fraud class is centralized around zero. The fraud class is centralized closer to -10 and its values have a larger distribution, ranging from -30 to 10. What was interesting was when I plotted this same plot but for another highly correlated variable, V14 I got very similar results.

![Distribution for V17](plots/V17.png)

I was also interested to see if there were any clear clusters of fraud charges in scatterplots between my variables, which would help in identifying fraud instances. I plotted the V5 variable against the V6 varaible and was able to see a slight difference between the fraud and the non-fraud charges. The fraud charges had more variation in their V5 amount, however the non-fraud charges still overlapped with most of the fraud charges.

![Scatterplot for V5 and V6](plots/V5_V6.png)

# Modeling

To train my models I randomly selected 50,000 rows and used them to train my model. This is because since I had so many data points running models on all my data or even 50 percent of my data would take a very long time. Since my classes were highly unbalanced I decided to use SMOTE, which is way to balance my classes so that my models did not predict not fraud every single time. SMOTE works by underrepresenting my majority class which is not fraud and overrepresenting my minority class which is fraud. I changed my data so that I had the same amount of fraud cases in my sample dataset and did not change my original dataset.

I wanted to run many models to make sure the model I was choosing for production worked the best. I have displayed a plot below that has all the models and the recall and accuracy scores for each model. For this project I wanted to reduce false negatives so I wanted the highest recall score. Since I had multiple models with a 1.0 recall score, I chose the model that had the highest accuracy score of the models that had perfect recall scores. Please see the scores of my models below.

|Model Name|Recall Score|Accuracy Score|F1 Score|
|---|---|---|---|
|Logistic Regression|0.8983739837398373|0.9910044345820151|0.2565293093441671|
|Random Forest Classification|1.0|0.99942417145646|0.8571428571428571|
|Gradient Boosting Classifier|1.0|0.9990414561439852|0.7828162291169452|
|Extreme Gradient Boost Classifier|1.0|0.9993609707626568|0.8439108061749571|
|K-Nearest Neighbors Classifier|1.0|0.99539337165168|0.42857142857142855|
|Support Vector Machines|0.8963414634146342|0.9926968087160779|0.29777177582714387|

I also ran a Bayesian logistic regression to see if having a distribution of coefficients would give me better predictability. Unfortunately, the Bayesian logistic regression had worse scores with an average recall score of .795.

![](plots/pymc_dist.png)

# Conclusions

My best model was the random forest model which was able to get a perfect recall score and an accuracy score .9994. I found out that boosting methods and support vector machines work well but do not outperform a random forest in this case. It is also nice that random forest is not a black box model so that we can see which features were most important. I also found that the Bayesian logistic regression models did not perform better the frequentist logistic regression. I was successfully able to create a credit card detection model that caught all instances of fraud.

I was also able to create a live widget that allows me to create prodictions on what the time and amount of the charge were as well as the amounts for all the PCA'd columns. 

# Going Forward  
Going forward I would hope to get information on the what the PCA’d columns mean so I could find out what features most impact fraud.

# Sources

https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4836738/

https://www.kaggle.com/mlg-ulb/creditcardfraud

https://stackoverflow.com/questions/14885895/color-by-column-values-in-matplotlib

https://stackoverflow.com/questions/34649969/how-to-find-the-features-names-of-the-coefficients-using-scikit-linear-regressio

https://stackoverflow.com/questions/25009284/how-to-plot-roc-curve-in-python

https://discourse.pymc.io/t/bad-initial-energy-inf-the-model-might-be-misspecified/1431/6

https://docs.pymc.io/notebooks/GLM-logistic.html

https://stackoverflow.com/questions/25009284/how-to-plot-roc-curve-in-python

https://github.com/hussainburhani/ds_bike_startup/blob/master/code/mlr_general_cycles_demo.ipynb
