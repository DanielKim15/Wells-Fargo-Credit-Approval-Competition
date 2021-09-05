# Wells-Fargo-Credit-Approval-Competition
For years banks have reliably used statistical modeling to predict whether to approve or disapprove credit to individuals. Banks have to carefully choose which person to give credit since they have a finite amount in their system, so accuracy is very important in prediction. Always looking for another method to improve their prediction ability, they started looking toward machine learning models as an alternative tool to use. To test their effectiveness, I decided to build a logistic regression, a statistical model, and a random forest, machine learning model, and compared side by side the results of their predictive power.

Goal: Figure out which predictors (past delinquency, average credit debt, etc) affected the account holder the most when it comes to getting a default or not.

#Data Pre-Processing: 
Data Exploratory/Data Processing/ Cleaning Section
When observing the three datasets, the first thing to note is how all of the predictors except the average monthly debt have a normal distribution, observations or datapoints gathering together near the average amount. The average monthly debt has a right skewed distribution, but upon closer inspection it seems that the 99999 in its observations is meant to be written as missing value instead. By replacing the 99999 with NA instead, the distribution changed into a normal distribution again. Three predictors, customer’s annual income, percentage of open credit cards with over 50% being used, and average monthly debt, has missing values in it. Having too many missing values could reduce the statistical significance and reduce the quality of our results due to bias. To reduce these negative effects mean imputation, the replacement of missing values with the mean of the affected predictor, were implemented in those three predictors. On some cases using this technique could negatively change our results, but considering how all of the predictors have a normal distribution, the technique is justified. 


After processing the data, we moved forward by quickly visualizing each part of the variables. The first thing to note is our response variable the default is wildly unbalanced, 92% has no default, while 8% has a default. This can cause our models to have huge difficult predicting the ones with default because of the lack of information there is compared to the no default one, resulting in our model to have an increase chance of having an error finding a default. This problem is later addressed in the paper. The second thing to note is that the predictor States has no relevance to predicting the default account. By looking at the count and the variance among the states, there’s little difference between each other so its relevance seems very little for the model. 



Logistic Regression
When running the logistic regression, the stepwise regression method, where the function determines which predictors are the best for the model by picking the ones that best reduce the prediction error, is used. The method revealed that total credit debit, average card debit, credit age, Number of non-mortgage credit-product accounts by the applicants with delinquency in the past 12 months, Number of non-mortgage credit-product accounts by the applicants with delinquency in the past 6 months, Number of credit inquiries in last 12 months, Number of credit cards opened by applicant in last 36 months, ratio of balance divided by credit limit on all credit card accounts (called utilization). 





