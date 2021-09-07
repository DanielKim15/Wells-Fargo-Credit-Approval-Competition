library(tidyverse)
library(e1071)
library(pscl) 
library(car)
library(caret)
train_data <- read_csv("/Users/dannykim/Desktop/UGA Wells Fargo Data Science Competion/Simulated_Data_Train.csv")
str(train_data)


#####Step 1: cleaning data
###Part 1.1: changing data type
train_data$Default_ind <- as_factor(train_data$Default_ind) #this is a factor now
train_data$non_mtg_acc_past_due_12_months_num <- as_factor(train_data$non_mtg_acc_past_due_12_months_num)
train_data$non_mtg_acc_past_due_6_months_num <- as_factor(train_data$non_mtg_acc_past_due_6_months_num)
train_data$mortgages_past_due_6_months_num <- as_factor(train_data$mortgages_past_due_6_months_num)
train_data$`auto_open_ 36_month_num` <- as_factor(train_data$`auto_open_ 36_month_num`)
train_data$card_open_36_month_num <- as_factor(train_data$card_open_36_month_num)
train_data$States <- as_factor(train_data$States)
train_data$ind_acc_XYZ <- as_factor(train_data$ind_acc_XYZ)

names(train_data)[names(train_data) == "auto_open_ 36_month_num"] <- "auto_open_36_month_num"

sum(is.na(train_data$uti_card_50plus_pct)) #there is NA here
sum(is.na(train_data$rep_income)) #There is NA

sapply(train_data,function(x) sum(is.na(x))) #This checks how many missing NA there are for each column
sapply(train_data, function(x) length(unique(x))) #This checks how many unqie variables there are for each columns

filter(train_data, avg_card_debt == 99999) #there's 212 99999
summary(train_data)

## Fixing the NA, use imputation regression
#summary(lm(formula = uti_card_50plus_pct ~ ., data = train_data)) #it shows that uti_card and uti_card_50plus_pct affect each other
#summary(lm(formula =  rep_income ~ ., data = train_data)) #it seems that tot_credit_debt, card_open_36_month_num2, uti_50plus_pct affect income the most

train_data$avg_card_debt <- replace(train_data$avg_card_debt,train_data$avg_card_debt == 99999, NA)
train_data$avg_card_debt

sum(is.na(train_data$avg_card_debt))
sum(train_data$avg_card_debt == 99999)

###Part 1.2: Mean Imputation
train_data$avg_card_debt[is.na(train_data$avg_card_debt)] <- mean(train_data$avg_card_debt, na.rm = TRUE) #imputation for avg_card_debt
train_data$rep_income[is.na(train_data$rep_income)] <- mean(train_data$rep_income, na.rm = TRUE) #imputation for rep_income
train_data$uti_card_50plus_pct[is.na(train_data$uti_card_50plus_pct)] <- mean(train_data$uti_card_50plus_pct, na.rm = TRUE) #imputation for uti_card_50plus_pct

#safe to do because almost all of the predictors have a normaal distrubtuion, very normal

#####Step 2: Logistic Regression time
str(train_data)
train_data$Default_ind <- factor(train_data$Default_ind, levels = c("no","yes"))
train_data$Default_ind

#making model
approval.model <- glm(Default_ind ~., data = train_data, family = "binomial")
summary(approval.model)

#plot diagnostic
plot(approval.model) #residual vs fitted test, terrible linearity,normality is out of control, constant variance is unknown, but the residuals do seem out of place suprisingly

#Improving the model
#used step wise regression test to see which variables decrease the AIC the most for predictive powers
approval.model.update <- step(approval.model)
summary(approval.model.update)

approval.model.update <- glm(formula = Default_ind ~ tot_credit_debt + avg_card_debt + 
                               credit_age + card_age + non_mtg_acc_past_due_12_months_num + 
                               non_mtg_acc_past_due_6_months_num + inq_12_month_num + card_open_36_month_num + 
                               uti_card + ind_acc_XYZ + rep_income, family = "binomial", 
                             data = train_data)

summary(approval.model.update)
plot(approval.model.update) #there's been changes with residual vs fitted plot, but the rest seems similar
#warning: using these diagnostic plot can be misinterpreted wrong, best not to use them
#Tests to use

4299/(4299+208) #precision is 0.9538496
#recall is 0.9348  

# 2*(Recall * Precision) / (Recall + Precision)
2*(0.9348 * 0.9538496)/ (0.9348 + 0.9538496) #0.9442287


###Part 3.2: Finding the ROC
library(ROCR)
p <- predict(approval.model.update,newdata=subset(train_data,select=c(1,2,3,5,6,7,10,12,14,18,19)),type='response')
pr <- prediction(p, train_data$Default_ind)
prf <- performance(pr, measure = "tpr", x.measure = "fpr")
plot(prf) #result, use the result from the roc curve to use for the confusion matrix

auc <- performance(pr, measure = "auc")
auc <- auc@y.values[[1]]
auc #checks the aerea under the curve accuracy, its 0.8090096, so we can use it for the prediction

?roc

###Finding threshold
test_data_graph <- subset(test_data,select=c(1,2,3,5,6,7,10,12,14,18,19,21))
train_data_graph <- subset(train_data,select=c(1,2,3,5,6,7,10,12,14,18,19,21))
cutoffs <- seq(0.1,0.9,0.1)
accuracy <- NULL
for (i in seq(along = cutoffs)){
  prediction <- ifelse(approval.model.update$fitted.values >= cutoffs[i], 1, 0) #Predicting for cut-off
  accuracy <- c(accuracy,length(which(test_data_graph$Default_ind ==prediction))/length(prediction)*100)
}

plot(cutoffs, accuracy, pch =19,type='b',col= "steelblue",
     main ="Logistic Regression", xlab="Cutoff Level", ylab = "Accuracy %")


##confusion matrix to see 
sum(pdata < 0.15) #Use this 2
pdata <- predict(approval.model.update,newdata=subset(test_data,select=c(1,2,3,5,6,7,10,12,14,18,19)),type='response')
pdata <- ifelse(pdata < 0.15,1,0) 
table(pdata > 0.15, test_data$Default_ind) # matrix table, the 0 and true is the true positives while 1 and false is true negatives


confusionMatrix(table(data = as.numeric(pdata > 0.15), reference = test_data$Default_ind))
str(train_data)



#1,2,3,5,6,7,12,14,18,19


####Test Data
test_data <- read_csv("/Users/dannykim/Desktop/UGA Wells Fargo Data Science Competion/Simulated_Data_Test.csv")

test_data$Default_ind <- as_factor(test_data$Default_ind) #this is a factor now
test_data$non_mtg_acc_past_due_12_months_num <- as_factor(test_data$non_mtg_acc_past_due_12_months_num)
test_data$non_mtg_acc_past_due_6_months_num <- as_factor(test_data$non_mtg_acc_past_due_6_months_num)
test_data$mortgages_past_due_6_months_num <- as_factor(test_data$mortgages_past_due_6_months_num)
test_data$auto_open_36_month_num <- as_factor(test_data$auto_open_36_month_num)
test_data$card_open_36_month_num <- as_factor(test_data$card_open_36_month_num)
test_data$States <- as_factor(test_data$States)
test_data$ind_acc_XYZ <- as_factor(test_data$ind_acc_XYZ)

names(test_data)[names(test_data) == "auto_open_ 36_month_num"] <- "auto_open_36_month_num"

test_data$a
str(test_data)

sum(is.na(test_data$uti_card_50plus_pct)) #there is NA here
sum(is.na(test_data$rep_income)) #There is NA

## Fixing the NA, use imputation regression
#summary(lm(formula = uti_card_50plus_pct ~ ., data = train_data)) #it shows that uti_card and uti_card_50plus_pct affect each other
#summary(lm(formula =  rep_income ~ ., data = train_data)) #it seems that tot_credit_debt, card_open_36_month_num2, uti_50plus_pct affect income the most

test_data$avg_card_debt <- replace(test_data$avg_card_debt,test_data$avg_card_debt == 99999, NA)
test_data$avg_card_debt

sum(is.na(test_data$avg_card_debt))
sum(test_data$avg_card_debt == 99999)

test_data$avg_card_debt[is.na(test_data$avg_card_debt)] <- mean(test_data$avg_card_debt, na.rm = TRUE) #imputation for avg_card_debt
test_data$rep_income[is.na(test_data$rep_income)] <- mean(test_data$rep_income, na.rm = TRUE) #imputation for rep_income
test_data$uti_card_50plus_pct[is.na(test_data$uti_card_50plus_pct)] <- mean(test_data$uti_card_50plus_pct, na.rm = TRUE) #imputation for uti_card_50plus_pct







