#Data Science Competition Logistic Regression

###Step 1: Understanding/Fixing the Data for train data
library(tidyverse)
install.packages("e1071")
library(e1071) #for confusion matrix
install.packages("pscl")
library(pscl) #Apparantly this library allows you to show different versions of R^2
library(ggpubr)
install.packages("mice") # Needs to be done only once
library(mice) #This allows the regression imputation to hap
detach(mice)
library(dplyr)
library(naniar) #this is to replace 99999 with NA
library(caret)
train_data <- read_csv("/Users/dannykim/Downloads/Simulated_Data_Train.csv")
str(train_data)
summary(train_data)
#max(train_data$Default_ind)
#min(train_data$Default_ind)
#str(train_data$Default_ind) #
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

mice(train_data, method = "norm.nob", m = 1, maxit=1) #The original method didn't work, so I decided to
methods(mice)

train_data_final <- complete(mice(train_data, method = "norm.nob", m = 1, maxit=1))
sum(is.na(train_data_final$rep_income))
sum(is.na(train_data_final$uti_card_50plus_pct))
sum(is.na(train_data_final$avg_card_debt))

str(train_data_final)

hist(train_data_final$avg_card_debt)


### Step 1.4: Creating a new csv cleaned file
train_data_clean <- train_data_final
write.csv(train_data_clean,'train_data_clean_final_updated.csv') #GIVE THIS TO THE GROUP
str(train_data_clean)


### Step 1.5: Find the distribution
ggdensity(train_data$tot_credit_debt) #Normally distributed for tot_credit_debt

ggdensity(train_data$avg_card_debt) #There's an outlier in avg_card_debt, must be rid of 
ggqqplot(train_data$avg_card_debt) #With the outlier, it made the distribution almost biomodial
hist(train_data$avg_card_debt)
boxplot(train_data$avg_card_debt)

#ggdensity(train_data$credit_age) #credit_age is Normally distributed 
#ggdensity(train_data$credit_good_age) #credit_good_age is Normally distributed
#ggdensity(train_data$card_age) #card_age is Normally distributed
#ggdensity(train_data$credit_good_age) #credit_good_age is Normally distributed
#ggdensity(train_data$uti_card) #uti_card is Normally distributed
#ggdensity(train_data$uti_50plus_pct) #uti_50plus_pct is Normally distributed
#ggdensity(train_data$uti_max_credit_line) #uti_max_credit_line is Normally distributed
#ggdensity(train_data$uti_card_50plus_pct) #uti_card_50plus_pct is Normally distributed (Note: NA is here)
#ggdensity(train_data$rep_income) #rep_income is Normally distributed (Note: NA is here)
ggdensity(train_data$credit_past_due_amount) #credot_past_due_amount is left skewed heavily, might have to use medians
hist(train_data$credit_past_due_amount)

ggqqplot(train_data$uti_card)
hist(train_data_final$uti_card)
###Step 1.75: Graph

#boxplot between Default_ind and other variables
ggplot(data=train_data_final)+ #tot credit debt
  geom_boxplot(mapping = aes(x=Default_ind, y=tot_credit_debt)) #it seems that 1, or default has a bigger IQR

ggplot(data=train_data_final)+ #credit good age
  geom_boxplot(mapping = aes(x=Default_ind, y=credit_good_age)) #it seems that 1, or default has a bigger IQR

ggplot(data=train_data_final)+ #avg card debt
  geom_boxplot(mapping = aes(x=Default_ind, y=avg_card_debt)) #it seems that 1, or default has a bigger IQR

ggplot(data=train_data_final)+ #avg card debt
  geom_boxplot(mapping = aes(x=Default_ind, y=avg_card_debt)) #it seems that 1, or default has a bigger IQR



#Scatter plot with the states as the factor
ggplot(data=train_data_final)+
  geom_point(mapping = aes(x=card_age, y=avg_card_debt , color=factor(States))) #Ehhhh not helping, but we can compare it to others



#Scatter plot with the income bracket as the factor
train_data_final_testing <- train_data_final

train_data_final_testing$income_bracket <- cut(train_data_final_testing$rep_income, breaks = c(0,50000,75000,100000,150000), labels = c(1,2,3,4)) #this breaks up the incomes into brackets
train_data_final_testing$income_bracket

ggplot(data=train_data_final_testing)+
  geom_point(mapping = aes(x=rep_income, y=avg_card_debt , color=factor(income_bracket))) #welp, I finally made the income bracket and wow its not helpful


###Step 2: Experiment

##Linear Regression (trying to see the relationship between variables)
str(train_data_final)
summary(lm(formula = rep_income ~ uti_card + avg_card_debt + credit_good_age + mortgages_past_due_6_months_num + card_open_36_month_num, data = train_data_final ))
summary(lm(formula = rep_income ~ ., data = train_data_final ))

summary(train_data_final$rep_income)
train_data_final$ind_acc_XYZ

## Anova
#I want to test rep_income with ind_acc_XYZ and mortages (result: nothing happend)
summary(aov(rep_income ~ ind_acc_XYZ + mortgages_past_due_6_months_num + ind_acc_XYZ:mortgages_past_due_6_months_num , data = train_data_final))
summary(aov(rep_income ~ ind_acc_XYZ + card_open_36_month_num + ind_acc_XYZ:mortgages_past_due_6_months_num , data = train_data_final))

hmm <- aov(rep_income ~ card_open_36_month_num, data = train_data_final)
summary(hmm)
hmm2 <- TukeyHSD(hmm) #seems like to run a tukey, you need one categorical variable
TukeyHSD(hmm)
plot(hmm2)

train_data_final$card_open_36_month_num

#for the non_mtg_acc_past_due_12_months_num, the more past due notice you have the higher chance you'll get a default
summary(glm(formula = Default_ind ~ non_mtg_acc_past_due_12_months_num, family = binomial(link = "logit"), data = train_data_final))
#for credit age, the more age there is the lower the chance you'll get a default
summary(glm(formula = Default_ind ~ credit_age, family = binomial(link = "logit"), data = train_data_final))

#for inq the more we have the higher chance we'll get a default, until you reach 11
train_data_final$inq_12_month_num <- as.factor(train_data_final$inq_12_month_num)
summary(glm(formula = Default_ind ~ inq_12_month_num, family = binomial(link = "logit"), data = train_data_final))
#For uticard, the more you have the higher chacne youll get a default
summary(glm(formula = Default_ind ~ uti_card, family = binomial(link = "logit"), data = train_data_final))
#for avg card debt, if you have more of it you have the less chance you get a default?
summary(glm(formula = Default_ind ~ avg_card_debt, family = binomial(link = "logit"), data = train_data_final))
t.test(train_data_final$avg_card_debt~train_data_final$Default_ind) #this shows that mean for 0 is 13236 while the one is 1 is 12828, interesting finding, maybe its because they think the more debt you have the more likely you have money
#For rep income, the higher the income the less likelly you'll get a default
summary(glm(formula = Default_ind ~ rep_income, family = binomial(link = "logit"), data = train_data_final))
t.test(train_data_final$rep_income~train_data_final$Default_ind) #this shows that mean for 0 is 13236 while the one is 1 is 12828, interesting finding, maybe its because they think the more debt you have the more likely you have money

summary(glm(formula = Default_ind ~ ind_acc_XYZ, family = binomial(link = "logit"), data = train_data_final))

#until you reach 3, the more you have the higher chance youll get a default
summary(glm(formula = Default_ind ~ non_mtg_acc_past_due_6_months_num, family = binomial(link = "logit"), data = train_data_final))


#Results: 
# 1. We know for sure that rep income and card open 36 month has a correlation with each other. It seems that 2 cards open has the biggest effect, and the it correlates with bigger income

##Logistic Regressions
#Note: I might have to do all of this again with sample sizes
#If I include all variables
test <- glm(formula = Default_ind ~ ., family = binomial(link = "logit"), data = train_data_final)
summary(test)

anova(test, test="Chisq")

pR2(test) #Use the mcFadden R^2 to assess the model fit, if include everything the fit is only 0.26 (turns out anything betwwen 0.2 to 0.4 is a really good fit)

#Backward elimination procedure
test11 <- glm(formula = Default_ind ~ avg_card_debt + credit_age + card_age + non_mtg_acc_past_due_12_months_num + non_mtg_acc_past_due_6_months_num + mortgages_past_due_6_months_num + inq_12_month_num + card_open_36_month_num  + uti_card +  ind_acc_XYZ  + rep_income , family = binomial(link = "logit"), data = train_data_final) #eliminate 
summary(test11) #use this

test11 <- step(test11)

anova(test11, test="Chisq")
pR2(test11) #the mcfadden R^2 is 0.23, so it seems that we have a good model





######This section is for predicting, that test dataset next to select has to be used with our other dataset, Most usefule one so far

###This is the Backward elimination procedure prediction (Use this 1)
summary(test11)

fitted.results2 <- predict(test11,newdata=subset(test_data_final,select=c(2,3,5,6,7, 8,10,12,14,18,19,21)),type='response')
fitted.results2 <- ifelse(fitted.results2 < 0.087,1,0) #Maybe this is the good one, my accuracy is 0.194

misClasificError2 <- mean(fitted.results2 != test_data_final$Default_ind)
print(paste('Accuracy',1-misClasificError1)) #accuracy is still 80%

#plot
library(ROCR)
p <- predict(test11,newdata=subset(test_data_final,select=c(2,3,5,6,7, 8,10,12,14,18,19,21)),type='response')
pr <- prediction(p, test_data_final$Default_ind)
prf <- performance(pr, measure = "tpr", x.measure = "fpr")
plot(prf) #result 

auc <- performance(pr, measure = "auc")
auc <- auc@y.values[[1]]
auc #checks the curvy accuracy

####################################
##odd ratio and confidence interval
exp(cbind(OR = coef(test11), confint(test11))) #odd ratio 

##confusion matrix
sum(pdata < 0.087) #Use this 2
pdata <- predict(test11,newdata=subset(test_data_final,select=c(2,3,5,6,7, 8,10,12,14,18,19,21)),type='response')
pdata <- ifelse(pdata < 0.087,1,0) 
table(pdata < 0.087, test_data_final$Default_ind) # matrix table, the 0 and true is the true positives while 1 and false is true negatives

table(test_data_final$Default_ind) #how to find threshold
401/4599 #From the test data, its 0.087
1586/18414 #Maybe make the threshold 0.086



################################################3
###Step 1: Clean the test data
test_data <- read_csv("/Users/dannykim/Downloads/Simulated_Data_Test.csv")
str(test_data)

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

mice(test_data, method = "norm.nob", m = 1, maxit=1) #The original method didn't work, so I decided to
methods(mice)

test_data_final <- complete(mice(test_data, method = "norm.nob", m = 1, maxit=1))
sum(is.na(test_data_final$rep_income))
sum(is.na(test_data_final$uti_card_50plus_pct))
sum(is.na(test_data_final$avg_card_debt))

str(test_data_final)



### Step 1.4: Creating a new csv cleaned file
test_data_clean <- test_data_final
write.csv(test_data_clean,'test_data_clean_final_updated.csv') #GIVE THIS TO THE GROUP
str(test_data_clean)


###Step 1.5: Find the distribution
ggdensity(test_data$credit_past_due_amount) #credot_past_due_amount is left skewed heavily, might have to use medians



####################################
###Step 1: Clean the Validation data
valid_data <- read_csv("/Users/dannykim/Downloads/Simulated_Data_Validation.csv")
str(valid_data)

valid_data$Default_ind <- as_factor(valid_data$Default_ind) #this is a factor now
valid_data$non_mtg_acc_past_due_12_months_num <- as_factor(valid_data$non_mtg_acc_past_due_12_months_num)
valid_data$non_mtg_acc_past_due_6_months_num <- as_factor(valid_data$non_mtg_acc_past_due_6_months_num)
valid_data$mortgages_past_due_6_months_num <- as_factor(valid_data$mortgages_past_due_6_months_num)
valid_data$`auto_open_ 36_month_num` <- as_factor(valid_data$`auto_open_ 36_month_num`)
valid_data$card_open_36_month_num <- as_factor(valid_data$card_open_36_month_num)
valid_data$States <- as_factor(valid_data$States)
valid_data$ind_acc_XYZ <- as_factor(valid_data$ind_acc_XYZ)

names(valid_data)[names(valid_data) == "auto_open_ 36_month_num"] <- "auto_open_36_month_num"

sum(is.na(valid_data$uti_card_50plus_pct)) #there is NA here
sum(is.na(valid_data$rep_income)) #There is NA


## Fixing the NA, use imputation regression
#summary(lm(formula = uti_card_50plus_pct ~ ., data = train_data)) #it shows that uti_card and uti_card_50plus_pct affect each other
#summary(lm(formula =  rep_income ~ ., data = train_data)) #it seems that tot_credit_debt, card_open_36_month_num2, uti_50plus_pct affect income the most

valid_data$avg_card_debt <- replace(valid_data$avg_card_debt,valid_data$avg_card_debt == 99999, NA)
valid_data$avg_card_debt

sum(is.na(valid_data$avg_card_debt))
sum(valid_data$avg_card_debt == 99999)

mice(test_data, method = "norm.nob", m = 1, maxit=1) #The original method didn't work, so I decided to
methods(mice)

valid_data_final <- complete(mice(valid_data, method = "norm.nob", m = 1, maxit=1))
sum(is.na(valid_data_final$rep_income))
sum(is.na(valid_data_final$uti_card_50plus_pct))
sum(is.na(valid_data_final$avg_card_debt))

str(valid_data_final)


### Step 1.4: Creating a new csv cleaned file
valid_data_clean <- valid_data_final
write.csv(valid_data_clean,'valid_data_clean_final_updated.csv') #GIVE THIS TO THE GROUP
str(valid_data_clean)


