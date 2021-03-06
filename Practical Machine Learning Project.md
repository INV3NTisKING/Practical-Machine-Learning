---
title: "Practical Machine Learning Project"
author: "Michael Onizak"
date: "January 23, 2017"
output: html_document
keep_md: true
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```


##**EXCECUTIVE SUMMARY**


This document contains the Practical Machine Learning Project. The goal of this project was to predict the manner in which 6 participants performed some exercise as described below. 

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).

I have used what we have learned in Coursera's Data Science Course thus far. This will be my reasoning as to what models I chose to use. After choosing the models it was then just a matter of seeing which model had a better accuracy. I expect to see at least a 90% accuracy for one of the 3 models I build.

**NOTE: Please see "images" folder in my github for the graphs pertaining to the Project. I apologize for the lack of embedding and any inconvenience this may cause as there were some personal issues that took me away from this weeks project. Thank you**

##**LOADING DATA**

The training data for this project are available here:

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

The test data are available here:

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

The data for this project come from http://groupware.les.inf.puc-rio.br/har.

Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. Qualitative Activity Recognition of Weight Lifting Exercises. Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013.

Read more: http://groupware.les.inf.puc-rio.br/har#ixzz4WZO1FiKL

First, to run this analysis, we will use the following packages: In here we will also create a validation split of 70/30.


```{r, echo=TRUE}
library(knitr)
library(caret)
library(rpart)
library(rpart.plot)
library(rattle)
library(randomForest)
library(corrplot)

##Setting up a directory, dowloading and reading the file and creating a partition

setwd("~/r.PROJECTS/R_dir") 
if(!file.exists("./DSCourse_9_Assignment_1")) dir.create("./DSCourse_9_Assignment_1")
setwd("~/r.PROJECTS/R_dir/DSCourse_9_Assignment_1")
file_Url_train <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
file_Url_test  <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
training <- read.csv(url(file_Url_train))
testing  <- read.csv(url(file_Url_test))
inTrain  <- createDataPartition(training$classe, p=0.7, list=FALSE)
train_set <- training[inTrain, ]
test_set  <- training[-inTrain, ]
```

##**Cleaning the Data**

Here I simply cleaned the data for our models.


```{r, echo=TRUE}

#Cleaning the data (removing zeros and N/As)
dim(train_set)
dim(test_set)

zero_removal <- nearZeroVar(train_set)
train_set <- train_set[, -zero_removal]
test_set  <- test_set[, -zero_removal]

NA_removal <- sapply(train_set, function(x) mean(is.na(x))) > 0.95
train_set <- train_set[, NA_removal==FALSE]
test_set  <- test_set[, NA_removal==FALSE]

train_set <- train_set[, -(1:5)]
test_set  <- test_set[, -(1:5)]

dim(train_set)
dim(test_set)


```


##**Prediction Model Building**

We will now build and test three models; random forest, decision trees, and generalized boost. Which ever model proves most accurate we will use for the quiz predictions. This will also prove our cross validation process as the models will be compared against each other coupled with graphs. (Please see "images" folder for graphs) 

Lastly a confusion matrix is plotted at the end of each analysis to confirm our model. (Please see "images" folder for graphs)

```{r, echo=TRUE}

set.seed(12345)
random_forest_control <- trainControl(method = "cv", 
                          number = 3, 
                          verboseIter = FALSE)

random_forest_model <- train(classe ~ ., 
                             data = train_set, 
                             method = "rf", 
                             trControl = random_forest_control)

random_forest_model$finalModel


#Prediction on Test dataset

random_forest_predict <- predict(random_forest_model, 
                                 newdata = test_set)

random_forest_confusion_matrix <- confusionMatrix(random_forest_predict, 
                                                  test_set$classe)
random_forest_confusion_matrix

plot(random_forest_confusion_matrix$table, 
          col = random_forest_confusion_matrix$byClass, 
          main = paste("Random Forest - Accuracy =", 
          round(random_forest_confusion_matrix$overall['Accuracy'], 4)))

```

Decision Trees


```{r, echo=TRUE}
set.seed(12345)
decision_tree_model <- rpart(classe ~ ., 
                             data = train_set, 
                             method = "class")

fancyRpartPlot(decision_tree_model)

#Prediction on Test dataset

decision_tree_predict <- predict(decision_tree_model, 
                                 newdata = test_set, 
                                 type="class")

decision_tree_confusion_matrix <- confusionMatrix(decision_tree_predict, 
                                                  test_set$classe)
decision_tree_confusion_matrix

plot(decision_tree_confusion_matrix$table, 
            col = decision_tree_confusion_matrix$byClass, 
            main = paste("Decision Tree - Accuracy =", 
            round(decision_tree_confusion_matrix$overall['Accuracy'], 4)))

```

Generalized Boost Model


```{r, echo=TRUE}

set.seed(12345)
boost_model_control <- trainControl(method = "repeatedcv", 
                                    number = 5, 
                                    repeats = 1)

#Prediction on Test dataset

boost_model_predict  <- train(classe ~ ., 
                              data = train_set, 
                              method = "gbm", 
                              trControl = boost_model_control, 
                              verbose = FALSE)

boost_model_predict$finalModel


boost_model_predict <- predict(boost_model_predict,
                               newdata = test_set)

boost_model_confusion_matrix <- confusionMatrix(boost_model_predict, test_set$classe)

boost_model_confusion_matrix

plot(boost_model_confusion_matrix$table, 
               col = boost_model_confusion_matrix$byClass, 
               main = paste("Generlized Boost Matrix - Accuracy =", 
              round(boost_model_confusion_matrix$overall['Accuracy'], 4)))

```


##**Selecting a Model to the Test Data**


Selecting a model to apply to the quiz. 


After our testing the Random Forest model bested and will be applied to predict quiz. As I expected we did get a 90% accuracy but on 2 model which I did not expect.


```{r, echo=TRUE}

quiz_model <- predict(random_forest_model, 
                      newdata = testing)
quiz_model

```
