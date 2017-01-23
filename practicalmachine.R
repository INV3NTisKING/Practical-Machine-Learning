
#Loading and Processing the Data

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

#Prediction and Model Building

#Random Forest model build

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
#Random Forest 

random_forest_predict <- predict(random_forest_model, 
                                 newdata = test_set)

random_forest_confusion_matrix <- confusionMatrix(random_forest_predict, 
                                                  test_set$classe)
random_forest_confusion_matrix

plot(random_forest_confusion_matrix$table, 
     col = random_forest_confusion_matrix$byClass, 
     main = paste("Random Forest - Accuracy =",
                  round(random_forest_confusion_matrix$overall['Accuracy'], 4)))

#Decision Tree
#Decision Tree model build

set.seed(12345)
decision_tree_model <- rpart(classe ~ ., 
                             data = train_set, 
                             method = "class")

fancyRpartPlot(decision_tree_model)

#Prediction on Test dataset
#Decision Tree

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

#Generalized Boost Model

set.seed(12345)
boost_model_control <- trainControl(method = "repeatedcv", 
                                    number = 5, 
                                    repeats = 1)

#Prediction on Test dataset
#Generalized Boost Model

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

predict_model <- predict(random_forest_model, newdata = testing)
predict_model
