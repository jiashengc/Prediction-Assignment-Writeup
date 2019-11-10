---
title: "Prediction Assignment Writeup"
author: "Jia Sheng Chong"
date: "10/30/2019"
output:
  html_document:
    keep_md: true
---


```r
setwd("C:/Users/jiash/Dropbox/R-repo/c8-cp1/")
library(ggplot2)
library(GGally)
```

```
## Registered S3 method overwritten by 'GGally':
##   method from   
##   +.gg   ggplot2
```

```r
library(caret)
```

```
## Loading required package: lattice
```

```r
library(klaR)
```

```
## Loading required package: MASS
```

```r
library(kernlab)
```

```
## 
## Attaching package: 'kernlab'
```

```
## The following object is masked from 'package:ggplot2':
## 
##     alpha
```

```r
library(corrplot)
```

```
## corrplot 0.84 loaded
```

```r
library(knitr)
library(readr)

set.seed(666)
```

## Overview
The purpose of this report is to predict the manner in which a person has done an exercise. The main aim of this report is to predict the manner of 6 different particpants and their exercise.


## Data Analysis

```r
raw <- data.frame()
raw.test  <- read_csv(file = './data/pml_testing.csv', col_names = TRUE)
```

```
## Warning: Missing column names filled in: 'X1' [1]
```

```
## Parsed with column specification:
## cols(
##   .default = col_logical(),
##   X1 = col_double(),
##   user_name = col_character(),
##   raw_timestamp_part_1 = col_double(),
##   raw_timestamp_part_2 = col_double(),
##   cvtd_timestamp = col_character(),
##   new_window = col_character(),
##   num_window = col_double(),
##   roll_belt = col_double(),
##   pitch_belt = col_double(),
##   yaw_belt = col_double(),
##   total_accel_belt = col_double(),
##   gyros_belt_x = col_double(),
##   gyros_belt_y = col_double(),
##   gyros_belt_z = col_double(),
##   accel_belt_x = col_double(),
##   accel_belt_y = col_double(),
##   accel_belt_z = col_double(),
##   magnet_belt_x = col_double(),
##   magnet_belt_y = col_double(),
##   magnet_belt_z = col_double()
##   # ... with 40 more columns
## )
```

```
## See spec(...) for full column specifications.
```

```r
raw.train <- read_csv(file = './data/pml_training.csv', col_names = TRUE)
```

```
## Warning: Missing column names filled in: 'X1' [1]
```

```
## Parsed with column specification:
## cols(
##   .default = col_double(),
##   user_name = col_character(),
##   cvtd_timestamp = col_character(),
##   new_window = col_character(),
##   kurtosis_roll_belt = col_character(),
##   kurtosis_picth_belt = col_character(),
##   kurtosis_yaw_belt = col_character(),
##   skewness_roll_belt = col_character(),
##   skewness_roll_belt.1 = col_character(),
##   skewness_yaw_belt = col_character(),
##   max_yaw_belt = col_character(),
##   min_yaw_belt = col_character(),
##   amplitude_yaw_belt = col_character(),
##   kurtosis_picth_arm = col_character(),
##   kurtosis_yaw_arm = col_character(),
##   skewness_pitch_arm = col_character(),
##   skewness_yaw_arm = col_character(),
##   kurtosis_yaw_dumbbell = col_character(),
##   skewness_yaw_dumbbell = col_character(),
##   kurtosis_roll_forearm = col_character(),
##   kurtosis_picth_forearm = col_character()
##   # ... with 8 more columns
## )
## See spec(...) for full column specifications.
```

```
## Warning: 182 parsing failures.
##  row               col expected  actual                      file
## 2231 kurtosis_roll_arm a double #DIV/0! './data/pml_training.csv'
## 2231 skewness_roll_arm a double #DIV/0! './data/pml_training.csv'
## 2255 kurtosis_roll_arm a double #DIV/0! './data/pml_training.csv'
## 2255 skewness_roll_arm a double #DIV/0! './data/pml_training.csv'
## 2282 kurtosis_roll_arm a double #DIV/0! './data/pml_training.csv'
## .... ................. ........ ....... .........................
## See problems(...) for more details.
```

```r
dim(raw.test)
```

```
## [1]  20 160
```

```r
dim(raw.train)
```

```
## [1] 19622   160
```

```r
# Calculating the percentage of test to training data provided
(dim(raw.test)[1]/dim(raw.train)[1]) * 100
```

```
## [1] 0.1019264
```

```r
# Percentage of classes
percentage <- prop.table(table(raw.train$classe)) * 100
cbind(freq=table(raw.train$classe), percentage = percentage)
```

```
##   freq percentage
## A 5580   28.43747
## B 3797   19.35073
## C 3422   17.43961
## D 3216   16.38977
## E 3607   18.38243
```
We can observe there are 160 columns for the data provided to us. The test data we have been given contains 20 rows, while the training data contains 19622 rows. However, because the percentage of testing data to training data is 0.1%, I will use the training data to create my own test data and perform cross validation. After which, I will use the final model created to predict the test data that was initially provided. 

### Cleaning Data
As the given testing data contains columns which only contains NA values, we will remove these columns as they are worthless for predicting our test data. We also remove the first 5 columns as they provide no value to our models.

```r
# Clean data, clean data which contains only NA values
clean <- data.frame()
keepCol <- colSums(is.na(raw.test)) < nrow(raw.test)
clean.test <- raw.test[,keepCol]
clean.train <- raw.train[,keepCol]

# Remove useless information
clean.test <- clean.test[,-c(1:5)]
clean.train <- clean.train[,-c(1:5)]

clean.train$classe <- factor(clean.train$classe)

# Remove any data that contains NA in the any columns
clean.trainNoNA <- data.frame(na.omit(clean.train))
dim(clean.trainNoNA)
```

```
## [1] 19622    55
```

## Using Cross Validation
To train and test our models, we will use cross validation (5 folds). We use cross validation so that we can use our model to predict over all our data. Using this method means that we do not have to split our data as the cross validation already does that for us.

```r
forTrain.data <- clean.trainNoNA
forTrain.control <- trainControl(method = 'cv', number = 5)
```

## Training and building the model
To build our final model, I will use 3 different algorithms, knn, svm and lda.

```r
predictions <- data.frame()

model.knn <- train(classe ~ ., data = forTrain.data, method = 'knn', trControl = forTrain.control)

model.svm <- train(classe ~ ., data = forTrain.data, method = 'svmRadial', trControl = forTrain.control)

model.lda <- train(classe ~ ., data = forTrain.data, method = 'lda', trControl = forTrain.control)

# Compare accuracy
results <- resamples(list(knn=model.knn, svm=model.svm, lda=model.lda))
summary(results)
```

```
## 
## Call:
## summary.resamples(object = results)
## 
## Models: knn, svm, lda 
## Number of resamples: 5 
## 
## Accuracy 
##          Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
## knn 0.9237828 0.9261146 0.9281346 0.9295681 0.9342508 0.9355578    0
## svm 0.9296636 0.9345390 0.9347604 0.9355315 0.9378027 0.9408917    0
## lda 0.7029299 0.7082059 0.7094801 0.7114463 0.7168705 0.7197452    0
## 
## Kappa 
##          Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
## knn 0.9035931 0.9064628 0.9090757 0.9108935 0.9168766 0.9184592    0
## svm 0.9109186 0.9170941 0.9174028 0.9183576 0.9212610 0.9251115    0
## lda 0.6245159 0.6307261 0.6325651 0.6349270 0.6415957 0.6452323    0
```

```r
summary(model.svm)
```

```
## Length  Class   Mode 
##      1   ksvm     S4
```
After training our models, I compare them using the resamples function. In the summary of our results, we can see that performed the best accuracy over knn and lda. Hence, we will use svm to predict the assignment's test data.

## Sample Error
Using our svm model to predict our training model, we get an out of sample error of 5.28%.

```r
trainPredict <- predict(model.svm, newdata = clean.train)
confusionMatrix(trainPredict, clean.train$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 5517  249    9    7    1
##          B    9 3449  111    0    9
##          C   52   89 3270  297   66
##          D    1    2   28 2910   92
##          E    1    8    4    2 3439
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9472          
##                  95% CI : (0.9439, 0.9502)
##     No Information Rate : 0.2844          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9331          
##                                           
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9887   0.9083   0.9556   0.9049   0.9534
## Specificity            0.9811   0.9918   0.9689   0.9925   0.9991
## Pos Pred Value         0.9540   0.9639   0.8665   0.9594   0.9957
## Neg Pred Value         0.9954   0.9783   0.9904   0.9816   0.9896
## Prevalence             0.2844   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2812   0.1758   0.1666   0.1483   0.1753
## Detection Prevalence   0.2947   0.1823   0.1923   0.1546   0.1760
## Balanced Accuracy      0.9849   0.9501   0.9622   0.9487   0.9762
```

## Final Prediction

```r
finalPredict <- predict(model.svm, newdata = clean.test)
finalPredict
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

