---
title: "MachineLearningProject_LCP"
date: "December 16, 2014"
output: html_document
---

Background
========================================================

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset). 

Data 
=====

The training data for this project are available here: 

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

The test data are available here: 

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

The data for this project come from this source: http://groupware.les.inf.puc-rio.br/har. If you use the document you create for this class for any purpose please cite them as they have been very generous in allowing their data to be used for this kind of assignment. 

What you should submit
=======================

The goal of your project is to predict the manner in which they did the exercise. This is the "classe" variable in the training set. You may use any of the other variables to predict with. You should create a report describing how you built your model, how you used cross validation, what you think the expected out of sample error is, and why you made the choices you did. You will also use your prediction model to predict 20 different test cases. 

1. Your submission should consist of a link to a Github repo with your R markdown and compiled HTML file describing your analysis. Please constrain the text of the writeup to < 2000 words and the number of figures to be less than 5. It will make it easier for the graders if you submit a repo with a gh-pages branch so the HTML page can be viewed online (and you always want to make it easy on graders :-).
2. You should also apply your machine learning algorithm to the 20 test cases available in the test data above. Please submit your predictions in appropriate format to the programming assignment for automated grading. See the programming assignment for additional details. 

Reproducibility 
================

Due to security concerns with the exchange of R code, your code will not be run during the evaluation by your classmates. Please be sure that if they download the repo, they will be able to view the compiled HTML version of your analysis. 

RESULTS
==========

**1.Load required libraries and data files containing the data:**


```r
# install.packages("caret")
# install.packages("randomForest")
# install.packages("rpart")
# install.packages("kernlab")
# install.packages("rattle")
# install.packages("ipred")
library(caret)
```

```
## Warning: package 'caret' was built under R version 3.1.2
```

```
## Loading required package: lattice
## Loading required package: ggplot2
## Find out what's changed in ggplot2 with
## news(Version == "1.0.0", package = "ggplot2")
## 
## Attaching package: 'ggplot2'
## 
## The following object is masked _by_ '.GlobalEnv':
## 
##     mpg
```

```r
library(randomForest)
```

```
## Warning: package 'randomForest' was built under R version 3.1.1
```

```
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
```

```r
library(rpart)
library(rpart.plot)
```

```
## Warning: package 'rpart.plot' was built under R version 3.1.2
```

```r
library(rattle)
```

```
## Warning: package 'rattle' was built under R version 3.1.1
```

```
## Rattle: A free graphical interface for data mining with R.
## Version 3.3.0 Copyright (c) 2006-2014 Togaware Pty Ltd.
## Type 'rattle()' to shake, rattle, and roll your data.
```

```r
library(kernlab)
library(ipred)

#Set initial seed for reproducibility
set.seed(54321)

workingDir <- "/Volumes/Data/E-courses/Johns\ Hopkins\ University\ Data\ Science/Practical_Machine_Learning/Project"
#Set the working directory in your computer
setwd(workingDir)

#Check if the files containing the data is already in your working directory. Otherwise, download it, save as a csv file and get the data replacing all missing values as NA
trainingDataURL<-"https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
trainFilename<-"trainData.csv"
#Check if the file containing the data is already in your working directory. Otherwise, download it, save as a csv file and get the data replacing all missing values as NA
if (file.exists(trainFilename)) {
        trainingData<-read.csv(trainFilename, na.strings=c("NA","#DIV/0!", ""))     
} else {
        download.file(trainingDataURL, destfile = trainFilename, method = "curl")
        trainingData<-read.csv(trainFilename, na.strings=c("NA","#DIV/0!", ""))  
}

testingDataURL<-"https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
testFilename<-"testData.csv"
if (file.exists(testFilename)) {
        testingData<-read.csv(testFilename, na.strings=c("NA","#DIV/0!", ""))    
} else {
        download.file(testingDataURL, destfile = testFilename, method = "curl")
        testingData<-read.csv(testFilename, na.strings=c("NA","#DIV/0!", ""))
}
```

**2. Preparing the data:**

First of all, I clean the data removing columns only containing missing values (NAs) and unnecessary variables related to data acquisition 
(i.e., [X, user_name, raw_timestamp_part_1, raw_timestamp_part_2, cvtd_timestamp, new_window, num_window]).
Variables with low variance (Near Zero Variance), were also removed.


```r
#Delete columns with all missing values
trainingData<-trainingData[,colSums(is.na(trainingData)) == 0]
testingData <-testingData[,colSums(is.na(testingData)) == 0]

#Discard unnecessary variables
trainingData   <-trainingData[,-c(1:7)]
testingData <-testingData[,-c(1:7)]

#Removing near zero covariatees
nearzerovarTr <- nearZeroVar(trainingData, saveMetrics = TRUE)
trainingData <- trainingData[, !nearzerovarTr$nzv]

nearzerovarTe <- nearZeroVar(trainingData, saveMetrics = TRUE)
testingData <- testingData[, !nearzerovarTe$nzv]

dim(trainingData)
```

```
## [1] 19622    53
```

```r
dim(testingData)
```

```
## [1] 20 53
```

After cleaning the data, **53** variables were selected to define the model, i.e., **roll_belt, pitch_belt, yaw_belt, total_accel_belt, gyros_belt_x, gyros_belt_y, gyros_belt_z, accel_belt_x, accel_belt_y, accel_belt_z, magnet_belt_x, magnet_belt_y, magnet_belt_z, roll_arm, pitch_arm, yaw_arm, total_accel_arm, gyros_arm_x, gyros_arm_y, gyros_arm_z, accel_arm_x, accel_arm_y, accel_arm_z, magnet_arm_x, magnet_arm_y, magnet_arm_z, roll_dumbbell, pitch_dumbbell, yaw_dumbbell, total_accel_dumbbell, gyros_dumbbell_x, gyros_dumbbell_y, gyros_dumbbell_z, accel_dumbbell_x, accel_dumbbell_y, accel_dumbbell_z, magnet_dumbbell_x, magnet_dumbbell_y, magnet_dumbbell_z, roll_forearm, pitch_forearm, yaw_forearm, total_accel_forearm, gyros_forearm_x, gyros_forearm_y, gyros_forearm_z, accel_forearm_x, accel_forearm_y, accel_forearm_z, magnet_forearm_x, magnet_forearm_y, magnet_forearm_z, classe**.

Next, I partition the training data set into two subdata sets (i.e., 70% for inTraining, 30% for inTesting) to perform cross-validation. 


```r
#Data splitting into two subdata sets

inTrain <- createDataPartition(y=trainingData$classe, p=0.7, list=FALSE)
subTraining <- trainingData[inTrain, ]
subTesting <- trainingData[-inTrain, ]
dim(subTraining)
```

```
## [1] 13737    53
```

```r
dim(subTesting)
```

```
## [1] 5885   53
```

**3. Using different classification algorithms to generate the prediction model:**

I next examined the fit of three different models (Decision Tree, Bagging and Random Forest) to the subTraining data set, and then I tested them on the subTesting data. The best fitting model will be finally used on the test data.

*3.1 Using a decision tree as first prediction model:*


```r
#Fitting the model
modDT <- rpart(classe ~ ., method = "class", data = subTraining)

#Plotting of the decision tree
rpart.plot(modDT, main="Classification Tree", cex = 1.5, under = TRUE) #faclen=0
```

![plot of chunk unnamed-chunk-4](figure/unnamed-chunk-41.png) 

```r
#Fancy plotting of the decision tree
fancyRpartPlot(modDT, main="Classification Tree", cex = 1.15, under = TRUE)
```

![plot of chunk unnamed-chunk-4](figure/unnamed-chunk-42.png) 

```r
#Predicting on the testing subdata set
predictionDT <- predict(modDT, subTesting, type = "class")
cMDT <- confusionMatrix(predictionDT,subTesting$classe)
cMDT
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1500  259   20  108   48
##          B   32  626   85   43   80
##          C   43  117  825  122  137
##          D   55   86   63  630   56
##          E   44   51   33   61  761
## 
## Overall Statistics
##                                         
##                Accuracy : 0.738         
##                  95% CI : (0.726, 0.749)
##     No Information Rate : 0.284         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.667         
##  Mcnemar's Test P-Value : <2e-16        
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.896    0.550    0.804    0.654    0.703
## Specificity             0.897    0.949    0.914    0.947    0.961
## Pos Pred Value          0.775    0.723    0.663    0.708    0.801
## Neg Pred Value          0.956    0.898    0.957    0.933    0.935
## Prevalence              0.284    0.194    0.174    0.164    0.184
## Detection Rate          0.255    0.106    0.140    0.107    0.129
## Detection Prevalence    0.329    0.147    0.211    0.151    0.161
## Balanced Accuracy       0.896    0.750    0.859    0.800    0.832
```

*3.2 Using Bagging as second prediction model:*


```r
#Fitting the model
modBA <- bagging(classe ~ ., method = "class", data = subTraining)

#Predicting on the testing subdata set and printing the resulting confusion matrix and statistics
predictionBA <- predict(modBA, subTesting, type = "class")
cMBA <- confusionMatrix(predictionBA,subTesting$classe)
cMBA
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1669    9    1    2    0
##          B    2 1119    6    2    2
##          C    2    9 1008    7    2
##          D    1    2    8  953    2
##          E    0    0    3    0 1076
## 
## Overall Statistics
##                                         
##                Accuracy : 0.99          
##                  95% CI : (0.987, 0.992)
##     No Information Rate : 0.284         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.987         
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.997    0.982    0.982    0.989    0.994
## Specificity             0.997    0.997    0.996    0.997    0.999
## Pos Pred Value          0.993    0.989    0.981    0.987    0.997
## Neg Pred Value          0.999    0.996    0.996    0.998    0.999
## Prevalence              0.284    0.194    0.174    0.164    0.184
## Detection Rate          0.284    0.190    0.171    0.162    0.183
## Detection Prevalence    0.286    0.192    0.175    0.164    0.183
## Balanced Accuracy       0.997    0.990    0.989    0.993    0.997
```

*3.3 Using random forest (improved bagging) as third prediction model:*


```r
#Fitting the model
modRF <- randomForest(classe ~ ., method = "class", data = subTraining)
#Predicting on the testing subdata set and printing the resulting confusion matrix and statistics
predictionRF <- predict(modRF, subTesting, type = "class")
cMRF <- confusionMatrix(predictionRF,subTesting$classe)
cMRF
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1672    3    0    0    0
##          B    2 1133    4    0    0
##          C    0    3 1022    7    0
##          D    0    0    0  957    4
##          E    0    0    0    0 1078
## 
## Overall Statistics
##                                         
##                Accuracy : 0.996         
##                  95% CI : (0.994, 0.998)
##     No Information Rate : 0.284         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.995         
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.999    0.995    0.996    0.993    0.996
## Specificity             0.999    0.999    0.998    0.999    1.000
## Pos Pred Value          0.998    0.995    0.990    0.996    1.000
## Neg Pred Value          1.000    0.999    0.999    0.999    0.999
## Prevalence              0.284    0.194    0.174    0.164    0.184
## Detection Rate          0.284    0.193    0.174    0.163    0.183
## Detection Prevalence    0.285    0.194    0.175    0.163    0.183
## Balanced Accuracy       0.999    0.997    0.997    0.996    0.998
```

**4. Comparing the different models and selecting the best fitting one:**


```r
dfAccuracies <- data.frame(cMDT$overall, cMBA$overall, cMRF$overall)
dfAccuracies
```

```
##                cMDT.overall cMBA.overall cMRF.overall
## Accuracy          7.378e-01       0.9898       0.9961
## Kappa             6.667e-01       0.9871       0.9951
## AccuracyLower     7.264e-01       0.9869       0.9941
## AccuracyUpper     7.490e-01       0.9922       0.9975
## AccuracyNull      2.845e-01       0.2845       0.2845
## AccuracyPValue    0.000e+00       0.0000       0.0000
## McnemarPValue     6.165e-61          NaN          NaN
```

BY comparing the accuracies of the different models, I select random forest as the one displaying the highest one (Accuracy: **0.9961**), with a 95% confidence interval: **0.9941 - 0.9975**. As the estimated out-of-sample error rate of the Random Forest trained model (1-Accuracy) is less than 1%, we expect very few errors in predictions. Finally, I display the best fitting final model:


```r
#List the variables by relative importance
varImp(modRF)
```

```
##                      Overall
## roll_belt             917.51
## pitch_belt            489.33
## yaw_belt              615.09
## total_accel_belt      152.84
## gyros_belt_x           64.65
## gyros_belt_y           80.84
## gyros_belt_z          207.31
## accel_belt_x           86.38
## accel_belt_y           92.36
## accel_belt_z          262.30
## magnet_belt_x         165.92
## magnet_belt_y         277.19
## magnet_belt_z         273.38
## roll_arm              220.32
## pitch_arm             125.13
## yaw_arm               168.32
## total_accel_arm        73.75
## gyros_arm_x            91.09
## gyros_arm_y            95.01
## gyros_arm_z            43.30
## accel_arm_x           176.14
## accel_arm_y           108.95
## accel_arm_z            99.73
## magnet_arm_x          178.83
## magnet_arm_y          161.87
## magnet_arm_z          130.89
## roll_dumbbell         293.93
## pitch_dumbbell        128.14
## yaw_dumbbell          169.66
## total_accel_dumbbell  198.30
## gyros_dumbbell_x       88.41
## gyros_dumbbell_y      169.37
## gyros_dumbbell_z       61.14
## accel_dumbbell_x      185.18
## accel_dumbbell_y      282.39
## accel_dumbbell_z      232.99
## magnet_dumbbell_x     353.89
## magnet_dumbbell_y     450.69
## magnet_dumbbell_z     522.37
## roll_forearm          424.23
## pitch_forearm         543.78
## yaw_forearm           123.75
## total_accel_forearm    74.35
## gyros_forearm_x        54.38
## gyros_forearm_y        92.13
## gyros_forearm_z        61.81
## accel_forearm_x       227.85
## accel_forearm_y        97.21
## accel_forearm_z       163.88
## magnet_forearm_x      154.65
## magnet_forearm_y      154.66
## magnet_forearm_z      191.95
```



**5. Apply the best fitting model to the test data:**

I then applied the machine Random Forest learning algorithm I built to each of the 20 test cases in the testing data set.

```r
# Predict testing results using trained best fitting model
results <- predict(modRF, testingData, type="class")
results
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```

```r
#Save the best fitting model model into a file for subsequent usage
save(modRF, file="modRF.R")

resultsfolder <- "/Volumes/Data/E-courses/Johns\ Hopkins\ University\ Data\ Science/Practical_Machine_Learning/Project/Answers"
setwd(resultsfolder)

pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

pml_write_files(results)
```
