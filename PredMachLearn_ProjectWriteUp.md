# Can a machine learning algorithm predict activity quality from activity monitors?
tomassve  
Saturday, August 16, 2014  

### Introduction
It is now possible to collect a large amount of data about personal activity relatively inexpensively. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. 

In this study data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants were used. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset) [1]

The following question was addressed in this project assignment: Can a machine learning algorithm predict activity quality from activity monitors?



```r
rm(list=ls())
invisible(Sys.setlocale('LC_ALL', 'C'))
require(caret)
```

### Data

Data was downloaded from the Coursera site using the code below.
 

```r
srcDir <- "http://d396qusza40orc.cloudfront.net/predmachlearn"
dstDir <- "./data"
trainFile <- "pml-training.csv" 
testFile  <- "pml-testing.csv"

if (!file.exists(paste(dstDir, trainFile, sep = "/"))) {
    download.file(paste(srcDir, trainFile, sep = "/"), 
        destfile = paste(dstDir, trainFile, sep = "/"))
}
if (!file.exists(paste(dstDir, testFile, sep = "/"))) {
    download.file(paste(srcDir, testFile, sep = "/"), 
        destfile = paste(dstDir, testFile, sep = "/"))
}
```


It was then read into R for further processing and analysis. 


```r
training <- read.csv(paste(dstDir, trainFile, sep = "/"))
testing <- read.csv(paste(dstDir, testFile, sep = "/"))
```


#### Data characteristics

Two sets of data were supplied, one for training and one for testing. The size of the two sets are shown in the table below.


```r
dataInfo<-rbind(training=dim(training),testing=dim(testing))
colnames(dataInfo)<-c('observations', 'variables')
knitr::kable(dataInfo, format = "html", caption = "Size of supplied data sets")
```

<table>
<caption>Size of supplied data sets</caption>
 <thead>
  <tr>
   <th align="left">   </th>
   <th align="right"> observations </th>
   <th align="right"> variables </th>
  </tr>
 </thead>
<tbody>
  <tr>
   <td align="left"> training </td>
   <td align="right"> 19622 </td>
   <td align="right"> 160 </td>
  </tr>
  <tr>
   <td align="left"> testing </td>
   <td align="right"> 20 </td>
   <td align="right"> 160 </td>
  </tr>
</tbody>
</table>

As shown in the table below were as many as 67 out of the 160 variables mostly NA. 
The rest of the variables were free from NAs.


```r
knitr::kable(as.data.frame(table(colSums(is.na(training)), dnn=c('NAs'))),
             format = "html")
```

<table>
 <thead>
  <tr>
   <th align="left"> NAs </th>
   <th align="right"> Freq </th>
  </tr>
 </thead>
<tbody>
  <tr>
   <td align="left"> 0 </td>
   <td align="right"> 93 </td>
  </tr>
  <tr>
   <td align="left"> 19216 </td>
   <td align="right"> 67 </td>
  </tr>
</tbody>
</table>
Only 2.1% of the 
observations were complete.  Removing those variables with mostly 
NA is preferable to throwing away 98% of the data away.

The distribution of outcome is shown in the table below.


```r
knitr::kable(as.data.frame(table(training$classe,dnn=c('Class'))), format = "html" )
```

<table>
 <thead>
  <tr>
   <th align="left"> Class </th>
   <th align="right"> Freq </th>
  </tr>
 </thead>
<tbody>
  <tr>
   <td align="left"> A </td>
   <td align="right"> 5580 </td>
  </tr>
  <tr>
   <td align="left"> B </td>
   <td align="right"> 3797 </td>
  </tr>
  <tr>
   <td align="left"> C </td>
   <td align="right"> 3422 </td>
  </tr>
  <tr>
   <td align="left"> D </td>
   <td align="right"> 3216 </td>
  </tr>
  <tr>
   <td align="left"> E </td>
   <td align="right"> 3607 </td>
  </tr>
</tbody>
</table>


#### Data Processing
After data was read into R, it was preprocessed. The data was cleaned by removing variables:

1. with mostly NA values to get many complete observations
2. with no or low variation
3. not recorded from the sensors (e.g. *X*, *user_name*, *raw_timestamp_part_1*, 
*raw_timestamp_part_2* *cvtd_timestamp*, *new_window*, *num_window*).


#### Removing variables with mostly NAs and those unrelated to the sensors


```r
useVariables <- colSums(is.na(training)) == 0
useVariables[1:7] <- FALSE
use.training <- training[,useVariables]
use.testing  <- testing[,useVariables]
```

#### Removing variables with no or little variation


```r
tbl <- nearZeroVar(use.training, saveMetrics = TRUE)
nzv <- which(tbl$zeroVar | tbl$nzv)
use.training <- use.training[,-nzv]
use.testing <- use.testing[,-nzv]
```

The training set had 19622 observations of 
53 variables after the preprocessing was completed.

### Methods
The strategy was to pick the most accurate method out of three, by use of 
cross validation. The supplied training set was split into two sets, one for
training (90%) and one for validation (10%). All three models were trained 
on the first set of data and the validated on the second validation set.



```r
set.seed(4711)
index <- createDataPartition(use.training$classe, list = FALSE, p = 0.9)
use.validation <- use.training[-index, ]
use.training   <- use.training[index, ]
```


#### Model selection
Boosting and random forests are the most common tools that win Kaggle and other
prediction contests and were selected. These are computational expensive and
a regression tree was included as less complex alternative.

The seed was set prior to training in order to get reproducible results
since some of the methods are not deterministic.

##### Regression Trees

```r
set.seed(125)
sysTime1<-system.time(model1 <- train(use.training$classe ~ .,method="rpart",
                                     data=subset(use.training, select=-c(classe))))
```


##### Random forest

```r
set.seed(125)
sysTime2<-system.time(model2 <- train(use.training$classe ~ .,method="rf",
                                      data=subset(use.training, select=-c(classe))))
```

```
## Loading required package: randomForest
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
```


##### Generalized boosted regression models

```r
set.seed(125)
sysTime3<-system.time(model3 <- train(use.training$classe ~ .,method="gbm", 
                                     subset(use.training, select=-c(classe)), verbose=FALSE))
```

```
## Loading required package: gbm
## Loading required package: survival
## Loading required package: splines
## 
## Attaching package: 'survival'
## 
## The following object is masked from 'package:caret':
## 
##     cluster
## 
## Loading required package: parallel
## Loaded gbm 2.1
## Loading required package: plyr
```


```r
elapsedTime<-rbind("RegressionTree"=sysTime1,
                   "RandomForest"=sysTime2,
                   "GeneralizedBoostedReg"=sysTime3)
```


### Results

#### In-sample accuracy
The in-samples accuracy is calculated for each model on the validation set. 

```r
trainingPrediction1 <- predict(model1, use.training)
trainingPrediction2 <- predict(model2, use.training)
trainingPrediction3 <- predict(model3, use.training)

insampleAcc <- rbind(
    RegressionTree=confusionMatrix(trainingPrediction1, use.training$classe)$overall[1],
    RandomForest=confusionMatrix(trainingPrediction2, use.training$classe)$overall[1],
    GeneralizedBoostedReg=confusionMatrix(trainingPrediction3, use.training$classe)$overall[1])

colnames(insampleAcc)<-c("In-sample")
```


#### Out-of-sample accuracy
The out-of-samples accuracy is calculated for each model on the independent training set.

```r
validationPrediction1 <- predict(model1, use.validation)
validationPrediction2 <- predict(model2, use.validation)
validationPrediction3 <- predict(model3, use.validation)

oosSampleAcc <- rbind(
    RegressionTree=confusionMatrix(validationPrediction1, use.validation$classe)$overall[1],
    RandomForest=confusionMatrix(validationPrediction2, use.validation$classe)$overall[1],
    GeneralizedBoostedReg=confusionMatrix(validationPrediction3, use.validation$classe)$overall[1])
colnames(oosSampleAcc)<-c("Out-of-sample")
```

The in-sample accuracy as well as the out-of-sample accuracy is reported in the
table below. It can be found that both the Boosting and the Random Forest 
outperformed the Regression Tree.


```r
knitr::kable(cbind(insampleAcc,oosSampleAcc), format = "html",
             caption = "Accuracy")
```

<table>
<caption>Accuracy</caption>
 <thead>
  <tr>
   <th align="left">   </th>
   <th align="right"> In-sample </th>
   <th align="right"> Out-of-sample </th>
  </tr>
 </thead>
<tbody>
  <tr>
   <td align="left"> RegressionTree </td>
   <td align="right"> 0.5209 </td>
   <td align="right"> 0.5240 </td>
  </tr>
  <tr>
   <td align="left"> RandomForest </td>
   <td align="right"> 1.0000 </td>
   <td align="right"> 0.9929 </td>
  </tr>
  <tr>
   <td align="left"> GeneralizedBoostedReg </td>
   <td align="right"> 0.9747 </td>
   <td align="right"> 0.9602 </td>
  </tr>
</tbody>
</table>


```r
confusionMatrix(validationPrediction2, use.validation$classe)$table 
```

```
          Reference
Prediction   A   B   C   D   E
         A 558   3   0   0   0
         B   0 375   1   0   0
         C   0   1 341   7   0
         D   0   0   0 314   2
         E   0   0   0   0 358
```

```r
confusionMatrix(validationPrediction2, use.validation$classe)$overall[1:4]
```

```
     Accuracy         Kappa AccuracyLower AccuracyUpper 
       0.9929        0.9910        0.9880        0.9961 
```


### Discussions and Conclusions
The Random Forest performed best. It has an accuracy of 0.993 on the independent 
validation set. The 95% CI is 0.988 to 0.996. The out-of-sample error is expected
to be about 1%. 

The model is so good that it will most likely correctly classify all 20 samples in the supplied test set.

#### Test set
Result are saved to disk and not disclosed.


```r
testResults<-predict(model2, subset(use.testing, select = -c(problem_id)))
save(testResults, file="testResults")
```


### References

1. Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. Qualitative Activity Recognition of Weight Lifting Exercises. Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human 13) . Stuttgart, Germany: ACM SIGCHI, 2013.
