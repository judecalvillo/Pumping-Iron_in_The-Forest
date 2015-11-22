## Pumping Iron in the Forest: A Machine Learning "Exercise." :)
by Jude Calvillo  

Below, I will build, cross-validate, and assess a "random forest" predictive model. This model aims to predict a physical exercise outcome, type of bicep curl (5 different types), from fitness tracker data (e.g. Fitbit) that initially includes 160 variables (yikes!). 

### Pre-Processing / Cleaning
As you can see, the source dataset is rather large. More importantly, it includes a slew of variables that might predict the outcome of concern, the last variable: classe. Therefore, we should cut this down.


```r
## Load libraries: Caret, for machine learning, and dplyr for convenience. :)
library(caret)
library(dplyr)

## Get the data, size up, and preview.
the_data <- read.csv("data/pml-training.csv", na.strings = c("NA", ""))
# print(head(the_data)) # Not going to preview in markdown (too large).
print(dim(the_data))
```

```
## [1] 19622   160
```

##### Removing Useless Columns (variables)
It looks like, upon preview (too large to show here), the source data includes many columns that are made purely of NA values. Therefore, let's find them, remove them, and see how many useful variables our cleaned data frame ends up with.


```r
## Here, we get the sum of NAs per column (note, the margin argument of Apply
## function = 2, for columns only; default is 1, for rows/lists). Then, we replace
## the original df w/the a cleaner version of itself.
na_colsum <- apply(the_data, 2, function(x){sum(is.na(x))})
the_data <- the_data[, which(na_colsum == 0)]

## Size up and preview.
# print(head(the_data)) # Not going to preview in markdown (too large).
print(dim(the_data))
```

```
## [1] 19622    60
```

### Create the Training and 'Test' (Validation) Sets
Here, we simply partition the cleaned dataset, for the training set, and leave the remainder to the test/validation set.


```r
## Create the training set partition.
train_sample <- createDataPartition(y = the_data$classe, p = 0.7, list = FALSE)
training_set <- the_data[train_sample, ]

## Create the validation data set.
validation_set <- the_data[-train_sample,]

## Size up and preview.
# print(head(training_set)) # Not going to preview here.
print(dim(training_set))
```

```
## [1] 13737    60
```

```r
print(dim(validation_set))
```

```
## [1] 5885   60
```


### Let's Learn!
Now, we train the model via random forest method. In doing so, we'll limit its number of folds, because, well, I ain't got all day!


```r
# ## Train using caret package (model = random forest or "rf"). Classe as outcome, predicted 
# ## by any/all other variables.
# ## DISCLAIMER: I found out that you can speed things up by limiting the number of folds the
# ##             method uses, so yeah, I'm doing that. I need my computer for other things!
# the_forest1 <- train(classe ~ ., method = "rf", data = training_set, 
#                      trControl = trainControl(method = "cv", number = 2))
```

##### What Does this Model Look Like?

```r
# ## Let's see how this model fits...
# print(the_forest1$finalModel)
```

### Let's Test What We've Learned!

