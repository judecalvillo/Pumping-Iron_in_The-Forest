## Load libraries: Caret, for machine learning, and dplyr for convenience. :)
library(caret)
library(dplyr)

## Get the data.
the_data <- read.csv("data/pml-training.csv", na.strings = c("NA", ""))

## Show size.
print(dim(the_data))


########### Pre-processing: Mostly cleaning! #############

## The data set has lots of NAs. Indeed some columns are purely NAs.
## Therefore, let's find those columns that are purely NAs, then remove them,
## as they won't be very useful to us and could be distracting to our model.

## Here, we get the sum of NAs per column (note, the margin argument of Apply
## function = 2, for columns only; default is 1, for rows/lists). Then, we replace
## the original df w/the a cleaner version of itself.
na_colsum <- apply(the_data, 2, function(x) {
    sum(is.na(x))
})
the_data <- the_data[, which(na_colsum == 0)]

## Show size.
print(dim(the_data))


########### Partitioning (for later cross-validation) ############

## Create the training set partition.
train_sample <- createDataPartition(y = the_data$classe, p = 0.7, list = FALSE)
training_set <- the_data[train_sample, ]

## Create the validation data set.
validation_set <- the_data[-train_sample,]

## Size up and preview.
print(dim(training_set))
print(dim(validation_set))
# print(head(training_set))


########### Training the model (random forest) #############

## Train using caret package (model = random forest or "rf"). 
## Classe as outcome, predicted by any/all other variables.
## DISCLAIMER: I found out that you can speed things up by limiting the number of folds the
##             method uses, so yeah, I'm doing that. I need my computer for other things!
the_forest1 <- train(classe ~ ., method = "rf", data = training_set, 
                     trControl = trainControl(method = "cv", number = 2))

## Let's see how this model fits...
print(the_forest1$finalModel)
