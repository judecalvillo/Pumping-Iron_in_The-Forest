## Load the necessary libraries.
library(caret) # For training/machine learning.


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
na_colsum <- apply(the_data, 2, function(x){sum(is.na(x))})
the_data <- the_data[, which(na_colsum == 0)]

## Show size.
print(dim(the_data))

## Remove other -probably- useless variables.
to_rm <- grep("timestamp|X|user_name|new_window", names(the_data))
the_data <- the_data[, -to_rm]
dim(the_data)

########### Partitioning (for later cross-validation) ############

## Create the training set partition and remaining validation set.
train_sample <- createDataPartition(y = the_data$classe, p = 0.7, list = FALSE)
training_set <- the_data[train_sample, ]
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
                     trControl = trainControl(method = "cv", number = 5))

## Let's 'see' the model.
print(the_forest1)

## Get the # of predictors for max accuracy.
print(the_forest1$bestTune)

## Plot the accuracy per predictors.
print(plot(the_forest1, col = "forestgreen", 
main = "Model Accuracy per Number of Predictors Selected"))

############ Cross-Validate ############

## We're cross-validating to see how well the model does with data that's outside
## of the original sample.

## Compute the predictions upon the validation set, using our forest model.
v_predictions <- predict(the_forest1, validation_set)

## Now, let's sum the # of correct predictions and divide by the total # of values to get 
## accuracy and, consequently, the error rate.
v_accuracy <- sum(v_predictions == validation_set$classe)/length(v_predictions)
oos_error <- 1 - v_accuracy

## Print error rate, prettily. :)
print(paste("Out of sample error rate: ", round(oos_error*100,3), "% (percent)", sep = ""))

########### Use test data! ############

## Get the data.
test_data <- read.csv("data/pml-testing.csv", na.strings = c("NA", ""))

########### Same pre-processing as before. ###########

na_colsum2 <- apply(test_data, 2, function(x){sum(is.na(x))})
test_data <- test_data[, which(na_colsum2 == 0)]
to_rm2 <- grep("timestamp|X|user_name|new_window", names(test_data))
test_data <- test_data[, -to_rm2]

## Show size.
print(dim(test_data))

## Test predictions
t_predictions <- predict(the_forest1, test_data)
print(t_predictions)

########### Export predicted values to text files for class submission ########
## Supplied by professor.
pml_write_files = function(t_predictions){
    n = length(t_predictions)
    for(i in 1:n){
        filename = paste0("data/problem_id_",i,".txt")
        write.table(t_predictions[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
    }
}