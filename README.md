## Pumping Iron in the Forest: A Machine Learning "Exercise." :)
by Jude Calvillo  

Below, I will build, cross-validate, and assess a "random forest" predictive model. This model aims to predict a physical exercise outcome, type of bicep curl (5 different types), from fitness tracker data (e.g. Fitbit) that initially includes 160 variables (yikes!). 

### Pre-Processing / Cleaning
As you can see, the source dataset is rather large. More importantly, it includes a slew of variables that might predict the outcome of concern, the last variable: classe. Therefore, we should cut this down.

#### Removing Useless Columns (variables)
It looks like, upon preview, the source data includes many columns that are made purely of NA values. Therefore, let's find them, remove them, and see how many useful variables we end up with.

### Create the Training and 'Test' (Validation) Sets
Here, we simply partition the cleaned dataset, for the training set, and leave the remainder to the test/validation set.

### Let's Learn!
Now, we train the model via random forest method. In doing so, we'll limit its number of folds, because, well, I ain't got all day!

### Let's Test What We've Learned!


