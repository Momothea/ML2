# Loadind the dataset
# and creating the testing and training set

dataset <- read.csv('http://www.mghassany.com/MLcourseEfrei/datasets/Social_Network_Ads.csv')

library(caTools)
set.seed(123) # CHANGE THE VALUE OF SEED. PUT YOUR STUDENT'S NUMBER INSTEAD OF 123.
split = sample.split(dataset$Purchased, SplitRatio = 0.75)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

training_set[,3:4] <- scale(training_set[,3:4]) #only first two columns
test_set[,3:4] <- scale(test_set[,3:4])




#2. Fit a LDA model of Purchased in function of Age and EstimatedSalary. Name the model classifier.lda.

library(MASS)
classifier.lda <- lda(Purchased~Age+EstimatedSalary, data=training_set)

#3. Call classifier.lda and understand what does it compute.

classifier.lda$prior
classifier.lda$means

#4. On the test set, predict the probability of purchasing the product by the users using 
#the model classifier.lda. Remark that when we predict using LDA, we obtain a list instead of a matrix, do str() 
#for your predictions to see what do you get.

prediction <- predict(classifier.lda, test_set)

(prediction$posterior)

#5. Compute the confusion matrix and compare the predictions results obtained by LDA 
# to the ones obtained by logistic regression. What do you remark? (Hint: compare the accuracy)

confusion_matrix <- table(predicted = prediction$class, actual = test_set$Purchased)

confusion_matrix

prediction$posterior


#6. Now let us plot the decision boundary obtained with LDA.

# create a grid corresponding to the scales of Age and EstimatedSalary
# and fill this grid with lot of points
X1 = seq(min(training_set[, 3]) - 1, max(training_set[, 3]) + 1, by = 0.01)
X2 = seq(min(training_set[, 4]) - 1, max(training_set[, 4]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
# Adapt the variable names
colnames(grid_set) = c('Age', 'EstimatedSalary')

# plot 'Estimated Salary' ~ 'Age'
plot(test_set[, 3:4],
     main = 'Decision Boundary LDA',
     xlab = 'Age', ylab = 'Estimated Salary',
     xlim = range(X1), ylim = range(X2))

# color the plotted points with their real label (class)
points(test_set[,3:4], pch = 21, bg = ifelse(test_set[, 5] == 1, 'green4', 'red3'))

# Make predictions on the points of the grid, this will take some time
pred_grid = predict(classifier.lda, newdata = grid_set)$class

# Separate the predictions by a contour
contour(X1, X2, matrix(as.numeric(pred_grid), length(X1), length(X2)), add = TRUE)
#test_set[, 3:4]
#test_set[1:2]
#test_set[, 3]


#Quadratic Discriminant Analysis (QDA)

#7. Fit a QDA model of Purchased in function of Age and EstimatedSalary. Name the model classifier.qda.

# qda() is a function of library(MASS)
classifier.qda <- qda(Purchased~Age+EstimatedSalary, data = training_set)



#8. Make predictions on the test_set using the QDA model classifier.qda. 
# Show the confusion matrix and compare the results with the predictions obtained using the LDA model classifier.lda.

prediction <- predict(classifier.qda, test_set)
prediction2 <- predict(classifier.qda, test_set)


confusion_matrix <- table(predicted = prediction$class, actual = test_set$Purchased)

confusion_matrix


#9. Plot the decision boundary obtained with QDA. Color the points with the real labels.


# create a grid corresponding to the scales of Age and EstimatedSalary
# and fill this grid with lot of points
X12 = seq(min(training_set[, 3]) - 1, max(training_set[, 3]) + 1, by = 0.01)
X22 = seq(min(training_set[, 4]) - 1, max(training_set[, 4]) + 1, by = 0.01)
grid_set2 = expand.grid(X1, X2)
# Adapt the variable names
colnames(grid_set2) = c('Age', 'EstimatedSalary')

# plot 'Estimated Salary' ~ 'Age'
plot(test_set[, 3:4],
     main = 'Decision Boundary LDA',
     xlab = 'Age', ylab = 'Estimated Salary',
     xlim = range(X12), ylim = range(X22))

# color the plotted points with their real label (class)
points(test_set[3:4], pch = 21, bg = ifelse(test_set[, 5] == 1, 'green4', 'red3'))

# Make predictions on the points of the grid, this will take some time
pred_grid = predict(classifier.qda, newdata = grid_set2)$class

# Separate the predictions by a contour
contour(X12, X22, matrix(as.numeric(pred_grid), length(X12), length(X22)), add = TRUE)


#10. 
require(ROCR)
library(pROC)
library(randomForest)

roc(test_set$Purchased, prediction$posterior[,2], plot=TRUE)

roc(test_set$Purchased, prediction2$posterior[,2], plot=TRUE)

