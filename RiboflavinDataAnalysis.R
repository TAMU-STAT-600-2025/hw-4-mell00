# Load the riboflavin data

library(hdi)
data(riboflavin) # this puts list with name riboflavin into the R environment, y - outcome, x - gene expression
dim(riboflavin$x) # n = 71 samples by p = 4088 predictors
?riboflavin # this gives you more information on the dataset

# This is to make sure riboflavin$x can be converted and treated as matrix for faster computations
class(riboflavin$x) <- class(riboflavin$x)[-match("AsIs", class(riboflavin$x))]


# Get matrix X and response vector Y
X = as.matrix(riboflavin$x)
Y = riboflavin$y

# Source your lasso functions
source("LassoFunctions.R")

# Use your fitLASSO function on the riboflavin data with 60 tuning parameters
fit60 <- fitLASSO(X, Y, n_lambda = 60)

# Based on the above output, plot the number of non-zero elements in each beta versus the value of tuning parameter
nnz <- colSums(abs(fit60$beta_mat) > 1e-8)  # treat very small values as 0
plot(fit60$lambda_seq, nnz, type = "b", log = "x",
     xlab = expression(lambda),
     ylab = "# non-zero coefficients",
     main = "Sparsity vs Tuning Parameter (LASSO)")

# [ToDo] Use microbenchmark 10 times to check the timing of your fitLASSO function above with 60 tuning parameters

# [ToDo] Report your median timing in the comments here: (~5.8 sec for Irina on her laptop)

# [ToDo] Use cvLASSO function on the riboflavin data with 30 tuning parameters (just 30 to make it faster)

# [ToDo] Based on the above output, plot the value of CV(lambda) versus tuning parameter. Note that this will change with each run since the folds are random, this is ok.

