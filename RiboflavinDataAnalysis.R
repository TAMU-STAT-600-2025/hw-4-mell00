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

# check that dimensions are square
#length(fit60$lambda_seq); ncol(fit60$beta_mat)

# Based on the above output, plot the number of non-zero elements in each beta versus the value of tuning parameter
nnz <- colSums(abs(fit60$beta_mat) > 1e-8)  # treat very small values as 0
plot(fit60$lambda_seq, nnz, type = "b", log = "x",
     xlab = expression(lambda),
     ylab = "# non-zero coefficients",
     main = "Sparsity vs Tuning Parameter (LASSO)")

# Use microbenchmark 10 times to check the timing of your fitLASSO function above with 60 tuning parameters
library(microbenchmark)
mb <- microbenchmark(
  fitLASSO(X, Y, n_lambda = 60),
  times = 10
)
print(mb)

# Report your median timing in the comments here: (~5.8 sec for Irina on her laptop)
median_seconds <- median(mb$time) / 1e9  # convert nanoseconds into seconds
cat(sprintf("Median time over 10 runs: %.3f seconds\n", median_seconds))

# 0.088 sec

# Use cvLASSO function on the riboflavin data with 30 tuning parameters (just 30 to make it faster)
set.seed(42)  # seed for reproducibility
cvfit <- cvLASSO(X, Y, n_lambda = 30, k = 5)

# Based on the above output, plot the value of CV(lambda) versus tuning parameter. Note that this will change with each run since the folds are random, this is ok.
plot(cvfit$lambda_seq, cvfit$cvm, type = "b", log = "x",
     xlab = expression(lambda),
     ylab = "CV MSE",
     main = "Cross-Validation Curve (LASSO)")

# SE error bars
arrows(x0 = cvfit$lambda_seq, y0 = cvfit$cvm - cvfit$cvse,
       x1 = cvfit$lambda_seq, y1 = cvfit$cvm + cvfit$cvse,
       angle = 90, code = 3, length = 0.05)
