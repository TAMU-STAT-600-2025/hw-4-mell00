# Standardize X and Y: center both X and Y; scale centered X
# X - n x p matrix of covariates
# Y - n x 1 response vector
standardizeXY <- function(X, Y){
  
  # Basic checks
  if (is.null(dim(X))) stop("X must be a 2D matrix")
  if (!is.numeric(X)) stop("X must be numeric")
  if (!is.numeric(Y)) stop("Y must be numeric")
  n <- nrow(X); p <- ncol(X)
  if (length(Y) != n) stop("length of Y must match number of rows in X")
  if (n < 1 || p < 1) stop("X must have positive dimensions")
  if (anyNA(X) || anyNA(Y)) stop("missing inputs not supported")
  
  # Center Y
  Ymean <- mean(Y)
  Ytilde <- as.numeric(Y - Ymean)
  
  # Center and scale X
  Xmeans <- colMeans(X)
  Xc <- sweep(X, 2, Xmeans, FUN = "-")
  
  # weights defined as sqrt(X_j^{\top}X_j/n) after centering of X but before scaling
  n_inv <- 1 / n
  weights <- sqrt(colSums(Xc * Xc) * n_inv)
  
  # check for columns with zero variance
  zero_var <- which(weights == 0)
  if (length(zero_var) > 0L) {
    
    # for constant columns, keep 0 after centering; set weight to 1 to avoid division by 0
    weights[zero_var] <- 1
    Xc[, zero_var] <- 0
  }
  
  # scale to ensure each column has n^{-1} X_j^T X_j = 1 after scaling
  Xtilde <- sweep(Xc, 2, weights, FUN = "/")
  
  # Return:
  # Xtilde - centered and appropriately scaled X
  # Ytilde - centered Y
  # Ymean - the mean of original Y
  # Xmeans - means of columns of X (vector)
  # weights - defined as sqrt(X_j^{\top}X_j/n) after centering of X but before scaling
  return(list(Xtilde = Xtilde, Ytilde = Ytilde, Ymean = Ymean, Xmeans = Xmeans, weights = weights))
}

# Soft-thresholding of a scalar a at level lambda 
# [OK to have vector version as long as works correctly on scalar; will only test on scalars]
soft <- function(a, lambda){
  if (!is.numeric(a) || !is.numeric(lambda) || length(lambda) != 1L)
    stop("inputs must be numeric & lambda must be length 1")
  if (lambda < 0) stop("lambda must be non-negative")
  sign(a) * pmax(abs(a) - lambda, 0)
}

# Calculate objective function of lasso given current values of Xtilde, Ytilde, beta and lambda
# Xtilde - centered and scaled X, n x p
# Ytilde - centered Y, n x 1
# lamdba - tuning parameter
# beta - value of beta at which to evaluate the function
lasso <- function(Xtilde, Ytilde, beta, lambda){
  if (is.null(dim(Xtilde))) stop("Xtilde must be a 2D matrix")
  if (!is.numeric(Xtilde) || !is.numeric(Ytilde) || !is.numeric(beta))
    stop("all inputs must be numeric")
  n <- nrow(Xtilde); p <- ncol(Xtilde)
  if (length(Ytilde) != n) stop("length of Ytilde must = number of rows in Xtilde")
  if (length(beta) != p) stop("length of beta must = number of columns in Xtilde")
  if (!is.numeric(lambda) || length(lambda) != 1L || lambda < 0)
    stop("lambda must be a non-negative numeric scalar")
  
  resid <- as.numeric(Ytilde - Xtilde %*% beta)
  (sum(resid * resid) / (2 * n)) + lambda * sum(abs(beta))
}

# Fit LASSO on standardized data for a given lambda
# Xtilde - centered and scaled X, n x p
# Ytilde - centered Y, n x 1 (vector)
# lamdba - tuning parameter
# beta_start - p vector, an optional starting point for coordinate-descent algorithm
# eps - precision level for convergence assessment, default 0.001
fitLASSOstandardized <- function(Xtilde, Ytilde, lambda, beta_start = NULL, eps = 0.001){
  #Check that n is the same between Xtilde and Ytilde
  if (is.null(dim(Xtilde))) stop("Xtilde must be a 2D matrix")
  if (!is.numeric(Xtilde) || !is.numeric(Ytilde))
    stop("Xtilde and Ytilde must be numeric")
  n <- nrow(Xtilde); p <- ncol(Xtilde)
  if (length(Ytilde) != n) stop("length of Ytilde must match number of rows in Xtilde")
  if (anyNA(Xtilde) || anyNA(Ytilde)) stop("missing values are not supported")
  
  # Check that lambda is non-negative
  
  if (!is.numeric(lambda) || length(lambda) != 1L || lambda < 0)
    stop("lambda must be a non-negative numeric scalar.")
  
  # Check for starting point beta_start. 
  # If none supplied, initialize with a vector of zeros.
  # If supplied, check for compatibility with Xtilde in terms of p
  
  if (is.null(beta_start)) {
    beta <- numeric(p)
  } else {
    if (!is.numeric(beta_start) || length(beta_start) != p)
      stop("beta_start must be a numeric vector of length ncol(Xtilde)")
    beta <- as.numeric(beta_start)
  }
  
  # Pre-compute column norms z_j = (1/n) * sum x_{ij}^2
  z <- colSums(Xtilde * Xtilde) / n
  
  # Initialize residual and objective
  r <- as.numeric(Ytilde - Xtilde %*% beta)
  f_prev <- (sum(r * r) / (2 * n)) + lambda * sum(abs(beta))
  
  # Coordinate-descent implementation. 
  # Stop when the difference between objective functions is less than eps for the first time.
  # For example, if you have 3 iterations with objectives 3, 1, 0.99999,
  # your should return fmin = 0.99999, and not have another iteration
  
  repeat {
    for (j in seq_len(p)) {
      if (z[j] == 0) { next }  # skip if column is all 0s
      bj_old <- beta[j]
      
      # re-add old contribution
      r <- r + Xtilde[, j] * bj_old
      
      # partial residual correlation
      rho <- sum(Xtilde[, j] * r) / n
      
      # soft-threshold update
      bj_new <- soft(rho, lambda) / z[j]
      beta[j] <- bj_new
      
      # remove new contribution
      r <- r - Xtilde[, j] * bj_new
    }
    
    # compute current objective
    f_curr <- (sum(r * r) / (2 * n)) + lambda * sum(abs(beta))
    
    # convergence check: stop at the first time the difference < eps
    if (abs(f_prev - f_curr) < eps) {
      fmin <- f_curr
      break
    }
    f_prev <- f_curr
  }
  
  # Return 
  # beta - the solution (a vector)
  # fmin - optimal function value (value of objective at beta, scalar)
  return(list(beta = beta, fmin = fmin))
}

# [ToDo] Fit LASSO on standardized data for a sequence of lambda values. Sequential version of a previous function.
# Xtilde - centered and scaled X, n x p
# Ytilde - centered Y, n x 1
# lamdba_seq - sequence of tuning parameters, optional
# n_lambda - length of desired tuning parameter sequence,
#             is only used when the tuning sequence is not supplied by the user
# eps - precision level for convergence assessment, default 0.001
fitLASSOstandardized_seq <- function(Xtilde, Ytilde, lambda_seq = NULL, n_lambda = 60, eps = 0.001){
  # Check that n is the same between Xtilde and Ytilde
  
  if (is.null(dim(Xtilde))) stop("Xtilde must be a 2D matrix")
  if (!is.numeric(Xtilde) || !is.numeric(Ytilde))
    stop("Xtilde and Ytilde must be numeric")
  n <- nrow(Xtilde); p <- ncol(Xtilde)
  if (length(Ytilde) != n) stop("length of Ytilde must match number of rows in Xtilde")
  if (anyNA(Xtilde) || anyNA(Ytilde)) stop("missing values not supported")
  if (!is.numeric(n_lambda) || length(n_lambda) != 1L || n_lambda < 1)
    stop("n_lambda must be a positive integer")

 
  # Check for the user-supplied lambda-seq (see below)
  # If lambda_seq is supplied, only keep values that are >= 0,
  # and make sure the values are sorted from largest to smallest.
  # If none of the supplied values satisfy the requirement,
  # print the warning message and proceed as if the values were not supplied.
  
  used_supplied <- FALSE
  if (!is.null(lambda_seq)) {
    if (!is.numeric(lambda_seq)) stop("supplied lambda_seq must be numeric")
    lambda_seq <- sort(lambda_seq[lambda_seq >= 0], decreasing = TRUE)
    if (length(lambda_seq) == 0L) {
      warning("No non-negative values in supplied lambda_seq; computing lambda_seq")
    } else {
      used_supplied <- TRUE
    }
  }
  
  # If lambda_seq is not supplied, calculate lambda_max 
  # (the minimal value of lambda that gives zero solution),
  # and create a sequence of length n_lambda as
  # lambda_seq = exp(seq(log(lambda_max), log(0.01), length = n_lambda))
  
  if (!used_supplied) {
    
    # lambda_max = max_j |(1/n) Xtilde_j^T Ytilde|
    n_inv <- 1 / n
    
    # compute cross-products
    lam_candidates <- abs(colSums(Xtilde * Ytilde) * n_inv)
    lambda_max <- max(lam_candidates)
    if (!is.finite(lambda_max)) lambda_max <- 0
    if (lambda_max <= 0) {
      
      # if perfectly orthogonal or Ytilde==0, use flat sequence of zeros
      lambda_seq <- rep(0, n_lambda)
    } else {
      lambda_seq <- exp(seq(log(lambda_max), log(0.01), length.out = n_lambda))
    }
  }
  
  # Ensure descending sort
  lambda_seq <- sort(as.numeric(lambda_seq), decreasing = TRUE)
  
  # [ToDo] Apply fitLASSOstandardized going from largest to smallest lambda 
  # (make sure supplied eps is carried over). 
  # Use warm starts strategy discussed in class for setting the starting values.
  
  # Return output
  # lambda_seq - the actual sequence of tuning parameters used
  # beta_mat - p x length(lambda_seq) matrix of corresponding solutions at each lambda value
  # fmin_vec - length(lambda_seq) vector of corresponding objective function values at solution
  return(list(lambda_seq = lambda_seq, beta_mat = beta_mat, fmin_vec = fmin_vec))
}

# [ToDo] Fit LASSO on original data using a sequence of lambda values
# X - n x p matrix of covariates
# Y - n x 1 response vector
# lambda_seq - sequence of tuning parameters, optional
# n_lambda - length of desired tuning parameter sequence, is only used when the tuning sequence is not supplied by the user
# eps - precision level for convergence assessment, default 0.001
fitLASSO <- function(X ,Y, lambda_seq = NULL, n_lambda = 60, eps = 0.001){
  # [ToDo] Center and standardize X,Y based on standardizeXY function
 
  # [ToDo] Fit Lasso on a sequence of values using fitLASSOstandardized_seq
  # (make sure the parameters carry over)
 
  # [ToDo] Perform back scaling and centering to get original intercept and coefficient vector
  # for each lambda
  
  # Return output
  # lambda_seq - the actual sequence of tuning parameters used
  # beta_mat - p x length(lambda_seq) matrix of corresponding solutions at each lambda value (original data without center or scale)
  # beta0_vec - length(lambda_seq) vector of intercepts (original data without center or scale)
  return(list(lambda_seq = lambda_seq, beta_mat = beta_mat, beta0_vec = beta0_vec))
}


# [ToDo] Fit LASSO and perform cross-validation to select the best fit
# X - n x p matrix of covariates
# Y - n x 1 response vector
# lambda_seq - sequence of tuning parameters, optional
# n_lambda - length of desired tuning parameter sequence, is only used when the tuning sequence is not supplied by the user
# k - number of folds for k-fold cross-validation, default is 5
# fold_ids - (optional) vector of length n specifying the folds assignment (from 1 to max(folds_ids)), if supplied the value of k is ignored 
# eps - precision level for convergence assessment, default 0.001
cvLASSO <- function(X ,Y, lambda_seq = NULL, n_lambda = 60, k = 5, fold_ids = NULL, eps = 0.001){
  # [ToDo] Fit Lasso on original data using fitLASSO
 
  # [ToDo] If fold_ids is NULL, split the data randomly into k folds.
  # If fold_ids is not NULL, split the data according to supplied fold_ids.
  
  # [ToDo] Calculate LASSO on each fold using fitLASSO,
  # and perform any additional calculations needed for CV(lambda) and SE_CV(lambda)
  
  # [ToDo] Find lambda_min

  # [ToDo] Find lambda_1SE
  
  
  # Return output
  # Output from fitLASSO on the whole data
  # lambda_seq - the actual sequence of tuning parameters used
  # beta_mat - p x length(lambda_seq) matrix of corresponding solutions at each lambda value (original data without center or scale)
  # beta0_vec - length(lambda_seq) vector of intercepts (original data without center or scale)
  # fold_ids - used splitting into folds from 1 to k (either as supplied or as generated in the beginning)
  # lambda_min - selected lambda based on minimal rule
  # lambda_1se - selected lambda based on 1SE rule
  # cvm - values of CV(lambda) for each lambda
  # cvse - values of SE_CV(lambda) for each lambda
  return(list(lambda_seq = lambda_seq, beta_mat = beta_mat, beta0_vec = beta0_vec, fold_ids = fold_ids, lambda_min = lambda_min, lambda_1se = lambda_1se, cvm = cvm, cvse = cvse))
}

