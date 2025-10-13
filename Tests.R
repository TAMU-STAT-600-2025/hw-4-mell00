# Source your lasso functions
source("LassoFunctions.R")

set.seed(123)
cat("\n=== Begin Tests (LASSO) ===\n")
n_ok <- 0L

## 1) standardizeXY

{
  test_name <- "standardizeXY shapes, centering, scaling, constant column handling"
  n <- 50; p <- 8
  X_raw <- matrix(rnorm(n * p), n, p)
  X_raw <- cbind(X_raw, const = 7)  # add constant col
  Y_raw <- rnorm(n) + 0.5
  
  std <- standardizeXY(X_raw, Y_raw)
  if (!all.equal(dim(std$Xtilde), c(n, p + 1))) stop(test_name, " (Xtilde dims)")
  if (!isTRUE(all.equal(mean(Y_raw), std$Ymean, tolerance = 1e-12))) stop(test_name, " (Ymean mismatch)")
  if (!isTRUE(all.equal(mean(std$Ytilde), 0, tolerance = 1e-12))) stop(test_name, " (Ytilde not centered)")
  
  xm <- colMeans(std$Xtilde)
  if (max(abs(xm)) > 1e-10) stop(test_name, " (Xtilde columns not centered)")
  
  zj <- colSums(std$Xtilde^2) / n
  const_j <- ncol(std$Xtilde)
  if (!isTRUE(all.equal(std$weights[const_j], 1))) stop(test_name, " (weight for constant col != 1)")
  if (sum(std$Xtilde[, const_j]^2) != 0) stop(test_name, " (constant col not zeroed)")
  if (max(abs(zj[-const_j] - 1)) > 1e-10) stop(test_name, " ((1/n) X^T X diag not ~ 1)")
  cat(test_name, "PASSED\n"); n_ok <- n_ok + 1L
}

## 2) soft-thresholding

{
  test_name <- "soft() basic correctness"
  if (!isTRUE(all.equal(soft(3, 1),  2, tolerance = 1e-12)))  stop(test_name, " (soft(3,1))")
  if (!isTRUE(all.equal(soft(-3,1), -2, tolerance = 1e-12)))  stop(test_name, " (soft(-3,1))")
  if (!isTRUE(all.equal(soft(0.5,1), 0, tolerance = 1e-12)))  stop(test_name, " (soft(0.5,1))")
  if (!isTRUE(all.equal(soft(2, 0),  2, tolerance = 1e-12)))  stop(test_name, " (soft(2,0))")
  cat(test_name, "PASSED\n"); n_ok <- n_ok + 1L
}

## 3) lasso objective

{
  test_name <- "lasso() objective increases with lambda for a fixed beta"
  n <- 30; p <- 5
  X <- matrix(rnorm(n*p), n, p)
  Y <- rnorm(n)
  std <- standardizeXY(X, Y)
  beta0 <- rep(0, p)
  f1 <- lasso(std$Xtilde, std$Ytilde, beta0, lambda = 0.1)
  f2 <- lasso(std$Xtilde, std$Ytilde, beta0, lambda = 0.3)
  if (!(f2 > f1 - 1e-15)) stop(test_name)
  cat(test_name, "PASSED\n"); n_ok <- n_ok + 1L
}

## 4) fitLASSOstandardized vs closed-form (orthonormal X)

{
  test_name <- "fitLASSOstandardized matches closed-form on orthonormal design"
  n <- 120; p <- 10
  A <- matrix(rnorm(n*p), n, p)
  Q <- qr.Q(qr(A))
  Xo <- sqrt(n) * Q # (1/n) X^T X = I
  beta_true <- c(rep(2, 3), rep(0, p - 3))
  Yo <- as.numeric(Xo %*% beta_true + rnorm(n, sd = 0.05))
  lambda <- 0.5
  corr <- drop(crossprod(Xo, Yo)) / n
  beta_cf <- sign(corr) * pmax(abs(corr) - lambda, 0)
  
  fit_o <- fitLASSOstandardized(Xo, Yo, lambda = lambda, eps = 1e-8)
  if (max(abs(fit_o$beta - beta_cf)) > 5e-3) stop(test_name, " (beta mismatch vs closed-form)")
  
  lambda_max <- max(abs(drop(crossprod(Xo, Yo)) / n))
  fit_zero <- fitLASSOstandardized(Xo, Yo, lambda = lambda_max, eps = 1e-8)
  if (max(abs(fit_zero$beta)) > 1e-6) stop(test_name, " (not zero at lambda_max)")
  cat(test_name, "PASSED\n"); n_ok <- n_ok + 1L
}

## 5) fitLASSOstandardized_seq

{
  test_name <- "fitLASSOstandardized_seq shapes, order, zero start, sparsity trend"
  n <- 100; p <- 20
  X <- matrix(rnorm(n*p), n, p)
  Y <- rnorm(n)
  std <- standardizeXY(X, Y)
  
  sf <- fitLASSOstandardized_seq(std$Xtilde, std$Ytilde, n_lambda = 50, eps = 1e-6)
  if (length(sf$lambda_seq) != ncol(sf$beta_mat)) stop(test_name, " (lambda_seq vs beta_mat mismatch)")
  if (!all(diff(sf$lambda_seq) <= 0)) stop(test_name, " (lambda_seq not decreasing)")
  
  # near-zero at first lambda
  if (max(abs(sf$beta_mat[,1])) > 1e-4) stop(test_name, " (first beta not near zero)")
  
  # sparsity should generally increase as lambda decreases
  nnz <- colSums(abs(sf$beta_mat) > 1e-8)
  if (!(tail(nnz,1) >= nnz[1])) stop(test_name, " (sparsity did not increase overall)")
  cat(test_name, "PASSED\n"); n_ok <- n_ok + 1L
}

## 6) fitLASSO

{
  test_name <- "fitLASSO shapes and intercept ~ mean(Y) at largest lambda"
  n <- 80; p <- 15
  X <- matrix(rnorm(n*p), n, p)
  Y <- rnorm(n)
  fit <- fitLASSO(X, Y, n_lambda = 40, eps = 1e-4)
  
  if (length(fit$lambda_seq) != ncol(fit$beta_mat)) stop(test_name, " (lambda_seq vs beta_mat mismatch)")
  if (length(fit$beta0_vec)  != length(fit$lambda_seq)) stop(test_name, " (beta0 length mismatch)")
  if (abs(fit$beta0_vec[1] - mean(Y)) > 1e-6) stop(test_name, " (beta0[1] != mean(Y))")
  cat(test_name, "PASSED\n"); n_ok <- n_ok + 1L
}

## 7) cvLASSO outputs & selection rules

{
  test_name <- "cvLASSO shapes & selection (lambda_min, lambda_1se)"
  n <- 90; p <- 12
  X <- matrix(rnorm(n*p), n, p)
  Y <- rnorm(n)
  set.seed(42)
  cv <- cvLASSO(X, Y, n_lambda = 25, k = 5, eps = 1e-3)
  
  if (length(cv$lambda_seq) != length(cv$cvm))  stop(test_name, " (cvm length mismatch)")
  if (length(cv$cvm) != length(cv$cvse)) stop(test_name, " (cvse length mismatch)")
  if (length(cv$beta0_vec) != ncol(cv$beta_mat)) stop(test_name, " (beta0 vs beta_mat mismatch)")
  if (!(cv$lambda_min %in% cv$lambda_seq)) stop(test_name, " (lambda_min not in path)")
  if (!(cv$lambda_1se %in% cv$lambda_seq)) stop(test_name, " (lambda_1se not in path)")
  
  imin <- which(cv$lambda_seq == cv$lambda_min)[1]
  i1se <- which(cv$lambda_seq == cv$lambda_1se)[1]
  if (!(i1se <= imin)) stop(test_name, " (lambda_1se not earlier/larger than lambda_min)")
  cat(test_name, "PASSED\n"); n_ok <- n_ok + 1L
}



## 8) Adversarial inputs

# huge magnitude X and Y stay finite; all-zero Y results in zero beta, zero intercepts
{
  test_name <- "huge magnitudes, all-zero Y"
  n <- 40; p <- 20
  X_big <- matrix(rnorm(n*p), n, p) * 1e8
  Y_big <- rnorm(n) * 1e8
  std_big <- standardizeXY(X_big, Y_big)
  if (!all(is.finite(std_big$Xtilde))) stop(test_name, " (Xtilde non-finite)")
  if (!all(is.finite(std_big$Ytilde))) stop(test_name, " (Ytilde non-finite)")
  fit_big <- fitLASSO(X_big, Y_big, n_lambda = 30, eps = 1e-3)
  if (!all(is.finite(fit_big$beta_mat))) stop(test_name, " (beta_mat non-finite)")
  if (!all(is.finite(fit_big$beta0_vec))) stop(test_name, " (beta0_vec non-finite)")
  
  Y_zero <- rep(0, n)
  fit_zero <- fitLASSO(X_big, Y_zero, n_lambda = 20, eps = 1e-3)
  if (max(abs(fit_zero$beta_mat)) > 1e-8) stop(test_name, " (zero-Y: betas not zero)")
  if (max(abs(fit_zero$beta0_vec)) > 1e-8) stop(test_name, " (zero-Y: intercepts not zero)")
  cat(test_name, "PASSED\n"); n_ok <- n_ok + 1L
}

# all-constant X -> Xtilde becomes zero, betas = zero, and intercept = mean(Y)
{
  test_name <- "all-constant X"
  n <- 30; p <- 10
  Xc <- matrix(5, n, p)
  Yc <- rnorm(n)
  stdc <- standardizeXY(Xc, Yc)
  if (sum(stdc$Xtilde^2) != 0) stop(test_name, " (Xtilde not all zero)")
  
  sf_c <- fitLASSOstandardized_seq(stdc$Xtilde, stdc$Ytilde, n_lambda = 10)
  if (length(sf_c$lambda_seq) != ncol(sf_c$beta_mat)) stop(test_name, " (seq shapes mismatch)")
  
  fit_c <- fitLASSO(Xc, Yc, n_lambda = 10)
  if (max(abs(fit_c$beta_mat)) > 1e-10) stop(test_name, " (betas not zero)")
  if (abs(fit_c$beta0_vec[1] - mean(Yc)) > 1e-10) stop(test_name, " (intercept != mean(Y))")
  cat(test_name, "PASSED\n"); n_ok <- n_ok + 1L
}

# junk lambda_seq (including NA/Inf/neg/dupes/unsorted) cleaned
{
  test_name <- "clean up junk lambda_seq"
  n <- 35; p <- 12
  X <- matrix(rnorm(n*p), n, p)
  Y <- rnorm(n)
  std <- standardizeXY(X, Y)
  lam_junk <- c(Inf, NA, -1, 0.5, 0.5, 10, 1e-3)
  
  sf <- fitLASSOstandardized_seq(std$Xtilde, std$Ytilde, lambda_seq = lam_junk, eps = 1e-4)
  if (length(sf$lambda_seq) != ncol(sf$beta_mat)) stop(test_name, " (shapes mismatch)")
  if (!all(sf$lambda_seq >= 0)) stop(test_name, " (negative lambda remained)")
  if (!all(diff(sf$lambda_seq) <= 0)) stop(test_name, " (lambda not decreasing)")
  cat(test_name, "PASSED\n"); n_ok <-  n_ok + 1L
}

# very large lambda yields all-zero solution
{
  test_name <- "very large lambda yields all-zero solution"
  n <- 50; p <- 10
  X <- matrix(rnorm(n*p), n, p)
  Y <- rnorm(n)
  std <- standardizeXY(X, Y)
  lam_max <- max(abs(drop(crossprod(std$Xtilde, std$Ytilde)))/n)
  fit_vl <- fitLASSOstandardized(std$Xtilde, std$Ytilde, lambda = lam_max * 100, eps = 1e-6)
  if (max(abs(fit_vl$beta)) > 1e-10) stop(test_name, " (beta not zero)")
  cat(test_name, "PASSED\n"); n_ok <- n_ok + 1L
}

# fitting remains numerically stable and outputs are finite
{
  test_name <- "p >> n with collinearity remains stable and finite"
  n <- 30; p <- 300
  X <- matrix(rnorm(n*p), n, p)
  X[,2] <- X[,1] + rnorm(n, sd = 1e-6)  # strong collinearity
  Y <- rnorm(n)
  fit <- fitLASSO(X, Y, n_lambda = 25, eps = 1e-3)
  if (length(fit$lambda_seq) != ncol(fit$beta_mat)) stop(test_name, " (shapes mismatch)")
  if (!all(is.finite(fit$beta_mat))) stop(test_name, " (non-finite betas)")
  cat(test_name, "PASSED\n"); n_ok <- n_ok + 1L
}

# test cvLASSO with custom, unbalanced folds
{
  test_name <- "cvLASSO with custom folds"
  n <- 45; p <- 25
  X <- matrix(rnorm(n*p), n, p)
  Y <- rnorm(n)
  fold_ids <- rep(1:5, length.out = n); fold_ids[1:3] <- 1
  cv <- cvLASSO(X, Y, n_lambda = 15, fold_ids = fold_ids, eps = 1e-3)
  if (length(cv$lambda_seq) != length(cv$cvm)) stop(test_name, " (cvm mismatch)")
  cat(test_name, "PASSED\n"); n_ok <- n_ok + 1L
}