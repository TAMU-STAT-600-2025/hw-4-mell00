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
