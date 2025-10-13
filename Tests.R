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
