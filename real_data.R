# Real Data Analysis
## Auto data (regression problem)
library(nnet)
library(DiceKriging)
library(randomForest)
library(ALEPlot)
library(ISLR)
data(Auto)
method_names <- c("1" = "PD", "2" = "ALE", "3" = "A2D2E")
K <- 11  # since grid_len = 40 => K = grid_len + 1

# functions adapted to real data

.make_1row_df <- function(center_vec) {
  stopifnot(is.numeric(center_vec))
  cn <- names(center_vec)
  df <- as.data.frame(as.list(center_vec), stringsAsFactors = FALSE)
  if (!is.null(cn)) names(df) <- cn
  df
}

cube_beta_target_from_model <- function(center, edge, model_function, model, target) {
  # normalize center to named numeric vector
  if (is.data.frame(center)) {
    center <- as.numeric(center[1, , drop = TRUE]); names(center) <- colnames(.subset2(center, 0)) # keep names if any
  } else {
    center <- as.numeric(center)
  }
  stopifnot(!is.null(names(center)))  # need names
  
  k <- length(center)
  n <- 2^k
  levels <- lapply(center, function(cj) c(cj - edge, cj + edge))
  X <- expand.grid(levels, KEEP.OUT.ATTRS = FALSE, stringsAsFactors = FALSE)
  colnames(X) <- names(center)
  
  y <- apply(X, 1, function(row) {
    newx <- as.numeric(row); names(newx) <- names(center)
    model_function(model, .make_1row_df(newx))
  })
  
  tgt_name <- if (is.numeric(target)) names(center)[target] else target
  z_target <- X[[tgt_name]] - center[[tgt_name]]
  beta_target <- sum(z_target * y) / (n * edge^2)
  as.numeric(beta_target)
}

optimal_edge_and_beta_nd <- function(model_function, model, target, center,
                                     min_window = 1e-12, sp = 0.001) {
  cube_beta_target_from_model(center, sp, model_function, model, target)
}

A2D2EPlot <- function(D, model, model_function, target_variable, K, sp) {
  stopifnot(is.data.frame(D))
  tgt_name <- if (is.numeric(target_variable)) colnames(D)[target_variable] else target_variable
  x_j <- D[[tgt_name]]
  
  x_j <- x_j[is.finite(x_j)]
  if (length(x_j) < 2L) stop("Not enough data for A2D2E on variable: ", tgt_name)
  
  breaks <- quantile(x_j, probs = seq(0, 1, length.out = K + 1), type = 7, na.rm = TRUE)
  # fallback if too many duplicates
  if (length(unique(breaks)) < 2L) {
    breaks <- seq(min(x_j), max(x_j), length.out = K + 1)
  }
  breaks[1]     <- min(x_j, na.rm = TRUE) - 1e-8
  breaks[K + 1] <- max(x_j, na.rm = TRUE)
  
  num_seg <- integer(0)
  beta <- numeric(0)
  
  for (k in seq_len(K)) {
    seg_idx <- which(D[[tgt_name]] >= breaks[k] & D[[tgt_name]] < breaks[k + 1])
    n_k <- length(seg_idx)
    num_seg <- c(num_seg, n_k)
    
    if (n_k > 0) {
      betas_k <- vapply(seg_idx, function(irow) {
        center <- D[irow, , drop = FALSE]
        optimal_edge_and_beta_nd(model_function, model,
                                 target = tgt_name,
                                 center = center,
                                 min_window = 1e-12, sp = sp)
      }, numeric(1))
      beta_k <- mean(betas_k, na.rm = TRUE)
      if (!is.finite(beta_k)) {
        center_mid <- D[1, , drop = FALSE]
        center_mid[[tgt_name]] <- (breaks[k] + breaks[k + 1]) / 2
        beta_k <- optimal_edge_and_beta_nd(model_function, model,
                                           target = tgt_name,
                                           center = center_mid,
                                           min_window = 1e-12, sp = sp)
      }
    } else {
      center_mid <- D[1, , drop = FALSE]
      center_mid[[tgt_name]] <- (breaks[k] + breaks[k + 1]) / 2
      beta_k <- optimal_edge_and_beta_nd(model_function, model,
                                         target = tgt_name,
                                         center = center_mid,
                                         min_window = 1e-12, sp = sp)
    }
    beta <- c(beta, as.numeric(beta_k))
  }
  
  delta <- diff(breaks)
  loc <- numeric(K + 1)
  loc[1] <- 0
  for (i in 2:(K + 1)) {
    loc[i] <- loc[i - 1] + delta[i - 1] * beta[i - 1]
  }
  
  # center by segment-count weighted mean
  norm_const <- sum(num_seg * loc[-1]) / sum(num_seg)
  list(
    x.values = as.numeric(breaks),
    f.values = as.numeric(loc - norm_const)
  )
}

# example usage
method_names <- c("1" = "PD", "2" = "ALE", "3" = "A2D2E")
y_col <- "mpg"
X <- Auto[, setdiff(names(Auto), c("mpg", "name", "origin", "cylinders"))]
y <- Auto[[y_col]]
df <- data.frame(X, mpg = y)

set.seed(123)
model_rf <- randomForest(mpg ~ ., data = df, ntree = 1000)

yhat_fun <- function(X.model, newdata) {
  as.numeric(predict(X.model, newdata = newdata))
}

K <- 11
sp <- 0.001
pred_names <- colnames(X)
method_names <- c("1" = "PD", "2" = "ALE", "3" = "A2D2E")

colors <- c("1"="#1b1b1b", "2"="#4e79a7", "3"="#e15759")
pch_vals <- c("1"=16, "2"=17, "3"=15)

for (v in seq_along(pred_names)) {
  var_name <- pred_names[v]
  rng <- range(X[[var_name]], na.rm = TRUE)
  
  # compute PD, ALE, A2D2E
  pd_res   <- PDPlot(X, model_rf, yhat_fun, J = v, K = K)
  ale_res  <- ALEPlot(X, model_rf, yhat_fun, J = v, K = K)
  a2d2e_res <- A2D2EPlot(X, model_rf, yhat_fun, target_variable = var_name, K = K, sp = sp)
  method_list <- list("1" = pd_res, "2" = ale_res, "3" = a2d2e_res)
  
  # plot
  plot(1, type = "n", xlim = rng, ylim = c(-5, 5),
       xlab = var_name, ylab = "Estimated Main Effect",
       main = paste("Main Effect on mpg\nFeature:", var_name),
       cex.lab = 1.3, cex.axis = 1.1)
  
  for (m in names(method_list)) {
    res <- method_list[[m]]
    lines(res$x.values, res$f.values, col = colors[m], lwd = 2)
    points(res$x.values, res$f.values, col = colors[m], pch = pch_vals[m])
  }
  
  abline(h = 0, lty = 2, col = "gray40")
  legend("topright", legend = method_names, col = colors, pch = pch_vals,
         lwd = 2, bty = "n", cex = 1)
}
##########################################################################
# iris data experiment
# Predict Log-Odds Function
predict_ref <- function(object, newdata, ref = "setosa", target = "versicolor") {
  probs <- nnet:::predict.multinom(object, newdata = newdata, type = "probs")
  
  if (is.null(dim(probs))) probs <- t(as.matrix(probs))
  if (is.null(colnames(probs))) colnames(probs) <- object$lev
  
  if (!(ref %in% colnames(probs)) || !(target %in% colnames(probs))) {
    stop(sprintf("Class names not found. Available: %s",
                 paste(colnames(probs), collapse = ", ")))
  }
  
  log_odds <- log(probs[, target, drop = TRUE] / probs[, ref, drop = TRUE])
  as.numeric(log_odds)
}

# Fine-tunning Neural Network using cross-Validation
library(caret)

data(iris)
iris$Species <- relevel(iris$Species, ref = "setosa")

# Prepare data
X <- iris[, 1:4]
Y <- iris$Species
pred_names <- colnames(X)

set.seed(123)

# Define tuning grid
grid <- expand.grid(
  size  = c(4, 8, 12, 16),      # number of hidden units
  decay = c(0.0001, 0.001, 0.01) # L2 regularization strength
)

# 10-fold CV
ctrl <- trainControl(method = "cv", number = 10)

# Train using caret to select optimal hyperparameters
nn_tuned <- caret::train(
  Species ~ .,
  data = iris,
  method = "nnet",
  metric = "Accuracy",
  trace = FALSE,
  maxit = 2000,
  tuneGrid = grid,
  trControl = ctrl
)


model <- nn_tuned$finalModel

# Reference and target classes for log-odds
ref_class <- "setosa"
target_class <- "versicolor"

# Prediction wrapper computing log-odds
pred_fun <- function(X.model, newdata) {
  probs <- predict(nn_tuned, newdata = newdata, type = "prob")
  log_odds <- log(probs[, target_class] / probs[, ref_class])
  as.numeric(log_odds)
}

K <- 11
sp <- 0.001
method_names <- c("1" = "PD", "2" = "ALE", "3" = "A2D2E")
colors <- c("1" = "#1b1b1b", "2" = "#4e79a7", "3" = "#e15759")
pch_vals <- c("1" = 16, "2" = 17, "3" = 15)

for (v in seq_along(pred_names)) {
  var_name <- pred_names[v]
  rng <- range(X[[var_name]])
  
  # Compute PD, ALE, A2D2E
  pd_res   <- PDPlot(X, model, pred_fun, J = v, K = K)
  ale_res  <- ALEPlot(X, model, pred.fun = pred_fun, J = v, K = K)
  a2d2e_res <- A2D2EPlot(X, model, pred_fun, target_variable = var_name, K = K, sp = sp)
  method_list <- list("1" = pd_res, "2" = ale_res, "3" = a2d2e_res)
  
  # Create a new plot window for each variable
  plot(1, type = "n",
       xlim = rng,
       ylim = c(-5, 5),
       xlab = var_name,
       ylab = "Estimated Main Effect (log-odds)",
       main = paste("Effect on log-odds of", target_class, "\nFeature:", var_name),
       cex.lab = 1.3, cex.axis = 1.1)
  
  for (m in names(method_list)) {
    res <- method_list[[m]]
    lines(res$x.values, res$f.values, col = colors[m], lwd = 2)
    points(res$x.values, res$f.values, col = colors[m], pch = pch_vals[m])
  }
  
  abline(h = 0, lty = 2, col = "gray40")
  legend("topleft",
         legend = method_names,
         col = colors,
         pch = pch_vals,
         lwd = 2,
         bty = "n",
         cex = 1)
}
