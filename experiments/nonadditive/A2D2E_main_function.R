# ALSE main function
# the input of the implemented A2D2E plot has the same stucture of ALE plot
library(ALEPlot)

A2D2EPlot <- function(D, md, predicted_model, target_variable, K, sp) {
  # Extract the target variable column
  x_j <- D[, target_variable]
  # Define breaks
  breaks <- quantile(x_j, probs = seq(0, 1, length.out = K + 1)) # use quantile to make bins
  breaks[1] <- min(x_j) - 1e-8  # slightly below min
  breaks[K + 1] <- max(x_j)     # exactly at max
  num_seg = 0 # collect the number at each bin
  beta <- NULL  # Collect estimated slopes (betas)
  
  
  for (k in seq_len(K)) {# for each bin
    # Select points in the k-th segment
    segment_indices <- which(x_j >= breaks[k] & x_j < breaks[k + 1])
    n_k <- length(segment_indices)
    num_seg = c(num_seg, n_k)
    
    beta_k <- c()
    for (i in seq_len(n_k)) {# for each location in the bin
      # Center point
      x_k_i <- D[segment_indices[i], ]
      
      beta_local <- optimal_edge_and_beta_nd(
        model_function = predicted_model,
        md = md,
        target = target_variable,
        center = x_k_i,
        min_window = 1e-12,
        sp = sp
      )
      beta_k <- c(beta_k, beta_local) 
    }
    
    # Average the local slopes
    beta <- c(beta, mean(beta_k))
  }
  
  # Compute accumulated effect
  delta <- breaks[-1] - breaks[-length(breaks)]
  loc <- c(0)
  for (i in 2:(K+1)) {
    loc <- c(loc, loc[i-1] + delta[i-1] * beta[i-1])
  }
  
  norm_const = as.numeric(num_seg%*%loc/sum(num_seg))
  return(list(
    x.values = as.numeric(breaks),
    f.values = as.numeric(loc-norm_const))) # no center
}

# estimate the beta
cube_beta_target_from_model <- function(center, edge, model_function, md, target) {
  k <- length(center) # dimension of x
  n <- 2^k # total number of vertices
  levels <- lapply(center, function(cj) c(cj - edge, cj + edge))
  X <- expand.grid(levels)
  colnames(X) <- names(center)  # ensure correct column names
  
  # Evaluate model on each row as named dataframe
  y <- apply(X, 1, function(row) model_function(md, as.data.frame(t(row))))
  
  z_target <- X[[target]] - center[[target]]
  beta_target <- sum(z_target * y) / (n * edge^2)
  return(beta_target)
}

# optimizing the window
optimal_edge_and_beta_nd <- function(model_function, md, target, center, 
                                     min_window, threshold = 0.05, sp) {
  k <- length(center)
  n <- 2^k
  spacing <- 0.025
  
  return(cube_beta_target_from_model(center, sp, model_function, md, target))
  
  while (spacing > 1e-12) {
    levels <- replicate(k, c(-spacing, spacing), simplify = FALSE)
    cube_points <- expand.grid(levels)
    colnames(cube_points) <- names(center)
    y_vals <- apply(cube_points, 1, function(offset) {
      x <- center + offset
      x_df <- as.data.frame(as.list(as.numeric(x)))
      names(x_df) <- names(center)
      model_function(md, x_df)
    })
    
    y_center <- model_function(md, center)
    curvature_effect <- y_center - mean(y_vals)
    if (abs(curvature_effect)/y_center < threshold) {
      return(cube_beta_target_from_model(center, spacing, model_function, md, target))
    }
    
    spacing <- spacing / 10
  }
  
  # Fallback
  spacing <- 1e-12
  return(cube_beta_target_from_model(center, spacing, model_function, md, target))
}

# example usage of A2D2E
set.seed(123)
n <- 500
x1 <- runif(n, -2, 2)
x2 <- runif(n, -2, 2)
y  <- sin(pi * x1) + 0.5 * x2^2 + rnorm(n, sd = 0.2)
D  <- data.frame(x1 = x1, x2 = x2, y = y)

# Fit a regression model
md <- randomForest(y ~ ., data = D, ntree = 300)

# Define a prediction wrapper
pred_fun <- function(X.model, newdata) {
  as.numeric(predict(X.model, newdata = newdata))
}


# Apply A2D2EPlot to visualize main effect
K <- 15
sp <- 0.05

a2d2e_x1 <- A2D2EPlot(D = D[, c("x1", "x2")],
                      md = md,
                      predicted_model = pred_fun,
                      target_variable = "x1",
                      K = K,
                      sp = sp)

a2d2e_x2 <- A2D2EPlot(D = D[, c("x1", "x2")],
                      md = md,
                      predicted_model = pred_fun,
                      target_variable = "x2",
                      K = K,
                      sp = sp)

# Compare with ALE for validation
ale_x1 <- ALEPlot(D[, c("x1", "x2")], md, pred.fun = pred_fun, J = 1, K = K)
ale_x2 <- ALEPlot(D[, c("x1", "x2")], md, pred.fun = pred_fun, J = 2, K = K)

# Visualization
par(mfrow = c(1,2))
plot(a2d2e_x1$x.values, a2d2e_x1$f.values, type = "l", lwd = 2,
     col = "#e15759", main = "Main Effect of x1",
     xlab = "x1", ylab = "Estimated Effect")
lines(ale_x1$x.values, ale_x1$f.values, col = "#4e79a7", lwd = 2, lty = 2)
legend("topright", legend = c("A2D2E", "ALE"), col = c("#e15759", "#4e79a7"),
       lwd = 2, lty = c(1,2), bty = "n")

plot(a2d2e_x2$x.values, a2d2e_x2$f.values, type = "l", lwd = 2,
     col = "#e15759", main = "Main Effect of x2",
     xlab = "x2", ylab = "Estimated Effect")
lines(ale_x2$x.values, ale_x2$f.values, col = "#4e79a7", lwd = 2, lty = 2)
legend("topright", legend = c("A2D2E", "ALE"), col = c("#e15759", "#4e79a7"),
       lwd = 2, lty = c(1,2), bty = "n")