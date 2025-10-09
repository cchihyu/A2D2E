# Location to be evaluate
x_loc = seq(0.01,1,0.01)

# Function and gound truth for franke2d

franke2d <- function(xx)
{
  x1 <- xx[1]
  x2 <- xx[2]
  
  term1 <- 0.75 * exp(-(9*x1-2)^2/4 - (9*x2-2)^2/4)
  term2 <- 0.75 * exp(-(9*x1+1)^2/49 - (9*x2+1)/10)
  term3 <- 0.5 * exp(-(9*x1-7)^2/4 - (9*x2-3)^2/4)
  term4 <- -0.2 * exp(-(9*x1-4)^2 - (9*x2-7)^2)
  
  y <- term1 + term2 + term3 + term4 + rnorm(1,0,0.088)
  return(y)
}

franke2d_dx1 <- function(xx) {
  x1 <- xx[1]
  x2 <- xx[2]
  
  t1 <- 0.75 * exp(-(9*x1 - 2)^2 / 4 - (9*x2 - 2)^2 / 4)
  t2 <- 0.75 * exp(-(9*x1 + 1)^2 / 49 - (9*x2 + 1) / 10)
  t3 <- 0.5 * exp(-(9*x1 - 7)^2 / 4 - (9*x2 - 3)^2 / 4)
  t4 <- -0.2 * exp(-(9*x1 - 4)^2 - (9*x2 - 7)^2)
  
  dt1 <- -0.75 * t1 * (9 * (9*x1 - 2) / 2)
  dt2 <- -0.75 * t2 * (18 * (9*x1 + 1) / 49)
  dt3 <- -0.5 * t3 * (9 * (9*x1 - 7) / 2)
  dt4 <- 3.6 * t4 * (9*x1 - 4)
  
  return(dt1 + dt2 + dt3 + dt4)
}

franke2d_dx2 <- function(xx) {
  x1 <- xx[1]
  x2 <- xx[2]
  
  t1 <- 0.75 * exp(-(9*x1 - 2)^2 / 4 - (9*x2 - 2)^2 / 4)
  t2 <- 0.75 * exp(-(9*x1 + 1)^2 / 49 - (9*x2 + 1) / 10)
  t3 <- 0.5 * exp(-(9*x1 - 7)^2 / 4 - (9*x2 - 3)^2 / 4)
  t4 <- -0.2 * exp(-(9*x1 - 4)^2 - (9*x2 - 7)^2)
  
  dt1 <- -0.75 * t1 * (9 * (9*x2 - 2) / 2)
  dt2 <- -0.75 * t2 * (9 / 10)
  dt3 <- -0.5 * t3 * (9 * (9*x2 - 3) / 2)
  dt4 <- 3.6 * t4 * (9*x2 - 7)
  
  return(dt1 + dt2 + dt3 + dt4)
}

## dimension 1
truth_franke1 = NULL
for(i in x_loc){
  d = cubature::cubintegrate(franke2d_dx1, lower = c(0,0), upper = c(i,1))$integral
  truth_franke1 = c(truth_franke1,d)
}
truth_franke1 = truth_franke1-mean(truth_franke1)

## dimension 1
truth_franke2 = NULL
for(i in x_loc){
  d = cubature::cubintegrate(franke2d_dx2, lower = c(0,0), upper = c(1,i))$integral
  truth_franke2 = c(truth_franke2,d)
}
truth_franke2 = truth_franke2-mean(v2)

vfranke = rbind(x_loc, truth_franke1,truth_franke2)


# Write as plain text (tab separated)
write.table(vfranke,
            file = "/vfranke.txt",
            sep = "\t",
            row.names = FALSE,
            col.names = FALSE)


# Function and gound truth for simple1
v1 = x_loc^2
v1 = v1-mean(v1)

v2 = x_loc
v2 = v2-mean(v2)

vsimple1 = rbind(x_loc, v1,v2)

simple <- function(xx)
  return(xx[1]^2+xx[2]+rnorm(1,0,0.13))


# Write as plain text (tab separated)
write.table(vsimple1,
            file = "/vsimple1.txt",
            sep = "\t",
            row.names = FALSE,
            col.names = FALSE)

# Function and gound truth for grlee09

v5 = rep(0,100)
v6 = rep(0,100)
v4 = seq(-0.49,0.5,0.01)
v2 = seq(-0.245,0.25,0.005)
v3 = seq(-0.245,0.25,0.005)

v1 = NULL
f_hdhc_acc1_prime <- function(x) {
  u <- 0.9 * (x + 0.48)
  inner <- u^10
  return(exp(sin(inner)) * cos(inner) * 10 * u^9 * 0.9)
}

for(i in x_loc){
  d = cubature::cubintegrate(f_hdhc_acc1_prime, lower = c(0), upper = c(i))$integral
  v1 = c(v1,d)
}
v1 = v1-mean(v1)
vgrlee09 = rbind(x_loc,v1,v2,v3,v4,v5,v6)

grlee09 <- function(xx)
{
  
  x1 <- xx[1]
  x2 <- xx[2]
  x3 <- xx[3]
  x4 <- xx[4]
  x5 <- xx[5]
  x6 <- xx[6]
  
  term1 <- exp(sin((0.9*(x1+0.48))^10))
  term2 <- x2 * x3
  term3 <- x4
  
  y <- term1 + term2 + term3+rnorm(1,0,0.63)
  return(y)
}

write.table(vgrlee09,
            file = "/vgrlee.txt",
            sep = "\t",
            row.names = FALSE,
            col.names = FALSE)



# Function and gound truth for branin
braninsc <- function(xx)
{
  x1 <- xx[1]
  x2 <- xx[2]
  
  x1bar <- 15*x1 - 5
  x2bar <- 15 * x2
  
  term1 <- x2bar - 5.1*x1bar^2/(4*pi^2) + 5*x1bar/pi - 6
  term2 <- (10 - 10/(8*pi)) * cos(x1bar)
  
  y <- (term1^2 + term2 - 44.81) / 51.95+rnorm(1,0,0.31)
  return(y)
}

braninsc_dx1 <- function(xx) {
  x1 <- xx[1]
  x2 <- xx[2]
  
  x1bar <- 15 * x1 - 5
  x2bar <- 15 * x2
  
  term1 <- x2bar - (5.1 * x1bar^2) / (4 * pi^2) + (5 * x1bar) / pi - 6
  term2 <- (10 - 10 / (8 * pi)) * cos(x1bar)
  
  dterm1_dx1 <- (-5.1 / (2 * pi^2)) * x1bar * 15 + (5 / pi) * 15
  dterm2_dx1 <- -(10 - 10 / (8 * pi)) * sin(x1bar) * 15
  
  dy_dx1 <- (2 * term1 * dterm1_dx1 + dterm2_dx1) / 51.95
  return(dy_dx1)
}

braninsc_dx2 <- function(xx) {
  x1 <- xx[1]
  x2 <- xx[2]
  
  x1bar <- 15 * x1 - 5
  x2bar <- 15 * x2
  
  term1 <- x2bar - (5.1 * x1bar^2) / (4 * pi^2) + (5 * x1bar) / pi - 6
  
  dterm1_dx2 <- 15
  
  dy_dx2 <- (2 * term1 * dterm1_dx2) / 51.95
  return(dy_dx2)
}


v1 = NULL

for(i in x_loc){
  d = cubature::cubintegrate(braninsc_dx1, lower = c(0,0), upper = c(i,1))$integral
  v1 = c(v1,d)
}
v1 = v1-mean(v1)

v2 = NULL

for(i in x_loc){
  d = cubature::cubintegrate(braninsc_dx2, lower = c(0,0), upper = c(1,i))$integral
  v2 = c(v2,d)
}
v2 = v2-mean(v2)

vbraninsc = rbind(x_loc, v1,v2)

write.table(vbraninsc,
            file = "/vbranin.txt",
            sep = "\t",
            row.names = FALSE,
            col.names = FALSE)

# Function and gound truth for simple2

simple2 <- function(xx) {
  return(xx[1] * xx[2] - xx[2] * xx[3] + xx[4] * xx[1]+rnorm(1,0,0.11))
}

# Define partial derivatives
simple2_dx1 <- function(xx) {
  return(xx[2] + xx[4])
}

simple2_dx2 <- function(xx) {
  return(xx[1] - xx[3])
}

simple2_dx3 <- function(xx) {
  return(-xx[2])
}

simple2_dx4 <- function(xx) {
  return(xx[1])
}

# Grid for integration
x_loc <- seq(0.01, 1, 0.01)

# v1: integral of dx1 over x1 in [0, i]
v1 <- NULL
for (i in x_loc) {
  d <- cubature::cubintegrate(simple2_dx1, lower = c(0, 0, 0, 0), upper = c(i, 1, 1, 1))$integral
  v1 <- c(v1, d)
}
v1 <- v1 - mean(v1)

# v2: integral of dx2 over x2 in [0, i]
v2 <- NULL
for (i in x_loc) {
  d <- cubature::cubintegrate(simple2_dx2, lower = c(0, 0, 0, 0), upper = c(1, i, 1, 1))$integral
  v2 <- c(v2, d)
}
v2 <- v2 - mean(v2)

# v3: integral of dx3 over x3 in [0, i]
v3 <- NULL
for (i in x_loc) {
  d <- cubature::cubintegrate(simple2_dx3, lower = c(0, 0, 0, 0), upper = c(1, 1, i, 1))$integral
  v3 <- c(v3, d)
}
v3 <- v3 - mean(v3)

# v4: integral of dx4 over x4 in [0, i]
v4 <- NULL
for (i in x_loc) {
  d <- cubature::cubintegrate(simple2_dx4, lower = c(0, 0, 0, 0), upper = c(1, 1, 1, i))$integral
  v4 <- c(v4, d)
}
v4 <- v4 - mean(v4)

# Combine all results
vsimple2 <- rbind(x_loc, v1, v2, v3, v4)
write.table(vsimple2,
            file = "/vsimple2.txt",
            sep = "\t",
            row.names = FALSE,
            col.names = FALSE)


levy <- function(xx)
{
  d <- length(xx)
  w <- 1 + (xx - 1)/4
  
  term1 <- (sin(pi*w[1]))^2 
  term3 <- (w[d]-1)^2 * (1+1*(sin(2*pi*w[d]))^2)
  
  wi <- w[1:(d-1)]
  sum <- sum((wi-1)^2 * (1+10*(sin(pi*wi+1))^2))
  
  y <- term1 + sum + term3+rnorm(1,0,0.067)
  return(y)
}


# Define exact partial derivative function generator
levy_dxj <- function(j) {
  function(xx) {
    d <- length(xx)
    w <- 1 + (xx - 1) / 4
    dw_dx <- 1 / 4
    
    if (j == 1) {
      dterm1 <- 2 * sin(pi * w[1]) * cos(pi * w[1]) * pi
      dsum <- 2 * (w[1] - 1) * (1 + 10 * sin(pi * w[1] + 1)^2) +
        (w[1] - 1)^2 * 10 * 2 * sin(pi * w[1] + 1) * cos(pi * w[1] + 1) * pi
      return((dterm1 + dsum) * dw_dx)
    } else if (j == d) {
      wd <- w[d]
      dterm3 <- 2 * (wd - 1) * (1 + sin(2 * pi * wd)^2) +
        (wd - 1)^2 * 2 * sin(2 * pi * wd) * cos(2 * pi * wd) * 2 * pi
      return(dterm3 * dw_dx)
    } else {
      wj <- w[j]
      dsum <- 2 * (wj - 1) * (1 + 10 * sin(pi * wj + 1)^2) +
        (wj - 1)^2 * 10 * 2 * sin(pi * wj + 1) * cos(pi * wj + 1) * pi
      return(dsum * dw_dx)
    }
  }
}

# Compute integrated main effects for each variable
d <- 10
x_loc <- seq(0.01, 1, 0.01)
v_list <- list()

for (j in 1:d) {
  dxj_fun <- levy_dxj(j)
  vj <- NULL
  for (i in x_loc) {
    lower <- rep(0, d)
    upper <- rep(1, d)
    upper[j] <- i
    dval <- cubintegrate(dxj_fun, lower = lower, upper = upper)$integral
    vj <- c(vj, dval)
  }
  vj <- vj - mean(vj)
  v_list[[j]] <- vj
}

# Combine all results
vlevy <- rbind(x_loc, do.call(rbind, v_list))
rownames(vlevy) <- c("x_loc", paste0("v", 1:d))
write.table(vlevy,
            file = "/vlevy.txt",
            sep = "\t",
            row.names = FALSE,
            col.names = FALSE)

#friedman 
# Friedman function
fried <- function(xx) {
  x1 <- xx[1]; x2 <- xx[2]; x3 <- xx[3]; x4 <- xx[4]; x5 <- xx[5]
  10 * sin(pi * x1 * x2) + 20 * (x3 - 0.5)^2 + 10 * x4 + 5 * x5
}

# Constructor for a "friedman model"
fried_model <- function() {
  structure(list(), class = "fried_model")
}

# Predict method for fried_model
predict.fried_model <- function(object, newdata, ...) {
  if (is.vector(newdata)) {
    return(fried(newdata))
  }
  if (is.matrix(newdata)) {
    return(apply(newdata, 1, fried))
  }
  if (is.data.frame(newdata)) {
    return(apply(as.matrix(newdata), 1, fried))
  }
  stop("newdata must be a vector of length 5, a matrix with 5 columns, or a data.frame with 5 columns.")
}

# ---- Example usage ----
fit <- fried_model()
X <- matrix(runif(10000 * 5), ncol = 5)
y <- apply(X, 1, fried)
library(ALEPlot)
vfried = x_loc
yhat <- function(X.model, newdata) {
  return(predict(X.model, newdata = data.frame(newdata)))
}
for (target_var in 1:5) {
  pd_res   <- PDPlot(data.frame(X), fit, yhat, J = target_var, K = (100+1))
  res = c()
  for (x in x_loc[1:99]) {
    idx = which(pd_res$x.values>=x)[1]
    res = c(res,pd_res$f.values[idx])
  }
  res = c(res,res[99])
  vfried = rbind(vfried,res)
}


library(ALEPlot)

# ----------------------------
# Define benchmark functions
# ----------------------------
fried <- function(xx) {
  x1 <- xx[1]; x2 <- xx[2]; x3 <- xx[3]; x4 <- xx[4]; x5 <- xx[5]
  10 * sin(pi * x1 * x2) + 20 * (x3 - 0.5)^2 + 10 * x4 + 5 * x5
}

grlee09 <- function(xx) {
  x1 <- xx[1]; x2 <- xx[2]; x3 <- xx[3]; x4 <- xx[4]; x5 <- xx[5]; x6 <- xx[6]
  exp(sin((0.9 * (x1 + 0.48))^10)) + x2 * x3 + x4
}

detpep108d <- function(xx) {
  x1 <- xx[1]; x2 <- xx[2]; x3 <- xx[3]
  ii <- 4:8
  term1 <- 4 * (x1 - 2 + 8*x2 - 8*x2^2)^2
  term2 <- (3 - 4*x2)^2
  term3 <- 16 * sqrt(x3+1) * (2*x3-1)^2
  xxmat <- matrix(rep(xx[3:8], times = 6), 6, 6, byrow = TRUE)
  xxmatlow <- xxmat; xxmatlow[upper.tri(xxmatlow)] <- 0
  inner <- rowSums(xxmatlow)[2:6]
  outer <- sum(ii * log(1 + inner))
  term1 + term2 + term3 + outer
}

levy <- function(xx) {
  d <- length(xx)
  w <- 1 + (xx - 1) / 4
  term1 <- sin(pi * w[1])^2
  term2 <- sum((w[1:(d-1)] - 1)^2 *
                 (1 + 10 * sin(pi * w[1:(d-1)] + 1)^2))
  term3 <- (w[d] - 1)^2 * (1 + sin(2 * pi * w[d])^2)
  term1 + term2 + term3
}

ackley <- function(xx) {
  d <- length(xx)
  term1 <- -20 * exp(-0.2 * sqrt(sum(xx^2) / d))
  term2 <- -exp(sum(cos(2*pi*xx)) / d)
  term1 + term2 + 20 + exp(1)
}


# -----------------------------
# Borehole function definition
# -----------------------------
borehole <- function(xx) {
  rw <- xx[1]; r <- xx[2]; Tu <- xx[3]; Hu <- xx[4]
  Tl <- xx[5]; Hl <- xx[6]; L <- xx[7]; Kw <- xx[8]
  frac1 <- 2 * pi * Tu * (Hu - Hl)
  frac2a <- 2 * L * Tu / (log(r/rw) * rw^2 * Kw)
  frac2b <- Tu / Tl
  frac2 <- log(r/rw) * (1 + frac2a + frac2b)
  y <- frac1 / frac2
  return(y)
}

borehole01 <- function(x) {
  # x ∈ [0,1]^8
  stopifnot(length(x) == 8)
  
  # parameter ranges
  lower <- c(0.05,  100,   63070,  990,  63.1, 700, 1120,  9855)
  upper <- c(0.15, 50000, 115600, 1110, 116.0, 820, 16200, 12045)
  
  # rescale to original domain
  xx <- lower + x * (upper - lower)
  
  rw <- xx[1]
  r  <- xx[2]
  Tu <- xx[3]
  Hu <- xx[4]
  Tl <- xx[5]
  Hl <- xx[6]
  L  <- xx[7]
  Kw <- xx[8]
  
  frac1 <- 2 * pi * Tu * (Hu - Hl)
  
  frac2a <- 2 * L * Tu / (log(r/rw) * rw^2 * Kw)
  frac2b <- Tu / Tl
  frac2  <- log(r/rw) * (1 + frac2a + frac2b)
  
  y <- frac1 / frac2
  return(y)
}


# Input bounds
bounds <- matrix(c(
  0.05,   0.15,     # rw
  100.0,  50000.0,  # r
  63070.0,115600.0, # Tu
  990.0,  1110.0,   # Hu
  63.1,   116.0,    # Tl
  700.0,  820.0,    # Hl
  1120.0, 1680.0,   # L
  9855.0, 12045.0   # Kw
), ncol = 2, byrow = TRUE)

# Normalize / denormalize
denorm <- function(z) bounds[,1] + z * (bounds[,2] - bounds[,1])
f_norm <- function(z) borehole(denorm(z))
# ----------------------------
# Generic wrapper to make functions "predictable"
# ----------------------------
make_model <- function(fun) {
  structure(list(fun = fun), class = "custom_model")
}

predict.custom_model <- function(object, newdata, ...) {
  fun <- object$fun
  if (is.vector(newdata)) return(fun(newdata))
  if (is.matrix(newdata)) return(apply(newdata, 1, fun))
  if (is.data.frame(newdata)) return(apply(as.matrix(newdata), 1, fun))
  stop("newdata must be a vector, matrix, or data.frame.")
}

# ----------------------------
# Helper: run PD for one function
# ----------------------------
compute_v <- function(fun, d, x_loc) {
  fit <- make_model(fun)
  X <- matrix(runif(10000 * d), ncol = d)   # training data
  yhat <- function(X.model, newdata) predict(X.model, newdata = data.frame(newdata))
  
  vmat <- x_loc
  for (target_var in 1:d) {
    pd_res <- PDPlot(data.frame(X), fit, yhat, J = target_var, K = (length(x_loc)))
    res <- numeric()
    for (x in x_loc[1:(length(x_loc)-1)]) {
      idx <- which(pd_res$x.values >= x)[1]
      res <- c(res, pd_res$f.values[idx])
    }
    res <- c(res, res[length(res)])  # repeat last
    vmat <- rbind(vmat, res)
  }
  vmat
}

# Run for each function
library(ALEPlot)
x_loc <- seq(0, 1, length.out = 100)  # grid of 101 points

vfried     <- compute_v(fried,     d = 5, x_loc)
vgrlee09   <- compute_v(grlee09,   d = 6, x_loc)
vdetpep108d<- compute_v(detpep108d,d = 8, x_loc)
vlevy      <- compute_v(levy,      d = 6, x_loc)
vackley    <- compute_v(ackley,    d = 6, x_loc)
vfnorm    <- compute_v(f_norm,    d = 8, x_loc)


# Define save function
save_v <- function(vmat, fname) {
  write.table(vmat,
              file = fname,
              sep = "\t",
              row.names = FALSE,
              col.names = FALSE)
}

# Output folder (adjust path to your needs)
outdir <- "your_dir"

# Save each benchmark
save_v(vfried,      file.path(outdir, "vfried.txt"))
save_v(vgrlee09,    file.path(outdir, "vgrlee09.txt"))
save_v(vdetpep108d, file.path(outdir, "vdetpep108d.txt"))
save_v(vlevy,       file.path(outdir, "vlevy.txt"))
save_v(vackley,     file.path(outdir, "vackley.txt"))
save_v(vfnorm,     file.path(outdir, "fnorm.txt"))

detpep108d_high_gp = main_effect_exp(f = detpep108d, md = "GP", dep = "highdep", dim = 8, v = vdetpep108d, n_train = 400, n_rep = 10, grid_len = 40, sp=0.001)
levy_high_gp = main_effect_exp(f = levy, md = "GP", dep = "highdep", dim = 6, v = vlevy, n_train = 600, n_rep = 10, grid_len = 40, sp=0.001)
ackley_high_gp = main_effect_exp(f = ackley, md = "GP", dep = "highdep", dim = 6, v = vackley, n_train = 600, n_rep = 10, grid_len = 40, sp=0.001)
summarize_ci(friedman_high_gp)

