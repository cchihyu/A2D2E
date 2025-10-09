library(openai)
Sys.setenv(
  OPENAI_API_KEY = 'enter your key'
)
# Prereqs
# Fit function (unchanged, except x can be matrix)
fit_chatgpt <- function(x, y,
                        model = "gpt-4o-mini",
                        temperature = 0,
                        system_prompt = "You are a regression model. Learn y as a function of x from the data. Always return only a numeric scalar.",
                        max_points = 200) {
  if (is.vector(x)) x <- matrix(x, ncol = 1)
  stopifnot(nrow(x) == length(y))
  dat <- data.frame(x, y = as.numeric(y))
  if (nrow(dat) > max_points) {
    set.seed(0); dat <- dat[sample.int(nrow(dat), max_points), , drop = FALSE]
  }
  structure(list(data = dat, model = model, temperature = temperature,
                 system_prompt = system_prompt),
            class = "MyChatGPTModel")
}

# build prompt for arbitrary d-dimensional x
.build_prompt <- function(dat, x_target) {
  # training pairs: ((x1,...,xd), y)
  X <- as.matrix(dat[, !names(dat) %in% "y", drop = FALSE])
  pairs <- apply(cbind(X, dat$y), 1, function(row) {
    paste0("((", paste(sprintf("%.6f", row[1:(length(row)-1)]), collapse = ","), 
           "),", sprintf("%.6f", row[length(row)]), ")")
  })
  train_block <- paste(pairs, collapse = ", ")
  
  # target
  xtxt <- paste(sprintf("%.6f", x_target), collapse = ",")
  sprintf(paste0(
    "You are given training data in the form of ((x1,...,xd),y):\n%s\n\n",
    "Question: What is your prediction of y at x = (%s)?\n",
    "Return only a single numerical scalar with no explanation."
  ), train_block, xtxt)
}

# Predict method (multi-d)
predict.MyChatGPTModel <- function(object, newdata, ...) {
  if (is.null(newdata)) stop("newdata must be provided.")
  Xnew <- as.matrix(newdata)
  
  apply(Xnew, 1, function(xi) {
    user_prompt <- .build_prompt(object$data, xi)
    resp <- openai::create_chat_completion(
      model = object$model, temperature = object$temperature,
      messages = list(
        list(role = "system", content = object$system_prompt),
        list(role = "user",   content = user_prompt)
      )
    )
    txt <- resp$choices$message.content
    m <- regmatches(txt, regexpr("-?\\d+(?:[.]\\d+)?(?:[eE][+-]?\\d+)?", txt, perl = TRUE))
    val <- suppressWarnings(as.numeric(m))
    if (is.na(val)) stop("Model did not return a numeric scalar. Got: ", txt)

    val
  })
}

# Example usage of GPT agent

# make 2-dim example
X <- cbind(runif(30), runif(30))
y <- X[,1] + 2*X[,2] + rnorm(30, 0, 0.1)

m2 <- fit_chatgpt(X, y, model = "gpt-4o-mini", temperature = 0)

# predict at two new points
newX <- rbind(c(0.2, 0.5))
preds <- predict(m2, newX)

preds
