full_loss = function(X, Y, w) {
  loss = 0
  for (i in 1:length(Y)) {
    loss = loss + log(1 + exp(- Y[i] * dot(w, X[i, ])))
  }
  loss = 1 / length(Y) * loss
  return(loss)
}

df_log_loss = function(x, y, w) {
  out = -x * y / (exp(dot(w, x) * y) + 1)
  return(out)
}

full_grad = function(X, Y, w) {
  n = length(Y)
  grad = df_log_loss(X[1, ], Y[1], w)
  for (i in 2:n) {
    grad = grad + df_log_loss(X[i, ], Y[i], w)
  }
  grad = 1 / n * grad
  return(grad)
}

gradient_descent = function(X, Y, w, eta, iter) {
  n = length(Y)
  grad_count = rep(0, iter)
  grad_counter = 0
  loss = rep(0, iter)
  for (i in 1:iter) {
    grad = full_grad(X, Y, w)
    grad_count[i] = grad_counter + n
    grad_counter = grad_counter + n
    w = w - eta * grad
    loss[i] = full_loss(X, Y, w)
  }
  out = list(grad_count, loss)
  return(out)
}
stochastic_gradient_descent = function(X, Y, w, eta, iter) {
  n = length(Y)
  grad_count = rep(0, iter)
  loss = rep(0, iter)
  for (i in 1:iter) {
    j = sample(1:n, 1)
    grad = df_log_loss(X[j, ], Y[j], w)
    grad_count[i] = i
    w = w - eta * grad
    loss[i] = full_loss(X, Y, w)
  }
  out = list(grad_count, loss)
  return(out)
  
}
SVRG = function(X, Y, w, eta, iter, m) {
  n = length(Y)
  grad_count = rep(0, iter)
  loss = rep(0, iter)
  grad_counter = 0
  for (i in 1:iter) {
    full_grad = full_grad(X, Y, w)
    w_old = w
    for (j in 1:m) {
      k = sample(1:n, 1)
      grad = df_log_loss(X[k, ], Y[k], w)
      grad_old = df_log_loss(X[k, ], Y[k], w_old)
      w = w - eta * (full_grad + grad - grad_old)
    }
    grad_count[i] = grad_counter + n + m
    grad_counter = grad_counter + n + m
    loss[i] = full_loss(X, Y, w)
  }
  out = list(grad_count, loss)
}

SAGA = function(X, Y, w, eta, iter, m) {
  n = length(Y)
  grad_table = matrix(ncol = n, nrow = length(w))
  grad_count = rep(0, iter)
  loss = rep(0, iter)
  for (i in 1:n) {
    grad_table[, i] = df_log_loss(X[i, ], Y[i], w)
  }
  for (k in 1:iter) {
    j = sample(1:n, 1)
    grad_curr = df_log_loss(X[j, ], Y[j], w)
    
    w = w - eta * (grad_curr - grad_table[, j] + 1 / n * c(sum(grad_table[1, ]), sum(grad_table[2, ])))
    grad_count[k] = k + n
    loss[k] = full_loss(X, Y, w)
  }
  out = list(grad_count, loss)
  return(out)
}