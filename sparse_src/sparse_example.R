library(Rcpp)
library(scvxclustr)
library(RcppEigen)


Rcpp::sourceCpp("sparse_APGM.cpp")
Rcpp::sourceCpp("sparse_prox_fista.cpp")
Rcpp::sourceCpp("sparse_prox_ama.cpp")
Rcpp::sourceCpp("sparse_prox_ama_fista.cpp")

simu = function(n, true_p, p, k, mu, sigma, seed = NULL){
  
  if(! is.null(seed) ){ set.seed(seed)  }
  n_p <- true_p / 2
  n_k <- n / k
  clust.ind <- rep(1:k, each = n_k)
  clust.mat <- rbind( c(rep( mu, n_p), rep(-mu, n_p)), 
                      c(rep(-mu, n_p), rep(-mu, n_p)),
                      c(rep(-mu, n_p), rep( mu, n_p)),
                      c(rep( mu, n_p), rep( mu, n_p)),
                      c(rep(0,n_p),rep(0,n_p))
  )
  
  X = matrix(0,n,p)
  for(i in 1:n){
    mu_mean <- c( clust.mat[clust.ind[i],], rep(0, p - true_p) )
    X[i,] <- rnorm(p, mu_mean,rep(sigma,p))
  }
  
  list(X = X, label = clust.ind, features = c(rep(TRUE, true_p), rep(FALSE, p - true_p)))
  
}

n = 100# Sample size
true_p = 20    # Number of true features
p = 500     # Number of total features 
k = 5           # Number of true cluster
mu = 2        # mean of normal distribution
sigma = 1       # sd of normal distribution


# Simiulate 4 cluster Gaussian data
data <- simu(n = n, true_p = true_p, p = p, k = k, mu = mu, sigma = sigma )

#standardize n by p data matrix
X <- matrix(data$X,n,p)


# Adaptive Weight (if possible)
g1 <- 5
g2 <- 5
Gamma2.weight <- c(rep(0.5,true_p),rep(1,p-true_p))
k_w <- 5   # Number of nearest neighbors
phi <- 0.5  # scale of the kernel
w <- dist_weight( t(X)/sqrt(p),phi, dist.type = "euclidean", p = 2 )
w <- knn_weights(w,k = k_w,n)
s=sum(w!=0)


##proposed method
scvxprox_eigen(X,w,g1,g2,Gamma2.weight,1,0.001,100,s)

##generalized ADMM
scvxapgm_eigen(X,w,g1,g2,Gamma2.weight,1,0.0005,1000,s)

##AMA
scvxama_eigen(X,w,g1,g2,Gamma2.weight,1/n,0.0005,10000,s)

##AMA FISTA
scvxama_fista_eigen(X,w,g1,g2,Gamma2.weight,1/n.l,0.0005,10000,s)
