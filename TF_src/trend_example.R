library(Rcpp)

setwd("C:/Users/ryosu/OneDrive/R/script/Rcpp/trendfilter/zikken/github")

Rcpp::sourceCpp("trend_admm_prox.cpp")



order = 2;
n=1000; y=5*sin(1:n/n*2*pi)+rnorm(n) ## データ生成



trend_eigen(y,order+1,100,0.001,100)

trend_admm(y,order+1,100,0.001,1000)



