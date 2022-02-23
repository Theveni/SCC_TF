// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>
#include <Rcpp.h>

using namespace Rcpp;

// [[Rcpp::export]]
List scvxama_fista_eigen(NumericMatrix matX, NumericVector vecw,
                    double gamma1, double gamma2, NumericVector vecgamma2_weight,
                    double nu, double tol_abs, double max_iter, int s)
{
  using namespace Eigen;
  Eigen::Map<Eigen::VectorXd> w(vecw.begin(), vecw.length());
  Eigen::Map<Eigen::MatrixXd> X(matX.begin(), matX.rows(), matX.cols());
  Eigen::Map<Eigen::VectorXd> gamma2_weight(vecgamma2_weight.begin(), vecgamma2_weight.length());

  int n = X.rows();
  int p = X.cols();
  int nk = 0;
  int np = 0;

  int eta = 0;
  IntegerVector diag_vec(n);
  diag_vec.fill(0);
  IntegerMatrix ix(s, 3);
  ix.fill(0);


  for (int i = 0; i < n - 1; i++)
  {
    for (int j = i + 1; j < n; j++)
    {
      if (vecw(np) > 0)
      {
        ix(nk, 0) = i;
        ix(nk, 1) = j;
        ix(nk, 2) = np;
        nk += 1;
      }
      np += 1;
    }
  }

  MatrixXd Lambda(nk, p);
  Lambda.fill(0);
  MatrixXd V(nk, p);
  MatrixXd L(n, n);
  L.fill(0);
  int l1;
  int l2;
  int l3;
  for (int l = 0; l < nk; l++)
  {
    l1 = ix(l, 0);
    l2 = ix(l, 1);
    L(l1, l2) = -1;
    L(l2, l1) = -1;
    L(l1, l1) += 1;
    L(l2, l2) += 1;
    diag_vec(l1) += 1;
    diag_vec(l2) += 1;
  }
  eta = which_max(diag_vec);
  eta = diag_vec(eta);

  // defined for update A
  VectorXd z(n);
  MatrixXd A(n, p);
  A = X.array();
  MatrixXd A_new(n, p);
  
  Eigen::MatrixXd B(n, p);
  B.fill(0);
  Eigen::VectorXd prox(p);

  Eigen::MatrixXd u_new(n, p), grad(n, p);
  u_new = A;

  // defined for update V
  double norm_lambda;
  double norm_term3;
  VectorXd term3(p);
  MatrixXd Lambda0(nk, p),Lambda1(nk, p);
  Lambda0.fill(0);Lambda1.fill(0);
  
  double t1,t2;
  t1=1;

  // defined for while loop
  VectorXd eva(2);
  eva.fill(1); // evaluate convergence
  int iter = 0;

  while ((iter < max_iter) & (eva.maxCoeff() > tol_abs))
  {
    iter++;
    
    //update U
    
      
      B.fill(0);
      for (int i = 0; i < nk; i++)
      {
        l1 = ix(i, 0);
        l2 = ix(i, 1);
        l3 = ix(i, 2);
        B.row(l1) += Lambda.row(i);
        B.row(l2) -= Lambda.row(i);
      }
      u_new = X - B;
      
      
      for (int i = 0; i < p; i++)
      {
        z = u_new.col(i);
        u_new.col(i) = std::max(0.0, 1 -  gamma2 * gamma2_weight(i) / z.norm()) * z;
      }
    eva(0) = (A - u_new).norm();
    A = u_new;
    
    
    
    //update Lambda
    for (int l = 0; l < nk; l++)
    {
      l1 = ix(l, 0);
      l2 = ix(l, 1);
      l3 = ix(l, 2);
      term3 = Lambda.row(l);
      term3 += nu * (A.row(l1) - A.row(l2));
      norm_lambda = gamma1 * w(l3);
      norm_term3 = term3.norm();
      
      // L2 dual norm projection on to ball
      if (norm_term3 < norm_lambda)
      {
        Lambda0.row(l) = term3;
      }
      else
      {
        Lambda0.row(l) = term3 / norm_term3 * norm_lambda;
      }
    }
    eva(1) = (Lambda1-Lambda0).norm();
    t2=(1+sqrt(1+4*t1*t1))/2;
    Lambda = (t1-1)/t2*(Lambda0-Lambda1)+Lambda0;
    t1=t2;
    Lambda1 = Lambda0;
    
  }
  // update V after convergence
  for (int l = 0; l < nk; l++)
  {
    l1 = ix(l, 0);
    l2 = ix(l, 1);
    V.row(l) = A.row(l1) - A.row(l2);
  }

  return Rcpp::List::create(Rcpp::Named("A") = A,
                            Rcpp::Named("V") = V,
                            Rcpp::Named("Lambda") = Lambda,
                            Rcpp::Named("iter") = iter);
}
// [[Rcpp::export]]
double subsum(NumericMatrix matX, NumericMatrix matA,
              NumericVector vecw, NumericVector vecgamma2_weight,
              double gamma1, double gamma2)
{
  Eigen::Map<Eigen::VectorXd> w(vecw.begin(), vecw.length());
  Eigen::Map<Eigen::MatrixXd> X(matX.begin(), matX.rows(), matX.cols());
  Eigen::Map<Eigen::MatrixXd> A(matA.begin(), matA.rows(), matA.cols());
  Eigen::Map<Eigen::VectorXd> gamma2_weight(vecgamma2_weight.begin(), vecgamma2_weight.length());

  int n = matX.rows();
  int p = matX.cols();
  int np = 0;
  double ss = 0;

  for (int i = 0; i < (n - 1); i++)
  {
    ss += (X.row(i) - A.row(i)).squaredNorm() / 2;
    for (int j = i + 1; j < n; j++)
    {
      ss += gamma1 * w(np) * (A.row(i) - A.row(j)).norm();
      np += 1;
    }
  }
  ss += (X.row(n-1) - A.row(n-1)).squaredNorm() / 2;

  for (int l = 0; l < p; l++)
  {
    ss += gamma2 * gamma2_weight(l) * A.col(l).norm();
  }
  return (ss);
}
