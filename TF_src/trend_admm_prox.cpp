// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>
#include <Rcpp.h>

using namespace Rcpp;

// [[Rcpp::export]]
List trend_admm(NumericVector vecX, int order,
                double lambda, double tol_abs, double max_iter){
  using namespace Eigen;
  Eigen::Map<Eigen::VectorXd> X(vecX.begin(), vecX.length());
  
  
  int n = vecX.length();
  IntegerVector diag_vec(n);diag_vec.fill(0);
  
  MatrixXd D(n-order,n),D_trans(n,n-order);D.fill(0);
  D(0,0)=double(1);
  
  for(int i = 0;i<order;i++){
    D(0,i+1)=D(0,i)*(order-i)/(i+1)*(-1);
    for(int j = 1;j<n-order;j++){
      D(j,j+i)=D(0,i); 
    }
  }
  for(int j = 1;j<n-order;j++){
    D(j,j+order)=D(0,order); 
  }
  
  
  Eigen::VectorXd Lambda(n-order),Lambda0(n-order);
  Lambda.fill(0);
  
  D_trans = D.transpose();
  
  Eigen::VectorXd A(n);A=X.array();
  Eigen::VectorXd u(n);u=X.array();
  Eigen::VectorXd u_new(n),u_old(n);
  Eigen::VectorXd prox(n-order),grad(n);
  
  const Eigen::VectorXi pos = Eigen::VectorXi::Ones(n-order);
  const Eigen::VectorXi neg = Eigen::VectorXi::Ones(n-order) * -1;
  VectorXd v(n-order);v.fill(0); 
  Eigen::VectorXi a_sign(n-order);
  
  
  // defined for while loop
  VectorXd eva(2); eva.fill(1); // evaluate convergence
  int iter = 0;
  double rho = 1;
  //print(ix);
  
  MatrixXd I = MatrixXd::Identity(n,n);
  
  MatrixXd M = (I+rho*D_trans*D).inverse();
  
  
  while( (iter < max_iter) & (eva(0) > tol_abs)){
    iter++;
    //std::cout<<"---------------------"<<std::endl;
    //newton
    
    u = M * (X+rho * D_trans*(v+Lambda));
    
    //std::cout<<u<<std::endl;
    
    eva(0) = (A - u).norm();
    //eva(0) = 1;
    A = u;
    
    grad = D*u - Lambda;
    a_sign=(grad.array()>=0).select(pos,neg);
    for(int i = 0;i<n-order;i++){
      v(i) = std::max(double(0),std::abs(grad(i))-lambda/rho)*a_sign(i);
    }
    //update Lambda
    Lambda0 = Lambda + rho*(v-D*u);
    
    //eva(1) = 1;
    Lambda = Lambda0;
    
  }
  
  
  return Rcpp::List::create(Rcpp::Named("A") = A,
                            Rcpp::Named("Lambda") = Lambda,
                            Rcpp::Named("iter") = iter
  );
}


// [[Rcpp::export]]
List trend_eigen(NumericVector vecX, int order,
                 double lambda, double tol_abs, double max_iter){
  using namespace Eigen;
  Eigen::Map<Eigen::VectorXd> X(vecX.begin(), vecX.length());
  
  
  int n = vecX.length();
  
  int eta = pow(4,order);
  
  IntegerVector diag_vec(n);diag_vec.fill(0);
  
  MatrixXd D(n-order,n);D.fill(0);
  MatrixXd D_trans(n,n-order);
  D(0,0)=double(1);
  
  for(int i = 0;i<order;i++){
    D(0,i+1)=D(0,i)*(order-i)/(i+1)*(-1);
    for(int j = 1;j<n-order;j++){
      D(j,j+i)=D(0,i); 
    }
  }
  for(int j = 1;j<n-order;j++){
    D(j,j+order)=D(0,order); 
  }
  Eigen::VectorXd Lambda(n-order),Lambda0(n-order);
  
  D_trans=D.transpose();
  
  //Lambda = (D*D_trans).inverse()*D*X;
  
  //for(int i = 0;i < n-order;i++){
    //Lambda(i)=std::min(double(1),lambda/std::abs(Lambda(i)))*Lambda(i);
  //}
  //Lambda = (D*D_trans).inverse()*D*y
  //Lambda.fill(0);
  
  double t1,t2;
  
  
  Eigen::VectorXd A(n);A=X.array();
  Eigen::VectorXd u(n);u=X.array();
  Eigen::VectorXd u_new(n),u_old(n);
  Eigen::VectorXd prox(n-order),grad(n);
  
  u_old=u;
  u_new=u;
  
  
  double dis;
  double nu = 1;
  double tol;
  
  // defined for while loop
  VectorXd eva(2); eva.fill(1); // evaluate convergence
  int iter = 0;
  
  
  while( (iter < max_iter) & (eva(0) > tol_abs)){
    iter++;
    t1=1;
    //update A
    
    dis = 10;
    tol = std::min(0.1,sqrt(1/nu))/iter;
    //newton
    while(dis>tol){
      
      
      prox=nu*D*u+Lambda;
      
      for(int i = 0; i<n-order;i++){
        prox(i)=std::min(double(1),lambda/std::abs(prox(i)))*prox(i);
      }
      grad=u-X+D_trans*prox;
      
      
      u_new= u- 1/(double(1)+eta*nu)*grad;
      
      t2=(1+sqrt(1+4*t1*t1))/2;
      u=(t1-1)/t2*(u_new-u_old)+u_new;
      dis = grad.norm();
      u_old=u_new;
      t1=t2;
      
    }
    
    
    eva(0) = (A - u_new).norm();
    //eva(0) = 1;
    A = u_new;
    
    
    //update Lambda
    Lambda0 = Lambda + nu*D*A;
    
    for(int i = 0;i < n-order;i++){
      Lambda0(i)=std::min(double(1),lambda/std::abs(Lambda0(i)))*Lambda0(i);
    }
    
    //eva(1) = 1;
    Lambda = Lambda0;
    
    
    nu = std::min(50.0, 1.1 * nu);
  }
  
  
  return Rcpp::List::create(Rcpp::Named("A") = A,
                            Rcpp::Named("Lambda") = Lambda,
                            Rcpp::Named("iter") = iter
  );
}