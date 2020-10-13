#include "tools.h"
#include <iostream>

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;


Tools::Tools() = default;

Tools::~Tools() = default;

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  /**
   * Calculate the RMSE here.
   */

  VectorXd rmse(4);
  rmse << 0,0,0,0; //Initialize

  // Sanity Checks for required sizes
  if(estimations.empty() || (estimations.size() != ground_truth.size())){
      std::cerr<<"Data sizes are incorrect to compute RMSE"<<std::endl;
      return rmse;
  }

  for (unsigned int i = 0; i< estimations.size();i++)
  {
      VectorXd residual = estimations[i]-ground_truth[i];
      residual = residual.array()*residual.array();
      rmse += residual;
  }
  //Mean
  rmse = rmse/estimations.size();
  //Square root
  rmse = rmse.array().sqrt();

  return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  /**
   * Calculate a Jacobian here.
   */
    MatrixXd Hj(3,4); //3 polar states and 4 cartesian states

    double px = x_state(0);
    double py = x_state(1);
    double vx = x_state(2);
    double vy = x_state(3);

    double rho2 = px*px + py*py;
    double rho = sqrt(rho2);
    double rho32 = rho * rho2;

    if(fabs(rho2) < 1e-4){
        std::cout << "CalculateJacobian () - Avoiding Error - Division by Zero" << std::endl;
        px += 0.0001;
        py += 0.0001;
        rho2 = px*px+py*py;
    }

    Hj << px/rho,py/rho,0,0,
          -(py/rho2),(px/rho2),0,0,
          py*(vx*py-vy*px)/rho32,px*(vy*px-vx*py)/rho32,px/rho,py/rho;

    return Hj;
}


