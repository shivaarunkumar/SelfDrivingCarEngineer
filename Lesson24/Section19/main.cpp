//
// Created by Admin on 9/20/2020.
//

#include <iostream>
#include "Eigen/Dense"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::cout;
using std::endl;

MatrixXd CalculateJacobian(const VectorXd& x_state);

int main() {
    /**
     * Compute the Jacobian Matrix
     */

    // predicted state example
    // px = 1, py = 2, vx = 0.2, vy = 0.4
    VectorXd x_predicted(4);
    x_predicted << 1, 2, 0.2, 0.4;

    MatrixXd Hj = CalculateJacobian(x_predicted);

    cout << "Hj:" << endl << Hj << endl;

    return 0;
}

MatrixXd CalculateJacobian(const VectorXd& x_state) {

    MatrixXd Hj(3,4);
    // recover state parameters
    double px = x_state(0);
    double py = x_state(1);
    double vx = x_state(2);
    double vy = x_state(3);

    // check division by zero
    double rho2 = pow(px,2) + pow(py,2);
    if(rho2==0){
        std::cout << "Cannot divide by zero "<< std::endl;
        return Hj;
    }

    // compute the Jacobian matrix
    Hj(0,0) = px / sqrt(rho2);
    Hj(0,1) = py / sqrt(rho2);
    Hj(1,0) = -py / rho2;
    Hj(1,1) = px / rho2;
    Hj(2,0) = (py * (vx*py-vy*px) ) / sqrt(rho2);
    Hj(2,1) = (px * (vy*px - vx*py)) / sqrt(rho2);
    Hj(2,2) = px / sqrt(rho2);
    Hj(2,3) = py / sqrt(rho2);

    return Hj;
}
