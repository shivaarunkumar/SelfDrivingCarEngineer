#include <iostream>
#include "kalman_filter.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

/* 
 * Please note that the Eigen library does not initialize 
 *   VectorXd or MatrixXd objects with zeros upon creation.
 */

KalmanFilter::KalmanFilter() {};

KalmanFilter::~KalmanFilter() {};

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  /**
   * predict the state
   */
   x_ = F_*x_;
   MatrixXd  Ft_ = F_.transpose();
   P_ = F_*P_*Ft_ + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  /**
   * update the state by using Kalman Filter equations
   */
    VectorXd y = z - H_*x_;

    CommonUpdate(y);
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  /**
   * update the state by using Extended Kalman Filter equations
   */

  //Retrieve cartesian coordinates
  double px = x_(0);
  double py = x_(1);
  double vx = x_(2);
  double vy = x_(3);

  double rho = sqrt(px*px + py*py);
  if (rho < .00001) {
    px += .0001;
    py += .0001;
    rho = sqrt(px*px + py*py);
  }
  double phi = atan2(py,px);


  double rho_dot= (px*vx+py*vy)/rho;
  VectorXd hx = VectorXd (3);
  hx << rho,phi,rho_dot;
  VectorXd y = z - hx;
  //Tried adding 2pi or subtracting 2pi, it yielded weird results.

  while ( y(1) > M_PI ) {
      y(1) -= 2*M_PI;
  }
  while(y(1) < -M_PI ) {
      y(1) += 2*M_PI;
  }
  CommonUpdate(y);
}

void KalmanFilter::CommonUpdate(const Eigen::VectorXd &y) {
    MatrixXd Ht_ = H_.transpose();
    MatrixXd S = H_ * P_ * Ht_ + R_;

    MatrixXd Si = S.inverse();
    MatrixXd K = P_*Ht_*Si;

    x_ = x_ + (K * y);
    MatrixXd  I = MatrixXd ::Identity(4,4);
    P_ = (I - K * H_) * P_;
}