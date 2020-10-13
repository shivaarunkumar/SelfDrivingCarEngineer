#include "FusionEKF.h"
#include <iostream>
#include "Eigen/Dense"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::cout;
using std::endl;
using std::vector;

/**
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);
  Hj_ = MatrixXd(3, 4);

  //measurement covariance matrix - laser
  R_laser_ << 0.0225, 0,
              0, 0.0225;

  //measurement covariance matrix - radar
  R_radar_ << 0.09, 0, 0,
              0, 0.0009, 0,
              0, 0, 0.09;

  /**
   * Finish initializing the FusionEKF.
   * Set the process and measurement noises
   */

  // Sensor function for laser measurements (position only)
  H_laser_ << 1,0,0,0,
                0,1,0,0;
  // Initial State Transition Matrix F (will be updated with dt in processMeasurements)
  ekf_.F_ = MatrixXd (4,4);
  ekf_.F_ << 1,0,1,0,
             0,1,0,1,
             0,0,1,0,
             0,0,0,1;

  // State Covariance matrix P
  ekf_.P_ = MatrixXd (4,4);
  ekf_.P_ << 1,0,0,0,
             0,1,0,0,
             0,0,1000,0,
             0,0,0,1000;
}

/**
 * Destructor.
 */
FusionEKF::~FusionEKF() {};

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {
  /**
   * Initialization
   */
  if (!is_initialized_) {
    /**
     * Initialize the state ekf_.x_ with the first measurement.
     * Create the covariance matrix.
     * You'll need to convert radar from polar to cartesian coordinates.
     */

    // first measurement
    cout << "EKF: " << endl;
    ekf_.x_ = VectorXd(4);
    ekf_.x_ << 0,0,0,0;

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      // Convert radar from polar to cartesian coordinates
      //         and initialize state.
        double rho = measurement_pack.raw_measurements_(0);
        double phi = measurement_pack.raw_measurements_(1);
        double rho_dot = measurement_pack.raw_measurements_(2);

        ekf_.x_(0) = rho*cos(phi); //px
        ekf_.x_(1) = rho*sin(phi); //py
        ekf_.x_(2) = rho_dot*cos(phi); //vx
        ekf_.x_(3) = rho_dot*sin(phi); //vy


    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      // Initialize state : Unpack the  cartesian position values from the raw measurements.
        ekf_.x_(0) = measurement_pack.raw_measurements_(0);
        ekf_.x_(1) = measurement_pack.raw_measurements_(1);
    }
    previous_timestamp_ = measurement_pack.timestamp_;
    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }

  /**
   * Prediction
   */

  /**
   * Update the state transition matrix F according to the new elapsed time.
   * Time is measured in seconds.
   * Update the process noise covariance matrix.
   * Use noise_ax = 9 and noise_ay = 9 for your Q matrix.
   */
  double dt = (double)(measurement_pack.timestamp_ - previous_timestamp_)/1e6;
  previous_timestamp_ = measurement_pack.timestamp_;

  // Update state transition matrix with elapsed time
  ekf_.F_(0,2) = dt;
  ekf_.F_(1,3) = dt;

  // Compute process noise
  double noise_ax = 9;
  double noise_ay = 9;
  double dt44 = pow(dt, 4) / 4;
  double dt32 = pow(dt, 3) / 2;
  double dt2 = pow(dt, 2);
  ekf_.Q_ = MatrixXd(4, 4);
  ekf_.Q_ << dt44 * noise_ax, 0, dt32* noise_ax,0,
            0,dt44* noise_ay, 0, dt32* noise_ay,
            dt32* noise_ax, 0, dt2* noise_ax, 0,
            0, dt32* noise_ay, 0, dt2* noise_ay;


  ekf_.Predict();

  /**
   * Update
   */

  /**
   *
   * - Use the sensor type to perform the update step.
   * - Update the state and covariance matrices.
   */

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    // Radar updates
    // Local linearization using Jacobian is required
    ekf_.H_ = tools.CalculateJacobian(ekf_.x_);
    ekf_.R_ = R_radar_;
    ekf_.UpdateEKF(measurement_pack.raw_measurements_);

  } else {
    // Laser updates
    // Straightforward linear operations
    ekf_.H_ = H_laser_;
    ekf_.R_ = R_laser_;

    ekf_.Update(measurement_pack.raw_measurements_);

  }

  // print the output
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
}
