#include "PID.h"

/**
 * TODO: Complete the PID class. You may add any additional desired functions.
 */

PID::PID() {}

PID::~PID() {}

void PID::Init(double Kp_, double Ki_, double Kd_) {
    /**
     * Initialize PID coefficients (and errors, if needed)
     */
    Kp = Kp_;
    Ki = Ki_;
    Kd = Kd_;
}

void PID::UpdateError(double cte) {
    /**
     * Update PID errors based on cte.
     */
    p_error = cte;
    d_error = cte - d_error;
    i_error += cte;
}

double PID::TotalError() {
    /**
     * Calculate and return the total error
     */
    double totalerror = -Kp * p_error - Kd * d_error - Ki * i_error;
    if(totalerror<-1)
        totalerror = -1;
    if(totalerror>1)
        totalerror = 1;
    return totalerror;
}