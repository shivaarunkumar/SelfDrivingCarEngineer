#include "PID.h"
#include<iostream>
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

    // Initialize the Errors
    p_error = 0;
    d_error = 0;
    i_error = 0;


}

void PID::UpdateError(double cte) {
    /**
     * Update PID errors based on cte.
     */
    p_error = cte;
    d_error = cte - p_error;
    i_error += cte;

    //std::cout<<p_error<<"\t"<<d_error<<"\t"<<i_error<<std::endl;
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

void PID::DisplayParams(){
    std::cout<<Kp<<"\t"<<Ki<<"\t"<<Kd<<std::endl;
}