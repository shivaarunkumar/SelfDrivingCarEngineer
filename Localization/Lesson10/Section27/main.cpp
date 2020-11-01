//
// Created by Admin on 11/1/2020.
//

#include <cmath>
#include <iostream>
#include <vector>

#include "Eigen/Dense"
#include "grader.h"

using std::vector;
using Eigen::MatrixXd;
using Eigen::VectorXd;

/**
 *  complete this function
 */
vector<double> JMT(vector<double> &start, vector<double> &end, double T) {
    /**
     * Calculate the Jerk Minimizing Trajectory that connects the initial state
     * to the final state in time T.
     *
     * @param start - the vehicles start location given as a length three array
     *   corresponding to initial values of [s, s_dot, s_double_dot]
     * @param end - the desired end state for vehicle. Like "start" this is a
     *   length three array.
     * @param T - The duration, in seconds, over which this maneuver should occur.
     *
     * @output an array of length 6, each value corresponding to a coefficent in
     *   the polynomial:
     *   s(t) = a_0 + a_1 * t + a_2 * t**2 + a_3 * t**3 + a_4 * t**4 + a_5 * t**5
     *
     * EXAMPLE
     *   > JMT([0, 10, 0], [10, 10, 0], 1)
     *     [0.0, 10.0, 0.0, 0.0, 0.0, 0.0]
     */
    vector<double> parameters;
    double alpha0 = start[0];
    double alpha1 = start[1];
    double alpha2 = start[2]/2;
    parameters.push_back(alpha0);
    parameters.push_back(alpha1);
    parameters.push_back(alpha2);

    MatrixXd C(3,1);
    C(0,0) = end[0]-(start[0]+start[1]*T+.5*start[2]*pow(T,2));
    C(1,0) = end[1] - (start[1] + start[2]* T);
    C(2,0) = end[2] - start[2];

    MatrixXd A(3,3);
    A << pow(T,3) , pow(T,4), pow(T,5),
         3*pow(T,2), 4* pow(T,3), 5*pow(T,4),
         6*T, 12*pow(T,2), 20*pow(T,3);
    MatrixXd B(3,1);
    B = A.inverse() * C;
    parameters.push_back(B(0,0));
    parameters.push_back(B(1,0));
    parameters.push_back(B(2,0));
    return parameters;
}

int main() {

    // create test cases
    vector< test_case > tc = create_tests();

    bool total_correct = true;

    for(int i = 0; i < tc.size(); ++i) {
        vector<double> jmt = JMT(tc[i].start, tc[i].end, tc[i].T);
        bool correct = close_enough(jmt, answers[i]);
        total_correct &= correct;
    }

    if(!total_correct) {
        std::cout << "Try again!" << std::endl;
    } else {
        std::cout << "Nice work!" << std::endl;
    }

    return 0;
}