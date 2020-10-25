//
// Created by Admin on 10/23/2020.
//

#ifndef PARTICLE_FILTER_CLASSIFIER_H
#define PARTICLE_FILTER_CLASSIFIER_H
#include <string>
#include <vector>
#include "Eigen/Dense"

using Eigen::ArrayXd;
using std::string;
using std::vector;

class GNB {
private:
    ArrayXd left_means;
    ArrayXd left_stds;
    ArrayXd right_means;
    ArrayXd right_stds;
    ArrayXd keep_means;
    ArrayXd keep_stds;
    double left_prior;
    double right_prior;
    double keep_prior;

public:
    /**
     * Constructor
     */
    GNB();

    /**
     * Destructor
     */
    virtual ~GNB();

    /**
     * Train classifier
     */
    void train(const vector<vector<double>> &data,
               const vector<string> &labels);

    /**
     * Predict with trained classifier
     */
    string predict(const vector<double> &sample);

    vector<string> possible_labels = {"left","keep","right"};
};

#endif //PARTICLE_FILTER_CLASSIFIER_H
