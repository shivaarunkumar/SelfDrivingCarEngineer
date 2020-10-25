//
// Created by Admin on 10/23/2020.
//

#include "classifier.h"
#include <math.h>
#include <string>
#include <vector>
#include <set>
#include <map>
#include <iostream>

using Eigen::ArrayXd;
using std::string;
using std::vector;
using std::set;
using std::map;

// Initializes GNB
GNB::GNB() {
    /**
     * TODO: Initialize GNB, if necessary. May depend on your implementation.
     */

    left_means = ArrayXd(4);
    left_means<<0,0,0,0;

    left_stds = ArrayXd(4);
    left_stds<<0,0,0,0;

    left_prior = 0;

    right_means = ArrayXd(4);
    right_means<<0,0,0,0;

    right_stds = ArrayXd(4);
    right_stds<<0,0,0,0;

    right_prior = 0;

    keep_means = ArrayXd(4);
    keep_means<<0,0,0,0;

    keep_stds = ArrayXd(4);
    keep_stds<<0,0,0,0;

    keep_prior = 0;
}

GNB::~GNB() {}

void GNB::train(const vector<vector<double>> &data,
                const vector<string> &labels) {
    /**
     * Trains the classifier with N data points and labels.
     * @param data - array of N observations
     *   - Each observation is a tuple with 4 values: s, d, s_dot and d_dot.
     *   - Example : [[3.5, 0.1, 5.9, -0.02],
     *                [8.0, -0.3, 3.0, 2.2],
     *                 ...
     *                ]
     * @param labels - array of N labels
     *   - Each label is one of "left", "keep", or "right".
     *
     * Implement the training function for your classifier.
     */


    float left_size = 0;
    float right_size = 0;
    float keep_size = 0;

    // Sum up all values by label
    for(int i = 0;i <data.size(); i++){
        if(labels[i]=="left"){
            left_means+=ArrayXd::Map(data[i].data(),data[i].size());
            left_size+=1;
        }
        else if(labels[i]=="right"){
            right_means+=ArrayXd::Map(data[i].data(),data[i].size());
            right_size+=1;
        }
        else{
            keep_means+=ArrayXd::Map(data[i].data(),data[i].size());
            keep_size+=1;
        }
    }
    left_means/=left_size;
    right_means/=right_size;
    keep_means/=keep_size;

    // Compute prior probability
    left_prior = left_size/data.size();
    right_prior = right_size/data.size();
    keep_prior = keep_size/data.size();

    // Compute Standard Deviations
    // Sum up all values by label
    for(int i = 0;i <data.size(); i++){
        ArrayXd data_point = ArrayXd::Map(data[i].data(),data[i].size());
        if(labels[i]=="left"){
            left_stds += (data_point-left_means)*(data_point-left_means);
        }
        else if(labels[i]=="right"){
            right_stds += (data_point-right_means)*(data_point-right_means);
        }
        else{
            keep_stds += (data_point-keep_means)*(data_point-keep_means);
        }
    }
    left_stds=(left_stds/left_size).sqrt();
    right_stds=(right_stds/right_size).sqrt();
    keep_stds=(keep_stds/keep_size).sqrt();

    std::cout<<"Done Training"<<std::endl;

}

string GNB::predict(const vector<double> &sample) {
    /**
     * Once trained, this method is called and expected to return
     *   a predicted behavior for the given observation.
     * @param observation - a 4 tuple with s, d, s_dot, d_dot.
     *   - Example: [3.5, 0.1, 8.5, -0.2]
     * @output A label representing the best guess of the classifier. Can
     *   be one of "left", "keep" or "right".
     *
     * Complete this function to return your classifier's prediction
     */

    // Calculate product of conditional probabilities for each label.
    double left_p = 1.0;
    double keep_p = 1.0;
    double right_p = 1.0;

    for (int i=0; i<4; ++i) {
        left_p *= (1.0/sqrt(2.0 * M_PI * pow(left_stds[i], 2)))
                  * exp(-0.5*pow(sample[i] - left_means[i], 2)/pow(left_stds[i], 2));
        keep_p *= (1.0/sqrt(2.0 * M_PI * pow(keep_stds[i], 2)))
                  * exp(-0.5*pow(sample[i] - keep_means[i], 2)/pow(keep_stds[i], 2));
        right_p *= (1.0/sqrt(2.0 * M_PI * pow(right_stds[i], 2)))
                   * exp(-0.5*pow(sample[i] - right_means[i], 2)/pow(right_stds[i], 2));
    }

    // Multiply each by the prior
    left_p *= left_prior;
    keep_p *= keep_prior;
    right_p *= right_prior;

    double probs[3] = {left_p, keep_p, right_p};
    double max = left_p;
    double max_index = 0;

    for (int i=1; i<3; ++i) {
        if (probs[i] > max) {
            max = probs[i];
            max_index = i;
        }
    }

    return this -> possible_labels[max_index];
}
