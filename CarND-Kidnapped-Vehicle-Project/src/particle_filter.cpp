/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>


#include "helper_functions.h"

using std::string;
using std::vector;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
    /**
     * Set the number of particles. Initialize all particles to
     *   first position (based on estimates of x, y, theta and their uncertainties
     *   from GPS) and all weights to 1.
     * Add random Gaussian noise to each particle.
     * NOTE: Consult particle_filter.h for more information about this method
     *   (and others in this file).
     */

    num_particles = 1000; // Set the number of particles
    // Initialize all particle weights to one
    weights = std::vector<double>(num_particles, 1.0);

    //create required distributions
    std::default_random_engine gen;
    std::normal_distribution<double> dist_x(x, std[0]);
    std::normal_distribution<double> dist_y(y, std[1]);
    std::normal_distribution<double> dist_theta(theta, std[2]);

    // Generate particles
    particles = std::vector<Particle>(num_particles);

    for (int i = 0; i < num_particles; i++) {
        particles[i].x = dist_x(gen);
        particles[i].y = dist_y(gen);
        particles[i].theta = dist_theta(gen);
        particles[i].id = i;
        particles[i].weight = weights[i];
    }

    //Initialization Complete
    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[],
                                double velocity, double yaw_rate) {
    /**
     * Add measurements to each particle and add random Gaussian noise.
     * NOTE: When adding noise you may find std::normal_distribution
     *   and std::default_random_engine useful.
     *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
     *  http://www.cplusplus.com/reference/random/default_random_engine/
     */
    std::default_random_engine gen;
    std::normal_distribution<double> dist_x(0, std_pos[1]);
    std::normal_distribution<double> dist_y(0, std_pos[1]);
    std::normal_distribution<double> dist_theta(0, std_pos[1]);


    for (int i = 0; i < num_particles; i++) {
        /*
         * There are two cases to consider when defining the motion model.
         * Case 1: Yaw Rate (theta_dot) = 0
         *   x_final = x0 + v*dt*cos(theta)
         *   y_final = x0 + v*dt*sin(theta)
         *   theta_final = theta
         * Case 2: Yaw Rate is greater than 0 (eps)
         *   x_final = x0 + (v/theta_dot)(sin(theta + dt*theta_dot) - sing(theta))
         *   y_final = y0 + (v/theta_dot)(cos(theta) - cos(theta+ dt*theta_dot))
         *   theta_final = theta+ dt* theta_dot
         */
        if (fabs(yaw_rate) < .00001) {
            particles[i].x += velocity * delta_t * cos(particles[i].theta) + dist_x(gen);
            particles[i].y += velocity * delta_t * sin(particles[i].theta) + dist_y(gen);
            particles[i].theta += dist_theta(gen);
        } else {
            double theta_new = particles[i].theta + delta_t * yaw_rate;
            particles[i].x += (velocity / yaw_rate) * (sin(theta_new) - sin(particles[i].theta)) + dist_x(gen);
            particles[i].y += (velocity / yaw_rate) * (cos(particles[i].theta) - cos(theta_new)) + dist_y(gen);
            particles[i].theta = theta_new + dist_theta(gen);
        }
    }

}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted,
                                     vector<LandmarkObs> &observations) {
    /**
     * Find the predicted measurement that is closest to each
     *   observed measurement and assign the observed measurement to this
     *   particular landmark.
     * NOTE: this method will NOT be called by the grading code. But you will
     *   probably find it useful to implement this method and use it as a helper
     *   during the updateWeights phase.
     */

    for (int i = 0; i<observations.size();i++) {
        auto observation = observations[i];
        double min_dist = std::numeric_limits<double>::max();
        int closest_landmark_id = -1;
        for (auto landmark: predicted) {
            double distance = dist(observation.x, observation.y, landmark.x, landmark.y);
            if (distance < min_dist) {
                min_dist = distance;
                closest_landmark_id = landmark.id;
            }
        }
        observations[i].id = closest_landmark_id;
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const vector<LandmarkObs> &observations,
                                   const Map &map_landmarks) {
    /**
     * Update the weights of each particle using a mult-variate Gaussian
     *   distribution. You can read more about this distribution here:
     *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
     * NOTE: The observations are given in the VEHICLE'S coordinate system.
     *   Your particles are located according to the MAP'S coordinate system.
     *   You will need to transform between the two systems. Keep in mind that
     *   this transformation requires both rotation AND translation (but no scaling).
     *   The following is a good resource for the theory:
     *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
     *   and the following is a good resource for the actual equation to implement
     *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
     */

    for (int i = 0; i < particles.size(); i++) {
        auto particle = particles[i];
        // Transform to Map Coordinates
        std::vector<LandmarkObs> transformed_observations;
        for (const auto obs: observations) {
            double x_p = particle.x;
            double y_p = particle.y;
            double theta = particle.theta;
            double x_c = obs.x;
            double y_c = obs.y;

            double x_m = x_p + (cos(theta) * x_c) - (sin(theta) * y_c);
            double y_m = y_p + (sin(theta) * x_c) + (cos(theta) * y_c);

            transformed_observations.push_back(LandmarkObs{obs.id, x_m, y_m});
        }

        // Find landmarks in sensor range
        std::vector<LandmarkObs> predicted;
        for (auto landmark : map_landmarks.landmark_list) {
            if (dist(particle.x, particle.y, landmark.x_f, landmark.y_f) <= sensor_range) {
                predicted.push_back(LandmarkObs{landmark.id_i, landmark.x_f, landmark.y_f});
            }
        }

        dataAssociation(predicted, transformed_observations);

        //Update Particle Weights
        particle.weight = 1.0; //Initialize
        for (auto obs: transformed_observations) {
            LandmarkObs neighbor = LandmarkObs{map_landmarks.landmark_list[obs.id - 1].id_i,
                                               map_landmarks.landmark_list[obs.id - 1].x_f,
                                               map_landmarks.landmark_list[obs.id - 1].y_f};
            // Compute probability of observation given closest landmark
            double sig_x = std_landmark[0];
            double sig_y = std_landmark[1];
            double mu_x = neighbor.x;
            double mu_y = neighbor.y;
            double x_obs = obs.x;
            double y_obs = obs.y;

            double gauss_norm;
            gauss_norm = 1 / (2 * M_PI * sig_x * sig_y);

            // calculate exponent
            double exponent;
            exponent = (pow(x_obs - mu_x, 2) / (2 * pow(sig_x, 2)))
                       + (pow(y_obs - mu_y, 2) / (2 * pow(sig_y, 2)));

            // calculate weight using normalization terms and exponent
            double weight;
            weight = gauss_norm * exp(-exponent);

            particle.weight *= weight;
        }
        weights[i] = particle.weight;

    }

}

void ParticleFilter::resample() {
    /**
     * Resample particles with replacement with probability proportional
     *   to their weight.
     * NOTE: You may find std::discrete_distribution helpful here.
     *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
     */
    std::default_random_engine gen;
    std::discrete_distribution<int> gendist(weights.begin(), weights.end());

    std::vector<Particle> new_particles(particles.size());
    for (int i = 0; i < particles.size(); i++) {
        new_particles[i] = std::move(particles[gendist(gen)]);
    }
    particles = std::move(new_particles);

}

void ParticleFilter::SetAssociations(Particle &particle,
                                     const vector<int> &associations,
                                     const vector<double> &sense_x,
                                     const vector<double> &sense_y) {
    // particle: the particle to which assign each listed association,
    //   and association's (x,y) world coordinates mapping
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates
    particle.associations = associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
    vector<int> v = best.associations;
    std::stringstream ss;
    copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length() - 1);  // get rid of the trailing space
    return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
    vector<double> v;

    if (coord == "X") {
        v = best.sense_x;
    } else {
        v = best.sense_y;
    }

    std::stringstream ss;
    copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length() - 1);  // get rid of the trailing space
    return s;
}