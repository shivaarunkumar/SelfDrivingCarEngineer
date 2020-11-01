//
// Created by Admin on 11/1/2020.
//

#include "cost.h"
#include <cmath>
#include <stdlib.h>
double goal_distance_cost(int goal_lane, int intended_lane, int final_lane,
                          double distance_to_goal) {
    // The cost increases with both the distance of intended lane from the goal
    //   and the distance of the final lane from the goal. The cost of being out
    //   of the goal lane also becomes larger as the vehicle approaches the goal.

    /**
     * Replace cost = 0 with an appropriate cost function.
     */
    double cost = 0;

    double delta_d = abs((intended_lane-goal_lane)) + abs(final_lane-goal_lane);

    double delta_s = distance_to_goal;

    cost = 1 - exp(-1*delta_d/delta_s);

    return cost;
}