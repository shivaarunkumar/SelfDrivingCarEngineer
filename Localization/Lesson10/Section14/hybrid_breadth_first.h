//
// Created by Admin on 11/1/2020.
//

#ifndef BEHAVIOR_PLANNER_HYBRID_BREADTH_FIRST_H
#define BEHAVIOR_PLANNER_HYBRID_BREADTH_FIRST_H
#include <vector>

using std::vector;

class HBF {
public:
    // Constructor
    HBF();

    // Destructor
    virtual ~HBF();

    // HBF structs
    struct maze_s {
        int g;  // iteration
        double f; // heuristic - euclidean distance to goal
        double x;
        double y;
        double theta;
    };

    struct maze_path {
        vector<vector<vector<int>>> closed;
        vector<vector<vector<maze_s>>> came_from;
        maze_s final;
    };

    // HBF functions
    int theta_to_stack_number(double theta);

    int idx(double float_num);

    vector<maze_s> expand(maze_s &state, vector<int> &goal);

    vector<maze_s> reconstruct_path(vector<vector<vector<maze_s>>> &came_from,
                                    vector<double> &start, HBF::maze_s &final);

    maze_path search(vector<vector<int>> &grid, vector<double> &start,
                     vector<int> &goal);

    double compute_distance(double x1, double y1, double x2, double y2);
private:
    const int NUM_THETA_CELLS = 90;
    const double SPEED = 1.45;
    const double LENGTH = 0.5;
};
#endif //BEHAVIOR_PLANNER_HYBRID_BREADTH_FIRST_H
