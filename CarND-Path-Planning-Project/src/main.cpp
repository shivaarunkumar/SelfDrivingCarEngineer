#include <uWS/uWS.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include "Eigen-3.3/Eigen/Core"
#include "Eigen-3.3/Eigen/QR"
#include "helpers.h"
#include "json.hpp"
#include "spline.h"

// for convenience
using nlohmann::json;
using std::string;
using std::vector;


/*
 * Requirements
 * ------------
 * The car is able to drive at least 4.32 miles without incident..
 * The car drives according to the speed limit : 50mph or 80kmph
 * The car does not exceed a total acceleration of 10 m/s^2 and a jerk of 10 m/s^3.
 * Car does not have collisions.
 * The car doesn't spend more than a 3 second length out side the lane lanes during changing lanes, and every other time the car stays inside one of the 3 lanes on the right hand side of the road.
 * The car is able to smoothly change lanes when it makes sense to do so, such as when behind a slower moving car and an adjacent lane is clear of other traffic.
 */
int main() {
    uWS::Hub h;

    // Load up map values for waypoint's x,y,s and d normalized normal vectors
    vector<double> map_waypoints_x;
    vector<double> map_waypoints_y;
    vector<double> map_waypoints_s;
    vector<double> map_waypoints_dx;
    vector<double> map_waypoints_dy;

    // Waypoint map to read from
    string map_file_ = "../data/highway_map.csv";
    // The max s value before wrapping around the track back to 0
    double max_s = 6945.554;

    std::ifstream in_map_(map_file_.c_str(), std::ifstream::in);

    string line;
    while (getline(in_map_, line)) {
        std::istringstream iss(line);
        double x;
        double y;
        float s;
        float d_x;
        float d_y;
        iss >> x;
        iss >> y;
        iss >> s;
        iss >> d_x;
        iss >> d_y;
        map_waypoints_x.push_back(x);
        map_waypoints_y.push_back(y);
        map_waypoints_s.push_back(s);
        map_waypoints_dx.push_back(d_x);
        map_waypoints_dy.push_back(d_y);
    }
    double target_velocity = 0.0; // mph
    int lane = 1; // 3 lanes in each direction, tagged 0, 1, 2
    bool too_close = false;
    bool prev_close = false;
    double close_count = 1;


    h.onMessage([&map_waypoints_x, &map_waypoints_y, &map_waypoints_s,
                        &map_waypoints_dx, &map_waypoints_dy, &target_velocity, &lane, &too_close, &prev_close, &close_count]
                        (uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length,
                         uWS::OpCode opCode) {

        // Lane
        int lane_width = 4; // m


        // Velocity
        int PREV_DATAPOINTS_TO_RETAIN = 10;
        double MAX_VELOCITY = 49.5; // mph, actual max is 50 mph (22.35 m/s)
        double max_change_velocity = .224; // To comply with max acceleration of 10m/s^2 , this is equivalent to 5 m/s^2. max_delta_v = 5 * .02 = .1 m/s = .1*2.24 mph = .224mph

        prev_close = too_close;
        if(!prev_close){
            close_count = 1;
        }
        too_close = false;

        /*
         * Ideally we would like to look ahead at least 10 sec. So if one is travelling at the speed limit, we should be
         * looking ahead around 250m. 100 way points would mean we are look at increments of 2.5 m.
         */
        // Trajectory
        int num_waypoints = 50; //

        // "42" at the start of the message means there's a websocket message event.
        // The 4 signifies a websocket message
        // The 2 signifies a websocket event
        if (length && length > 2 && data[0] == '4' && data[1] == '2') {

            auto s = hasData(data);

            if (s != "") {
                auto j = json::parse(s);

                string event = j[0].get<string>();
                if (event == "telemetry") {
                    // j[1] is the data JSON object

                    // Main car's localization Data
                    double car_x = j[1]["x"];
                    double car_y = j[1]["y"];
                    double car_s = j[1]["s"];
                    double car_d = j[1]["d"];
                    double car_yaw = j[1]["yaw"];
                    double car_speed = j[1]["speed"];

                    // Previous path data given to the Planner
                    auto previous_path_x = j[1]["previous_path_x"];
                    auto previous_path_y = j[1]["previous_path_y"];
                    int prev_size = previous_path_x.size(); // Number of points remaining from the previous set of waypoints

                    // Previous path's end s and d values
                    double end_path_s = j[1]["end_path_s"];
                    double end_path_d = j[1]["end_path_d"];

                    // Sensor Fusion Data, a list of all other cars on the same side
                    //   of the road.
                    auto sensor_fusion = j[1]["sensor_fusion"];

                    json msgJson;

                    vector<double> next_x_vals;
                    vector<double> next_y_vals;

                    /**
                     *  define a path made up of (x,y) points that the car will visit
                     *   sequentially every .02 seconds
                     */

                    // Check to see if there is any car in front of ego
                    // There are essentially 12 cars on the track. We ned identify the ones closest to ego.
                    // Do not collide with the one in front.
                    // Do not switch to the left or right lane if there is another car that is close to us.
                    bool car_left = false;
                    bool car_right = false;
                    bool car_ahead = false;


                    for(int i = 0; i<sensor_fusion.size();i++){
                        vector<double> other_car = sensor_fusion[i];
                        double other_car_vx = other_car[3];
                        double other_car_vy = other_car[4];
                        double other_car_s = other_car[5];
                        double other_car_d = other_car[6];
                        int other_car_lane;

                        if(other_car_d<=lane_width){
                            other_car_lane = 0;
                        }
                        else if(other_car_d<=2*lane_width){
                            other_car_lane = 1;
                        }
                        else{
                            other_car_lane = 2;
                        }

                        double other_car_speed = sqrt(other_car_vx * other_car_vx + other_car_vy * other_car_vy);
                        other_car_s += (double)prev_size * .02 * other_car_speed;

                        if((lane == other_car_lane) && (other_car_s > car_s) && (other_car_s < (car_s+50.0))) {
                            car_ahead = true;
                            too_close = true;
                            if(prev_close){
                                close_count+=.1;
                            }
                        }
                        else if (((lane - other_car_lane)==1) && (other_car_s > (car_s-30.0)) && (other_car_s < (car_s+30.0)))
                            car_left = true;
                        else if (((lane - other_car_lane)==-1) && (other_car_s > (car_s-30.0)) && (other_car_s < (car_s+30.0)))
                            car_right = true;

                    }



                    // Look for possibility of lane change
                    if(car_ahead){
                        if((lane>0) && !car_left && (target_velocity > 30.0) && (target_velocity <= 40.0))
                            lane--;
                        else if((lane<2) && !car_right && (target_velocity > 30.0) && (target_velocity <= 40.0))
                            lane++;
                        else{
                            // Decrement velocity
                            target_velocity -= max_change_velocity*close_count; // the closer you are to the car the faster you need to slow down
                            if (target_velocity <= 0) {
                                target_velocity = max_change_velocity;
                            }
                        }
                    }
                    else if (target_velocity < MAX_VELOCITY) {
                        target_velocity += max_change_velocity;
                    }


//                    std::cout<<close_count<<std::endl;
//                    std::cout<<target_velocity<<std::endl;



                    // Construct spline that will be used for interpolation
                    vector<double> ptsx;
                    vector<double> ptsy;
                    double prev_x_1, prev_x_2, prev_y_1, prev_y_2, prev_yaw;
                    prev_size = (PREV_DATAPOINTS_TO_RETAIN <= prev_size) ? PREV_DATAPOINTS_TO_RETAIN : prev_size;
                    // We need atleast two past points and few in the future.
                    if (prev_size >= 2) {
                        // Last but one
                        prev_x_2 = previous_path_x[prev_size - 2];
                        prev_y_2 = previous_path_y[prev_size - 2];
                        // Last one
                        prev_x_1 = previous_path_x[prev_size - 1];
                        prev_y_1 = previous_path_y[prev_size - 1];

                        // Previous Yaw
                        prev_yaw = atan2((prev_y_1-prev_y_2),(prev_x_1-prev_x_2));
                    }
                    else {
                        // We do not have sufficient points left from the previous set of waypoints. So we need to start from scratch using current car location.
                        // Last one
                        prev_x_1 = car_x;
                        prev_y_1 = car_y;
                        prev_yaw = deg2rad(car_yaw);

                        // Last but one
                        // Assuming car travel a distance d = 1 , d * cos(yaw) = delta_x. car_prev = car_x - d*cos(yaw)
                        prev_x_2 = car_x - cos(prev_yaw);
                        prev_y_2 = car_y - sin(prev_yaw);


                    }

                    ptsx.push_back(prev_x_2);
                    ptsy.push_back(prev_y_2);
                    ptsx.push_back(prev_x_1);
                    ptsy.push_back(prev_y_1);

                    // We will be looking 90m
                    // We will divide that into 3 coarse sections and let the spline interpolate the rest.
                    for(int i = 1; i <=3 ; i++){
                        double target_s = car_s + i * 30;
                        double target_d = lane_width/2 + lane_width * lane;
                        vector<double> nextXY = getXY(target_s, target_d, map_waypoints_s, map_waypoints_x, map_waypoints_y);
                        ptsx.push_back(nextXY[0]);
                        ptsy.push_back(nextXY[1]);
                    }

                    // Convert to cars reference
                    for(int i = 0;i < ptsx.size(); i++){
                        vector<double> carXY = getCarXY(ptsx[i],ptsy[i],prev_x_1, prev_y_1, prev_yaw);
                        ptsx[i] = carXY[0];
                        ptsy[i] = carXY[1];
                    }

                    // Initialize the spline to cars reference frame
                    tk::spline s;
                    s.set_points(ptsx, ptsy);

                    // Populate previous way points
                    for(int i = 0; i<prev_size; i++){
                        next_x_vals.push_back(previous_path_x[i]);
                        next_y_vals.push_back(previous_path_y[i]);
                    }

                    int required_waypoints = num_waypoints - prev_size;
                    double lookahead_x_distance = 30; //m
                    double lookahead_y_distance = s(lookahead_x_distance);
                    double d = distance(0,0,lookahead_x_distance,lookahead_y_distance); // remember cars reference frame
                    double N_segments = d/(.02*(target_velocity/2.24));
                    double x_add_on = 0; //start at current


                    for (int i = 1; i <= required_waypoints; ++i) {
                        double new_x = i*lookahead_x_distance/N_segments;
                        double new_y = s(new_x);
                        vector<double> mapXY = getMapXY(new_x, new_y, prev_x_1, prev_y_1, prev_yaw);
                        next_x_vals.push_back(mapXY[0]);
                        next_y_vals.push_back(mapXY[1]);
                    }


                    msgJson["next_x"] = next_x_vals;
                    msgJson["next_y"] = next_y_vals;

                    auto msg = "42[\"control\"," + msgJson.dump() + "]";

                    ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
                }  // end "telemetry" if
            } else {
                // Manual driving
                std::string msg = "42[\"manual\",{}]";
                ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
            }
        }  // end websocket if
    }); // end h.onMessage

    h.onConnection([&h](uWS::WebSocket<uWS::SERVER> ws, uWS::HttpRequest req) {
        std::cout << "Connected!!!" << std::endl;
    });

    h.onDisconnection([&h](uWS::WebSocket<uWS::SERVER> ws, int code,
                           char *message, size_t length) {
        ws.close();
        std::cout << "Disconnected" << std::endl;
    });

    int port = 4567;
    if (h.listen(port)) {
        std::cout << "Listening to port " << port << std::endl;
    } else {
        std::cerr << "Failed to listen to port" << std::endl;
        return -1;
    }

    h.run();
}