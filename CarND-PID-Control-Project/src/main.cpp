#include <math.h>
#include <uWS/uWS.h>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include "json.hpp"
#include "PID.h"

// for convenience
using nlohmann::json;
using std::string;

// For converting back and forth between radians and degrees.
constexpr double pi() { return M_PI; }

double deg2rad(double x) { return x * pi() / 180; }

double rad2deg(double x) { return x * 180 / pi(); }

// Checks if the SocketIO event has JSON data.
// If there is data the JSON object in string format will be returned,
// else the empty string "" will be returned.
string hasData(string s) {
    auto found_null = s.find("null");
    auto b1 = s.find_first_of("[");
    auto b2 = s.find_last_of("]");
    if (found_null != string::npos) {
        return "";
    } else if (b1 != string::npos && b2 != string::npos) {
        return s.substr(b1, b2 - b1 + 1);
    }
    return "";
}

int main(int argc, char *argv[]) {
    uWS::Hub h;

    PID pid = PID();
    /**
     *  Initialize the pid variable.
     */
    bool twiddle = false;
    static bool twiddle_in_progress = false;
    int timestep = 0;
    double tolerance = .001;
    int iteration = 0;
    std::vector<double> p = {0.0905586, 0.000164132, .00449996};
    std::vector<double> dp = {1, 1, 1};
    double err, best_err;
    int param, state;
    bool is_off_track;
    static int TOTAL_TIMESTEPS = 1200;
    static int IGNORE_STEPS = 300;
    double max_speed = 30;
    bool init=false;
    if (argc > 1) {
        twiddle = true;
    }
    /*
     * Iteration 78 complete. Best error : 0.983347
        0.0905586	0.000164232	0.00449996
        0.000487794	0.000146331	0.000399104
     */
    //pid.Init(0.538104, -0.000135559, 1.93032);
    //pid.Init(1.40123, 0.000164132, -0.27793);
    pid.Init(1.4741, 0.000164132, 0.00449996);
    h.onMessage([&](uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length,
                                 uWS::OpCode opCode) {
        is_off_track = false;
        if (length && length > 2 && data[0] == '4' && data[1] == '2') {
            // "42" at the start of the message means there's a websocket message event.
            // The 4 signifies a websocket message
            // The 2 signifies a websocket event
            auto s = hasData(string(data).substr(0, length));
            //std::cout<<timestep<<std::endl;
            if (s != "") {
                auto j = json::parse(s);
                string event = j[0].get<string>();

                if (event == "telemetry") {
                    // j[1] is the data JSON object
                    double cte = std::stod(j[1]["cte"].get<string>());
                    double speed = std::stod(j[1]["speed"].get<string>());
                    double angle = std::stod(j[1]["steering_angle"].get<string>());
                    double steer_value;
                    //std::cout<<speed<<"\t"<<angle<<"\t"<<cte<<std::endl;



                    //https://www.youtube.com/watch?v=2uQ2BSzDvXs
                    if (twiddle && !twiddle_in_progress) {
                        twiddle_in_progress = true;
                        //pid.Init(0.0905586, 0.000164132, .00449996);
                        //pid.Init(0.538104, -0.000135559, 1.93032);
                        //pid.Init(0.532145, 0.000164132, 9.62144);
                        //pid.Init(0.819559, 0.000164132, 0.783055);
                        //pid.Init(1.40123, 0.000164132, -0.27793);
                        pid.Init(1.4741, 0.000164132, 0.00449996);
                        best_err = std::numeric_limits<double>::max();
                        param = 0;
                        state = 0;
                        std::cout << "Starting the Twiddle process to optimize parameters" << std::endl;
                    }

                    if(twiddle && timestep>=IGNORE_STEPS){ //&& timestep>=TOTAL_TIMESTEPS/2
                        err+=pow(cte,2);
                    }

                    //if(twiddle && timestep>10 && speed < 4 && fabs(cte)>5){
                    if((twiddle && timestep>10 && fabs(cte)>3) || (twiddle && timestep>40 && speed<0.2)){
                        std::cout<<"Car is off track or stuck!!!"<<"Timestep : "<<timestep<<std::endl;
                        err+=100000;
                        is_off_track = true;
                        //timestep = TOTAL_TIMESTEPS; //end this experiment
                    }

                    if(twiddle && std::accumulate(dp.begin(),dp.end(),(double)0.0)<tolerance){
                        std::cout<<std::accumulate(dp.begin(),dp.end(),0)<<std::endl;
                        for(auto elem:dp){
                            std::cout<<elem<<"\t";
                        }
                        std::cout<<std::endl;
                        std::cout<<"Twiddle complete !!! Final set of parameter values are : "<<std::endl;
                        twiddle_in_progress = false;
                        pid.DisplayParams();
                        ws.close();
                    }

                    if(twiddle && iteration>0 && timestep==0 && state==0){
                        if(param==0) {
                            std::cout << "Iteration " << iteration << " Starting. sum(dp) : "
                                      << std::accumulate(dp.begin(), dp.end(), 0.0) << std::endl;
                        }
                        p[param]+=dp[param];
                        pid.Init(p[0],p[1],p[2]);
                    }

                    if((twiddle && timestep==TOTAL_TIMESTEPS) || is_off_track){
                        std::cout<<"Param : "<<param<<" state :"<<state<<std::endl;
                        if(is_off_track){
                           err = (timestep>(IGNORE_STEPS)) ? err/(timestep-IGNORE_STEPS) : 10000000;
                           timestep = TOTAL_TIMESTEPS; //End this experiment
                        }
                        else {
                            err /= (TOTAL_TIMESTEPS -IGNORE_STEPS);
                        }
                        if(iteration==0)
                        {
                            best_err = err;
                            iteration++;
                            err = 0;
                        }
                        else {
                            if (state == 0) {
                                std::cout<<"State 0 error : "<< err <<std::endl;
                                if (err < best_err) {
                                    best_err = err;
                                    dp[param] *= 1.1;
                                    param = (param + 1) % 3;
                                } else {
                                    state++;
                                    p[param] -= 2 * dp[param];
                                }
                            } else {
                                std::cout<<"State 1 error : "<< err <<std::endl;
                                if (err < best_err) {
                                    best_err = err;
                                    dp[param] *= 1.1;
                                } else {
                                    p[param] += dp[param];
                                    dp[param] *= 0.9;
                                }
                                param = (param + 1) % 3;
                                state = 0;
                            }
                            if (param == 0) {
                                std::cout << "Iteration " << iteration << " complete. Best error : " << best_err
                                          << std::endl;
                                iteration++;
                                pid.DisplayParams();
                                for(auto elem: dp){
                                    std::cout<<elem<<"\t";
                                }
                                std::cout<<std::endl;
                            }
                            err = 0;
                        }
                    }

                    /**
                     * Calculate steering value here, remember the steering value is
                     *   [-1, 1].
                     * NOTE: Feel free to play around with the throttle and speed.
                     *   Maybe use another PID controller to control the speed!
                     */
                    pid.UpdateError(cte);
                    steer_value = pid.TotalError();
                    double desired_speed = std::max(.01,(1-fabs(steer_value))) * max_speed;
                    //std::cout<<"Desired speed : "<< desired_speed<<std::endl;
                    double norm_speed_error = (desired_speed - speed)/max_speed;
                    //std::cout<<"Norm speed factor: "<< norm_speed_error<<std::endl;
                    json msgJson;
                    msgJson["steering_angle"] = steer_value;
                    msgJson["throttle"] = fabs(norm_speed_error)*.3;
                    auto msg = "42[\"steer\"," + msgJson.dump() + "]";
                    if(!twiddle) {
                        std::cout << msg << std::endl;
                    }
                    ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
                    timestep++;


                    if(twiddle && timestep>TOTAL_TIMESTEPS){
                        timestep = 0;
                        
                        //Reset scene
                        //https://knowledge.udacity.com/questions/6171
                        string resetmsg = "42[\"reset\",{}]";
                        //std::cout << "Resetting scene" << std::endl;
                        ws.send(resetmsg.data(), resetmsg.length(), uWS::OpCode::TEXT);
                    }
                }  // end "telemetry" if
            } else {
                // Manual driving
                string msg = "42[\"manual\",{}]";
                ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
            }
        }  // end websocket message if
    }); // end h.onMessage

    h.onConnection([&](uWS::WebSocket<uWS::SERVER> ws, uWS::HttpRequest req) {
        if(iteration==0) {
            std::cout << "Connected!!!" << std::endl;
            if(!init) {
                init = true;
                string resetmsg = "42[\"reset\",{}]";
                ws.send(resetmsg.data(), resetmsg.length(), uWS::OpCode::TEXT);
            }

        }
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