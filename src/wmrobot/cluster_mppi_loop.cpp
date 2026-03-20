#include <wmrobot_map.h>
#include <cluster_mppi.h>

#include <iostream>
#include <Eigen/Dense>
#include <chrono>

#include "csv.h"

int main() {
    auto model = WMRobotMap();
    
    using Solver = ClusterMPPI;
    using SolverParam = MPPIParam;

    SolverParam param;
    param.dt = 0.1;
    param.T = 100;
    param.x_init.resize(model.dim_x);
    param.x_init << 2.5, 0.0, M_PI_2;
    param.x_target.resize(model.dim_x);
    param.x_target << 1.5, 5.0, M_PI_2;
    param.N = 12000;
    param.gamma_u = 10.0;
    Eigen::VectorXd sigma_u(model.dim_u);
    sigma_u << 0.6, 0.6;
    param.sigma_u = sigma_u.asDiagonal();

    int maxiter = 200;

    std::array<double,3> start_x = {0.5, 1.5, 2.5};

    std::string result_csv_path = "../CLUSTER_MPPI_loop.csv";
    std::ofstream result_fout = open_result_to_csv(result_csv_path);
    
    // for (int map = 299; map >= 0 ; --map) {
    // for (int map = 276; map >= 0; --map) {
    // for (int s = 1; s < 2; ++s) {
    // for (int s = 0; s < 2; ++s) {
        int s=0;
        for (int map = 299; map >= 200; --map) {
            param.x_init(0) = start_x[s];

        // for (int map = 0; map < 300; ++map) {
            CollisionChecker collision_checker = CollisionChecker();
            collision_checker.loadMap("../BARN_dataset/txt_files/output_"+std::to_string(map)+".txt", 0.1);
            Solver solver(model);
            solver.U_0 = Eigen::MatrixXd::Zero(model.dim_u, param.T);
            solver.init(param);
            solver.setCollisionChecker(&collision_checker);
            
            bool is_success = false;
            bool is_collision = false;
            int i = 0;
            double total_elapsed = 0.0;
            double f_err = 0.0;
            for (i = 0; i < maxiter; ++i) {

                solver.solve();
                solver.move();
                
                // std::cout<<"1 solved in "<<solver.elapsed_1.count()<<std::endl;
                // std::cout<<"2 solved in "<<solver.elapsed_2.count()<<std::endl;
                // std::cout<<"3 solved in "<<solver.elapsed_3.count()<<std::endl;

                total_elapsed += solver.elapsed;

                if (collision_checker.getCollisionGrid(solver.x_init)) {
                    is_collision = true;
                    break;
                }
                else {
                    f_err = (solver.x_init - param.x_target).norm();
                    // std::cout<<"f_err = "<<f_err<<std::endl;
                    if (f_err < 0.1) {
                        is_success = true;
                        break;
                    }
                }
                // solver.show();
            }
            std::cout  << map << '\t'
                << is_success << '\t' 
                << i << '\t'
                << total_elapsed << '\n';
                // << L << '\n';

            std::string fname = "../paths/CLUSTER/path_" + std::to_string(map) + "_" + std::to_string(start_x[s]) + "SVGD_MPPI.csv";
            solver.savePathToCSV(fname); // 현재 시그니처(std::string& , vector& )에도 바인딩 가능

            write_result_to_csv(result_fout,
                        map,
                        is_success,
                        total_elapsed,
                        start_x[s],
                        0);
        }
    // }    
    result_fout.close();
    return 0;
}
