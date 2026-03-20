#include <bicycle.h>
#include <wmrobot_map.h>
#include <bi_mppi.h>

#include <iostream>
#include <Eigen/Dense>
#include <chrono>

int main() {
    // auto model = WMRobotMap();
    auto model = Bicycle(0.17, 0.13, 0.0, 3.0, 0.35);

    using Solver = BiMPPI;
    using SolverParam = BiMPPIParam;
    
    SolverParam param;
    param.dt = 0.1;
    param.Tf = 50;
    param.Tb = 50;
    param.x_init.resize(model.dim_x);
    param.x_init << 1.5, 0.0, M_PI_2;
    param.x_target.resize(model.dim_x);
    param.x_target << 1.5, 5.0, M_PI_2;

    param.Nf = 6000;
    param.Nb = 6000;

    param.Nr = 3000;

    // param.Ns=20;
    // param.istep = 10;
    param.gamma_u = 10.0;
    Eigen::VectorXd sigma_u(model.dim_u);
    sigma_u << 0.25, 0.25;
    param.sigma_u = sigma_u.asDiagonal();
    param.deviation_mu = 1.0;
    param.cost_mu = 1.0;
    param.minpts = 5;
    param.epsilon = 0.01;
    param.psi = 0.6;

    int maxiter = 269;


    int map = 1; // Start from the last maps
    std::array<double,3> start_x = {0.5, 1.5, 2.5};


    for (int s =0; s<3;++s){

        param.x_init(0) = start_x[s];

        CollisionChecker collision_checker = CollisionChecker();
        collision_checker.loadMap("../BARN_dataset/txt_files/output_"+std::to_string(map)+".txt", 0.1);
        Solver solver(model);
        solver.U_f0 = Eigen::MatrixXd::Zero(model.dim_u, param.Tf);
        solver.U_b0 = Eigen::MatrixXd::Zero(model.dim_u, param.Tb);
        solver.init(param);
        solver.setCollisionChecker(&collision_checker);
        
        bool is_success = false;
        bool is_collision = false;
        int i = 0;
        double total_elapsed  = 0.0;
        double f_err = 0.0;

        double L;

        for (i = 0; i < maxiter; ++i) {
            solver.solve();
            
            solver.move();
            total_elapsed += solver.elapsed;

            
            if (collision_checker.getCollisionGrid(solver.x_init)) {
                is_collision = true;
                break;
            }
            else {
                f_err = (solver.x_init - param.x_target).norm();
                if (f_err < 0.1) {
                    is_success = true;
                    break;
                }
            }
        }

        L= solver.pathLengthXY();

        std::cout  << map << '\t'
            << is_success << '\t' 
            << i << '\t'
            << total_elapsed << '\t'
            << L << '\n';


        solver.showTraj();

        std::string fname = "/home/uk/MPPI/BiC-MPPI/paths/path_" + std::to_string(map) + "_" + std::to_string(start_x[s]) + "BiC_MPPI.csv";
        solver.savePathToCSV(fname); // 현재 시그니처(std::string& , vector& )에도 바인딩 가능

    }

    return 0;    
}


