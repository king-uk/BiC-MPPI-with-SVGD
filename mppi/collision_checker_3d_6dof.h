#include "manipulator_6dof.h"
#include "collision_checker_3d.h"
#include "bi_mppi_6dof.h" // 이름만 6dof로 변경한 코어 헤더 포함

#include <iostream>
#include <Eigen/Dense>
#include <chrono>

int main() {
    Manipulator6DOF model(2.0);
    
    BiMPPIParam param;
    param.dt = 0.05; 
    param.Tf = 40;   
    param.Tb = 40;   
    
    // 크기를 6으로 맞추고 6개의 값만 할당
    param.x_init.resize(model.dim_x);
    param.x_init << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    
    param.x_target.resize(model.dim_x);
    param.x_target << M_PI_4, M_PI_4, -M_PI_4, -M_PI_4, 0.0, M_PI_2;

    param.Nf = 2000; 
    param.Nb = 2000;
    param.Nr = 1000;
    
    param.gamma_u = 15.0;
    
    // 크기를 6으로 맞추고 6개의 노이즈 분산 할당
    Eigen::VectorXd sig(model.dim_u);
    sig << 0.5, 0.5, 0.5, 0.5, 0.5, 0.5; 
    param.sigma_u = sig.asDiagonal();
    
    param.deviation_mu = 1.0;
    param.cost_mu = 1.0;
    param.minpts = 5;
    param.epsilon = 0.1;
    param.psi = 0.6;

    CollisionChecker3D collision_checker;
    
    collision_checker.addBox(0.2, 0.3, -0.2, 0.5, 0.0, 0.4);
    collision_checker.addBox(-0.1, 0.1, 0.4, 0.5, 0.0, 0.8);
    collision_checker.addSphere(0.4, 0.2, 0.6, 0.12);
    
    BiMPPI<Manipulator6DOF> solver(model);
    solver.U_f0 = Eigen::MatrixXd::Zero(model.dim_u, param.Tf);
    solver.U_b0 = Eigen::MatrixXd::Zero(model.dim_u, param.Tb);
    solver.init(param);
    solver.setCollisionChecker(&collision_checker);
    
    int maxiter = 50; 
    bool is_success = false;
    double total_elapsed = 0.0;

    std::cout << "Starting 6-DOF Bi-MPPI Offline Path Planning...\n";

    for (int i = 0; i < maxiter; ++i) {
        solver.solve();
        
        total_elapsed += solver.elapsed;

        int final_idx = solver.Xo.cols() - 1;
        Eigen::VectorXd x_end = solver.Xo.col(final_idx);
        double f_err = (x_end - param.x_target).norm();
        
        std::cout << "Iter: " << i 
                  << " | Path End Error to Target: " << f_err 
                  << " | Solve Time: " << solver.elapsed << "s\n";

        if (f_err < 0.05) {
            is_success = true;
            std::cout << "\n[SUCCESS] Found optimal full trajectory to target!\n";
            std::cout << "Total iterations: " << i << "\n";
            std::cout << "Total planning time: " << total_elapsed << "s\n";
            break;
        }
    }

    if (!is_success) {
        std::cout << "\n[FAILED] Could not find a path that reaches the target within max iterations.\n";
    }

    solver.visual_traj.clear();
    for(int j = 0; j < solver.Xo.cols(); ++j) {
        solver.visual_traj.push_back(solver.Xo.col(j));
    }

    std::string fname = "6dof_mppi_trajectory.csv";
    solver.savePathToCSV(fname);
    std::cout << "Planned trajectory saved to " << fname << "\n";

    return 0;    
}