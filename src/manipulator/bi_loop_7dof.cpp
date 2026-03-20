#include "manipulator_7dof.h"
#include "collision_checker_3d_7dof.h"
#include "bi_mppi_7dof.h" // 방금 위에서 만든 7dof용 헤더

#include <iostream>
#include <Eigen/Dense>
#include <chrono>

int main() {
    // 1. 7자유도 매니퓰레이터 모델 생성 (최대 조인트 속도 2.0 rad/s)
    Manipulator7DOF model(2.0);
    
    // 2. MPPI 파라미터 세팅
    BiMPPIParam param;
    param.dt = 0.05; // 로봇 제어 주기는 2D보다 짧은 50ms로 설정 (정밀도 상승)
    param.Tf = 40;   // Forward Horizon
    param.Tb = 40;   // Backward Horizon
    
    // 초기 자세 (q1 ~ q7): 로봇이 모두 펴진 상태라고 가정
    param.x_init.resize(model.dim_x);
    param.x_init << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    
    // 목표 자세: 일부 조인트를 크게 회전시킨 자세
    param.x_target.resize(model.dim_x);
    param.x_target << M_PI_4, M_PI_4, -M_PI_4, -M_PI_4, 0.0, M_PI_2, -M_PI_2;

    // 샘플링 개수
    param.Nf = 2000; // 7자유도는 공간이 넓으므로 샘플 수를 넉넉히
    param.Nb = 2000;
    param.Nr = 1000;
    
    // Cost 및 노이즈 파라미터
    param.gamma_u = 15.0;
    
    // 각 조인트(7개)별 노이즈 분산 세팅 (대각 행렬)
    Eigen::VectorXd sig(model.dim_u);
    sig << 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5; // 모든 조인트에 0.5씩 부여
    param.sigma_u = sig.asDiagonal();
    
    param.deviation_mu = 1.0;
    param.cost_mu = 1.0;
    param.minpts = 5;
    param.epsilon = 0.1;
    param.psi = 0.6;

    // 3. 3D 환경(장애물) 세팅 - 난이도 대폭 상승!
    CollisionChecker3D collision_checker;
    
    // 장애물 1 [Box]: 로봇 정면을 가로막는 '낮은 담벼락'
    // 로봇은 이 벽을 부수지 않고 팔을 위로 넘기거나 옆으로 돌려서 가야 합니다.
    // addBox(xmin, xmax, ymin, ymax, zmin, zmax)
    collision_checker.addBox(0.2, 0.3, -0.2, 0.5, 0.0, 0.4);
    
    // 장애물 2 [Box]: 측면 접근을 좁게 만드는 '수직 기둥'
    // 목표 자세로 가는 최단 경로를 방해하여 팔꿈치(Elbow)를 등 뒤로 빼도록 유도합니다.
    collision_checker.addBox(-0.1, 0.1, 0.4, 0.5, 0.0, 0.8);

    // 장애물 3 [Sphere]: 목표 지점 근처에 떠 있는 '기뢰'
    // End-effector가 목표에 도달하기 직전에 궤적을 한 번 더 비틀게 만듭니다.
    collision_checker.addSphere(0.4, 0.2, 0.6, 0.12);
    
    // 4. MPPI 솔버 초기화
    BiMPPI<Manipulator7DOF> solver(model);
    solver.U_f0 = Eigen::MatrixXd::Zero(model.dim_u, param.Tf);
    solver.U_b0 = Eigen::MatrixXd::Zero(model.dim_u, param.Tb);
    solver.init(param);
    solver.setCollisionChecker(&collision_checker);
    
// 5. 오프라인 경로 계획 (Offline Trajectory Planning) 실행
    // 이동(move)을 하지 않고 동일한 조건에서 최적화만 반복하여 궤적을 다듬습니다.
    int maxiter = 50; // MPC처럼 매 스텝 이동하는 것이 아니므로, 50번 정도만 최적화해도 충분히 수렴합니다.
    bool is_success = false;
    double total_elapsed = 0.0;

    std::cout << "Starting 7-DOF Bi-MPPI Offline Path Planning...\n";

    for (int i = 0; i < maxiter; ++i) {
        solver.solve();
        // solver.move(); // <--- 핵심: move()를 주석 처리하여 로봇을 이동시키지 않습니다.
        
        total_elapsed += solver.elapsed;

        // 플래닝 모드에서는 현재 로봇의 위치(x_init)가 아니라, 
        // '계획된 전체 궤적(Xo)의 맨 마지막 지점'이 목표에 도달했는지를 검사해야 합니다.
        int final_idx = solver.Xo.cols() - 1;
        Eigen::VectorXd x_end = solver.Xo.col(final_idx);
        double f_err = (x_end - param.x_target).norm();
        
        std::cout << "Iter: " << i 
                  << " | Path End Error to Target: " << f_err 
                  << " | Solve Time: " << solver.elapsed << "s\n";

        // 전체 궤적의 끝부분이 목표 지점에 충분히 가까워지면 계획 성공!
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

    // 6. 결과 궤적 추출 및 CSV 저장
    // 기존에는 move()를 할 때마다 x_init을 visual_traj에 누적했지만,
    // 이제는 플래닝이 끝난 후 완성된 '최적 궤적 행렬(Xo)' 전체를 visual_traj로 덮어씌워 줍니다.
    solver.visual_traj.clear();
    for(int j = 0; j < solver.Xo.cols(); ++j) {
        solver.visual_traj.push_back(solver.Xo.col(j));
    }

    std::string fname = "7dof_mppi_trajectory.csv";
    solver.savePathToCSV(fname);
    std::cout << "Planned trajectory saved to " << fname << "\n";

    return 0;    
}