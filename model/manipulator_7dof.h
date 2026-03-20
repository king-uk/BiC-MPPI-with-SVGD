#pragma once

#include "model_base.h"
#include <cmath>
#include <vector>
#include <iostream>

class Manipulator7DOF : public ModelBase {
public:
    // 로봇의 최대 조인트 속도 한계(rad/s)를 파라미터로 받습니다.
    Manipulator7DOF(double v_max = 2.0);
    ~Manipulator7DOF() = default;

    // 순운동학(Forward Kinematics) 
    // 향후 3D 충돌 체크 및 Task-space 비용 계산을 위해 필수적인 함수입니다.
    Eigen::Vector3d getEndEffectorPosition(const Eigen::VectorXd& q) const;

private:
    double vmax;
    
    // Denavit-Hartenberg (DH) 파라미터 (Franka Emika Panda 로봇과 유사한 예시)
    // 실제 사용하실 로봇 스펙에 맞춰 수정이 필요합니다.
    std::vector<double> d;
    std::vector<double> a;
    std::vector<double> alpha;
};

inline Manipulator7DOF::Manipulator7DOF(double v_max) : vmax(v_max) {
    // 상태 공간 차원: 7개의 조인트 각도 (q1 ~ q7)
    dim_x = 7;
    // 제어 입력 차원: 7개의 조인트 각속도 (q_dot1 ~ q_dot7)
    dim_u = 7;

    // --- 1. Continuous-time dynamics f(x,u) ---
    // 운동학 모델이므로 상태 변화율(x_dot)은 곧 제어 입력인 조인트 각속도(u)와 같습니다.
    // 내부의 Euler 적분(x_new = x + f(x,u)*dt)에 의해 위치가 업데이트됩니다.
    f = [](const Eigen::VectorXd& x, const Eigen::VectorXd& u) -> Eigen::VectorXd {
        return u; 
    };

    // --- 2. Stage cost q(x,u) ---
    // 에너지 효율 및 부드러운 움직임을 위해 조인트 속도의 크기를 페널티로 부여합니다.
    q = [](const Eigen::VectorXd& /*x*/, const Eigen::VectorXd& u) -> double {
        return u.squaredNorm(); 
    };

    // --- 3. Terminal cost p(x, x_target) ---
    // 조인트 공간(Joint Space) 기준의 목표 도달 비용입니다.
    // p = [](const Eigen::VectorXd& x, const Eigen::VectorXd& x_target) -> double {
    //     return (x - x_target).squaredNorm();
    // };


    // Task-space 기준 Terminal Cost 예시 아이디어
    p = [this](const Eigen::VectorXd& x, const Eigen::VectorXd& ref_q) -> double {
        Eigen::Vector3d current_xyz = this->getEndEffectorPosition(x);
        Eigen::Vector3d target_xyz = this->getEndEffectorPosition(ref_q);
        
        double task_cost = (current_xyz - target_xyz).squaredNorm(); // XYZ 거리 
        double joint_cost = (x - ref_q).squaredNorm();               // 조인트 각도 차이
        
        // Task 공간에 높은 가중치(예: 10.0), 조인트 안정성에 낮은 가중치(0.1) 부여
        return 10.0 * task_cost + 0.1 * joint_cost; 
    };

    // --- 4. Projection / Control Limits h(U) ---
    // MPPI에서 생성된 랜덤 노이즈가 포함된 제어 입력 행렬 U의 속도 범위를 제한합니다.
    h = [this](Eigen::Ref<Eigen::MatrixXd> U) -> void {
        // 모든 7개의 조인트에 대해 속도를 [-vmax, vmax]로 클램핑
        for(int i = 0; i < dim_u; ++i) {
            U.row(i) = U.row(i).cwiseMax(-vmax).cwiseMin(vmax);
        }
    };

    // 7-DOF DH 파라미터 초기화 (예: 표준 7축 매니퓰레이터)
    // d: link offset, a: link length, alpha: link twist
    d = {0.333, 0.0, 0.316, 0.0, 0.384, 0.0, 0.107};
    a = {0.0, 0.0, 0.0, 0.0825, -0.0825, 0.0, 0.088};
    alpha = {0.0, -M_PI_2, M_PI_2, M_PI_2, -M_PI_2, M_PI_2, M_PI_2};
}

inline Eigen::Vector3d Manipulator7DOF::getEndEffectorPosition(const Eigen::VectorXd& q) const {
    // 동차 변환 행렬(Homogeneous Transformation Matrix) 계산
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    
    for (int i = 0; i < 7; ++i) {
        Eigen::Matrix4d A;
        double ct = std::cos(q(i));
        double st = std::sin(q(i));
        double ca = std::cos(alpha[i]);
        double sa = std::sin(alpha[i]);

        // Standard DH Convention
        A << ct, -st*ca,  st*sa, a[i]*ct,
             st,  ct*ca, -ct*sa, a[i]*st,
             0,   sa,     ca,    d[i],
             0,   0,      0,     1;
             
        T = T * A;
    }
    // 최종 끝단(End-effector)의 X, Y, Z 위치 반환
    return T.block<3, 1>(0, 3);
}