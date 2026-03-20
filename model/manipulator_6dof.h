#pragma once

#include "model_base.h"
#include <cmath>
#include <vector>
#include <iostream>

class Manipulator6DOF : public ModelBase {
public:
    Manipulator6DOF(double v_max = 2.0);
    ~Manipulator6DOF() = default;

    Eigen::Vector3d getEndEffectorPosition(const Eigen::VectorXd& q) const;

    std::vector<double> d = {0.333, 0.0, 0.316, 0.0, 0.384, 0.0};
    std::vector<double> a = {0.0, 0.0, 0.0, 0.0825, -0.0825, 0.0};
    std::vector<double> alpha = {0.0, -M_PI_2, M_PI_2, M_PI_2, -M_PI_2, M_PI_2};

private:
    double vmax;
    
};



inline Manipulator6DOF::Manipulator6DOF(double v_max) : vmax(v_max) {
    // 상태 공간 및 제어 입력 차원 6으로 변경
    dim_x = 6;
    dim_u = 6;

    f = [](const Eigen::VectorXd& x, const Eigen::VectorXd& u) -> Eigen::VectorXd {
        return u; 
    };

    q = [](const Eigen::VectorXd& /*x*/, const Eigen::VectorXd& u) -> double {
        return u.squaredNorm(); 
    };

    p = [this](const Eigen::VectorXd& x, const Eigen::VectorXd& ref_q) -> double {
        Eigen::Vector3d current_xyz = this->getEndEffectorPosition(x);
        Eigen::Vector3d target_xyz = this->getEndEffectorPosition(ref_q);
        
        double task_cost = (current_xyz - target_xyz).squaredNorm(); 
        double joint_cost = (x - ref_q).squaredNorm();               
        
        return 10.0 * task_cost + 0.1 * joint_cost; 
    };

    h = [this](Eigen::Ref<Eigen::MatrixXd> U) -> void {
        for(int i = 0; i < dim_u; ++i) {
            U.row(i) = U.row(i).cwiseMax(-vmax).cwiseMin(vmax);
        }
    };    
}

inline Eigen::Vector3d Manipulator6DOF::getEndEffectorPosition(const Eigen::VectorXd& q) const {
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    
    // 루프 상한선을 6으로 변경
    for (int i = 0; i < 6; ++i) {
        Eigen::Matrix4d A;
        double ct = std::cos(q(i));
        double st = std::sin(q(i));
        double ca = std::cos(alpha[i]);
        double sa = std::sin(alpha[i]);

        A << ct, -st*ca,  st*sa, a[i]*ct,
             st,  ct*ca, -ct*sa, a[i]*st,
             0,   sa,     ca,    d[i],
             0,   0,      0,     1;
             
        T = T * A;
    }
    return T.block<3, 1>(0, 3);
}