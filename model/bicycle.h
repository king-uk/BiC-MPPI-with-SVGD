#pragma once

#include "model_base.h"
#include <cmath>

class Bicycle : public ModelBase {
public:
    Bicycle(double lf_in = 0.17, double lr_in = 0.13,   // 예시: F1TENTH급 wheelbase ~0.3 m
            double v_min = 0.6,  double v_max = 1.5,    // m/s (트랙/튜닝에 맞춰 조정), 
            double d_max = 0.35);                        // rad ≈ 20° (하드웨어에 맞게 조정)
    ~Bicycle() = default;

private:
    double lf, lr, L;
    double vmin, vmax;
    double deltamax;
};

// inline double clamp(double v, double lo, double hi) {
//     return std::max(lo, std::min(v, hi));
// }

inline Bicycle::Bicycle(double lf_in, double lr_in, double v_min, double v_max, double d_max)
    : lf(lf_in), lr(lr_in), L(lf_in + lr_in),
      vmin(v_min), vmax(v_max), deltamax(d_max)
{
    // 상태/입력 차원
    dim_x = 3;
    dim_u = 2;

    // --- Continuous-time dynamics f(x,u) ---
    f = [this](const Eigen::VectorXd& x, const Eigen::VectorXd& u) -> Eigen::VectorXd {
        // x = [x, y, psi], u = [v, delta]
        Eigen::VectorXd xdot(3);

        const double v     = u(0);
        const double delta = u(1);

        // 수치안정: delta는 하드웨어/기구 한계 내로 가정
        // const double d = clamp(delta, -deltamax, deltamax);   // clamp once more

        const double beta = std::atan( (lr / L) * std::tan(delta) );

        const double spb = std::sin(x(2) + beta);
        const double cpb = std::cos(x(2) + beta);

        xdot(0) = v * cpb;
        xdot(1) = v * spb;
        xdot(2) = (v / L) * std::cos(beta) * std::tan(delta);
        return xdot;
    };

    // --- Stage cost q(x,u): 예시 (필요시 가중치 행렬화) ---
    q = [](const Eigen::VectorXd& /*x*/, const Eigen::VectorXd& u) -> double {
        return u.squaredNorm();
    };

    // --- Terminal cost p(x, x_target): 예시 ---
    p = [](const Eigen::VectorXd& x, const Eigen::VectorXd& x_target) -> double {
        return (x - x_target).squaredNorm();
    };

    // 행별 클램프: v in [vmin, vmax], delta in [-deltamax, deltamax]
    h = [this](Eigen::Ref<Eigen::MatrixXd> U) -> void {
        // 속도
        U.row(0) = U.row(0).cwiseMax(vmin).cwiseMin(vmax);
        // 조향
        U.row(1) = U.row(1).cwiseMax(-deltamax).cwiseMin(deltamax);
    };
}

