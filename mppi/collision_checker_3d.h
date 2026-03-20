#pragma once

#include <eigen3/Eigen/Dense>
#include <vector>
#include <array>
#include <cmath>
#include <iostream>

class CollisionChecker3D {
public:
    CollisionChecker3D();
    ~CollisionChecker3D() = default;

    // 3D 환경 장애물 추가
    void addSphere(double x, double y, double z, double r);
    void addBox(double xmin, double xmax, double ymin, double ymax, double zmin, double zmax);

    // MPPI에서 호출할 메인 충돌 검사 함수 (입력: 7개의 조인트 각도)
    bool getCollision(const Eigen::VectorXd &q);

    // 로봇 팔을 감싸는 가상의 충돌 구(Sphere) 반지름 (m)
    double robot_radius;
    // 하나의 링크(관절과 관절 사이)를 몇 개의 구로 쪼개어 검사할 것인지
    int num_interp;

private:
    // 장애물 저장소
    // spheres: [x, y, z, radius, radius^2] (연산 속도를 위해 r^2 미리 저장)
    std::vector<std::array<double, 5>> spheres; 
    // boxes: [xmin, xmax, ymin, ymax, zmin, zmax]
    std::vector<std::array<double, 6>> boxes;

    // FK 계산용 DH 파라미터 (조인트 위치 계산용)
    std::vector<double> d;
    std::vector<double> a;
    std::vector<double> alpha;

    // 특정 3D 좌표(pt)가 장애물과 겹치는지 확인하는 내부 함수
    bool isPointColliding(const Eigen::Vector3d& pt);
};

inline CollisionChecker3D::CollisionChecker3D() {
    robot_radius = 0.05; // 로봇 팔 두께를 5cm로 가정
    num_interp = 4;      // 관절 사이에 4개의 포인트를 추가로 검사

    spheres.clear();
    boxes.clear();

    // ModelBase에 정의했던 DH 파라미터와 동일하게 설정
    d = {0.333, 0.0, 0.316, 0.0, 0.384, 0.0, 0.107};
    a = {0.0, 0.0, 0.0, 0.0825, -0.0825, 0.0, 0.088};
    alpha = {0.0, -M_PI_2, M_PI_2, M_PI_2, -M_PI_2, M_PI_2, M_PI_2};
}

inline void CollisionChecker3D::addSphere(double x, double y, double z, double r) {
    spheres.push_back({x, y, z, r, r * r});
}

inline void CollisionChecker3D::addBox(double xmin, double xmax, double ymin, double ymax, double zmin, double zmax) {
    boxes.push_back({xmin, xmax, ymin, ymax, zmin, zmax});
}

inline bool CollisionChecker3D::isPointColliding(const Eigen::Vector3d& pt) {
    // 1. 구형 장애물 충돌 검사
    for (const auto& obs : spheres) {
        double dx = pt(0) - obs[0];
        double dy = pt(1) - obs[1];
        double dz = pt(2) - obs[2];
        double dist_sq = dx*dx + dy*dy + dz*dz;
        
        // (장애물 반지름 + 로봇 팔 두께)의 제곱보다 가까우면 충돌
        double col_dist = obs[3] + robot_radius;
        if (dist_sq <= col_dist * col_dist) {
            return true;
        }
    }

    // 2. 박스형 장애물 충돌 검사 (AABB)
    for (const auto& box : boxes) {
        if (pt(0) + robot_radius >= box[0] && pt(0) - robot_radius <= box[1] &&
            pt(1) + robot_radius >= box[2] && pt(1) - robot_radius <= box[3] &&
            pt(2) + robot_radius >= box[4] && pt(2) - robot_radius <= box[5]) {
            return true;
        }
    }
    return false;
}

inline bool CollisionChecker3D::getCollision(const Eigen::VectorXd &q) {
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    std::vector<Eigen::Vector3d> joint_positions;
    
    // 로봇의 베이스 좌표 추가
    joint_positions.push_back(T.block<3, 1>(0, 3));

    // 1. Forward Kinematics를 통해 모든 관절의 3D 위치 계산
    for (int i = 0; i < 7; ++i) {
        Eigen::Matrix4d A;
        double ct = std::cos(q(i)), st = std::sin(q(i));
        double ca = std::cos(alpha[i]), sa = std::sin(alpha[i]);

        A << ct, -st*ca,  st*sa, a[i]*ct,
             st,  ct*ca, -ct*sa, a[i]*st,
             0,   sa,     ca,    d[i],
             0,   0,      0,     1;
             
        T = T * A;
        joint_positions.push_back(T.block<3, 1>(0, 3));
    }

    // 2. 링크 보간(Interpolation) 충돌 검사
    // 관절 위치만 검사하면 얇은 장애물이 관절 사이(링크)를 통과할 수 있으므로, 
    // 관절과 관절 사이를 선형 보간하여 여러 점을 검사합니다.
    for (size_t i = 0; i < joint_positions.size() - 1; ++i) {
        Eigen::Vector3d p1 = joint_positions[i];
        Eigen::Vector3d p2 = joint_positions[i+1];

        for (int j = 0; j <= num_interp; ++j) {
            double t = static_cast<double>(j) / num_interp;
            Eigen::Vector3d pt = p1 + t * (p2 - p1);
            
            if (isPointColliding(pt)) {
                return true; // 충돌 발견 즉시 true 반환 (연산 최적화)
            }
        }
    }
    return false; // 충돌 없음
}