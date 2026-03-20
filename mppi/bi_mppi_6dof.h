#pragma once

#include <EigenRand/EigenRand>
#include "mppi_param.h"
#include "collision_checker_3d_6dof.h" // 3D 충돌 체커
#include "model_base.h"

#include <ctime>
#include <vector>
#include <deque>
#include <map>
#include <numeric>
#include <chrono>
#include <iostream>
#include <fstream>
#include <omp.h>
#include <cmath>

template <typename ModelClass>
class BiMPPI {
public:
    BiMPPI(ModelClass model);
    ~BiMPPI() = default;

    void init(BiMPPIParam bi_mppi_param);
    void setCollisionChecker(CollisionChecker3D *collision_checker); 
    
    Eigen::MatrixXd getNoise(const int &T);
    void move();
    void solve();
    void backwardRollout();
    void forwardRollout();

    void selectConnection();
    void concatenate();
    void guideMPPI();
    void partitioningControl();

    void dbscan(std::vector<std::vector<int>> &clusters, const Eigen::MatrixXd &Di, const Eigen::VectorXd &costs, const int &N, const int &T);
    void calculateU(Eigen::MatrixXd &U, const std::vector<std::vector<int>> &clusters, const Eigen::VectorXd &costs, const Eigen::MatrixXd &Ui, const int &T);

    void savePathToCSV(const std::string& filename) const;
    double pathLengthJoint(double eps_ignore = 0.0) const;

    std::chrono::time_point<std::chrono::high_resolution_clock> start, finish;
    std::chrono::duration<double> elapsed_1, elapsed_2, elapsed_3;
    double elapsed;
    std::vector<Eigen::VectorXd> visual_traj;

    Eigen::MatrixXd U_f0, U_b0;
    Eigen::VectorXd x_init, x_target, dummy_u;
    Eigen::MatrixXd Uo, Xo;
    Eigen::VectorXd u0;
    
private:
    int dim_x, dim_u;

    std::function<Eigen::VectorXd(Eigen::VectorXd, Eigen::VectorXd)> f;
    std::function<double(Eigen::VectorXd, Eigen::VectorXd)> q, p;
    std::function<void(Eigen::Ref<Eigen::MatrixXd>)> h;

    std::mt19937_64 urng{static_cast<std::uint_fast64_t>(std::time(nullptr))};
    Eigen::Rand::NormalGen<double> norm_gen{0.0, 1.0};

    float dt;
    int Tf, Tb, Nf, Nb, Nr;
    double gamma_u, deviation_mu, cost_mu, epsilon, psi;
    int minpts;
    Eigen::MatrixXd sigma_u;

    CollisionChecker3D *collision_checker;

    std::vector<std::vector<int>> clusters_f, clusters_b, joints;
    std::vector<int> full_cluster_f, full_cluster_b;
    Eigen::MatrixXd Uf, Xf, Ub, Xb;
    std::vector<Eigen::MatrixXd> Xc, Uc, Ur, Xr;
    std::vector<double> Cr;
};

// =========================================================================
// Implementation Section
// =========================================================================

template<typename ModelClass>
BiMPPI<ModelClass>::BiMPPI(ModelClass model) {
    this->dim_x = model.dim_x;
    this->dim_u = model.dim_u;
    this->f = model.f;
    this->q = model.q;
    this->p = model.p;
    this->h = model.h;
}

template<typename ModelClass>
void BiMPPI<ModelClass>::init(BiMPPIParam param) {
    this->dt = param.dt;
    this->Tf = param.Tf;
    this->Tb = param.Tb;
    this->x_init = param.x_init;
    this->x_target = param.x_target;
    this->Nf = param.Nf;
    this->Nb = param.Nb;
    this->Nr = param.Nr;
    this->gamma_u = param.gamma_u;
    this->sigma_u = param.sigma_u;
    this->deviation_mu = param.deviation_mu;
    this->cost_mu = param.cost_mu;
    this->minpts = param.minpts;
    this->epsilon = param.epsilon;
    
    full_cluster_f.resize(Nf); std::iota(full_cluster_f.begin(), full_cluster_f.end(), 0);
    full_cluster_b.resize(Nb); std::iota(full_cluster_b.begin(), full_cluster_b.end(), 0);

    u0 = Eigen::VectorXd::Zero(dim_u);
    dummy_u = Eigen::VectorXd::Zero(dim_u);
}

template<typename ModelClass>
void BiMPPI<ModelClass>::setCollisionChecker(CollisionChecker3D *cc) {
    this->collision_checker = cc;
}

template<typename ModelClass>
Eigen::MatrixXd BiMPPI<ModelClass>::getNoise(const int &T) {
    return sigma_u * norm_gen.template generate<Eigen::MatrixXd>(dim_u, T, urng);
}

template<typename ModelClass>
void BiMPPI<ModelClass>::move() {
    x_init = x_init + (dt * f(x_init, u0));
    U_f0.leftCols(U_f0.cols() - 1) = U_f0.rightCols(U_f0.cols() - 1);
}

template<typename ModelClass>
void BiMPPI<ModelClass>::solve() {
    omp_set_nested(1);
    
    start = std::chrono::high_resolution_clock::now();
    backwardRollout();
    forwardRollout();
    finish = std::chrono::high_resolution_clock::now();
    elapsed_1 = finish - start;

    start = std::chrono::high_resolution_clock::now();
    selectConnection();
    concatenate();
    finish = std::chrono::high_resolution_clock::now();
    elapsed_2 = finish - start;

    start = std::chrono::high_resolution_clock::now();
    guideMPPI();
    finish = std::chrono::high_resolution_clock::now();
    elapsed_3 = finish - start;

    partitioningControl();

    elapsed = elapsed_1.count() + elapsed_2.count() + elapsed_3.count();
    visual_traj.push_back(x_init);
}

template<typename ModelClass>
void BiMPPI<ModelClass>::backwardRollout() {
    Eigen::MatrixXd Ui = U_b0.replicate(Nb, 1);
    Eigen::MatrixXd Di(dim_u, Nb);
    Eigen::VectorXd costs(Nb);
    bool all_feasible = true;

    #pragma omp parallel for
    for (int i = 0; i < Nb; ++i) {
        Eigen::MatrixXd Xi(dim_x, Tb + 1);
        Eigen::MatrixXd noise = getNoise(Tb);
        Ui.middleRows(i * dim_u, dim_u) += noise;
        h(Ui.middleRows(i * dim_u, dim_u));

        Xi.col(Tb) = x_target;
        double cost = 0.0;
        for (int j = Tb - 1; j >= 0; --j) {
            Xi.col(j) = Xi.col(j+1) - (dt * f(Xi.col(j+1), Ui.block(i * dim_u, j + (j == Tb - 1 ? 0 : 1), dim_u, 1)));
            cost += p(Xi.col(j), x_init);
        }
        cost += p(Xi.col(0), x_init);
        
        for (int j = Tb; j >= 0; --j) {
            if (collision_checker->getCollision(Xi.col(j))) { 
                all_feasible = false;
                cost = 1e8;
                break;
            }
        }
        costs(i) = cost;
        Di.col(i) = (Ui.middleRows(i * dim_u, dim_u) - U_b0).rowwise().mean();
    }

    if (!all_feasible) {dbscan(clusters_b, Di, costs, Nb, Tb);}
    else {clusters_b.clear();}

    if (clusters_b.empty()) {clusters_b.push_back(full_cluster_b);}
    calculateU(Ub, clusters_b, costs, Ui, Tb);

    Xb.resize(clusters_b.size() * dim_x, Tb + 1);
    for (int i = 0; i < clusters_b.size(); ++i) {
        Xb.block(i * dim_x, Tb, dim_x, 1) = x_target;
        for (int t = Tb - 1; t >= 0; --t) {
            Xb.block(i * dim_x, t, dim_x, 1) = Xb.block(i * dim_x, t + 1, dim_x, 1) - (dt * f(Xb.block(i * dim_x, t + 1, dim_x, 1), Ub.block(i * dim_u, t + (t == Tb - 1 ? 0 : 1), dim_u, 1)));
        }
    }
}

template<typename ModelClass>
void BiMPPI<ModelClass>::forwardRollout() {
    Eigen::MatrixXd Ui = U_f0.replicate(Nf, 1);
    Eigen::MatrixXd Di(dim_u, Nf);
    Eigen::VectorXd costs(Nf);
    bool all_feasible = true;

    #pragma omp parallel for
    for (int i = 0; i < Nf; ++i) {
        Eigen::MatrixXd Xi(dim_x, Tf + 1);
        Eigen::MatrixXd noise = getNoise(Tf);
        Ui.middleRows(i * dim_u, dim_u) += noise;
        h(Ui.middleRows(i * dim_u, dim_u));

        Xi.col(0) = x_init;
        double cost = 0.0;
        for (int j = 0; j < Tf; ++j) {
            cost += p(Xi.col(j), x_target);
            Xi.col(j+1) = Xi.col(j) + (dt * f(Xi.col(j), Ui.block(i * dim_u, j, dim_u, 1)));
        }
        cost += p(Xi.col(Tf), x_target);
        
        for (int j = 0; j <= Tf; ++j) {
            if (collision_checker->getCollision(Xi.col(j))) { 
                all_feasible = false;
                cost = 1e8;
                break;
            }
        }
        costs(i) = cost;
        Di.col(i) = (Ui.middleRows(i * dim_u, dim_u) - U_f0).rowwise().mean();
    }

    if (!all_feasible) {dbscan(clusters_f, Di, costs, Nf, Tf);}
    else {clusters_f.clear();}

    if (clusters_f.empty()) {clusters_f.push_back(full_cluster_f);}
    calculateU(Uf, clusters_f, costs, Ui, Tf);

    Xf.resize(clusters_f.size() * dim_x, Tf + 1);
    for (int i = 0; i < clusters_f.size(); ++i) {
        Xf.block(i * dim_x, 0, dim_x, 1) = x_init;
        for (int t = 0; t < Tf; ++t) {
            Xf.block(i * dim_x, t + 1, dim_x, 1) = Xf.block(i * dim_x, t, dim_x, 1) + (dt * f(Xf.block(i * dim_x, t, dim_x, 1), Uf.block(i * dim_u, t, dim_u, 1)));
        }
    }
}

template<typename ModelClass>
void BiMPPI<ModelClass>::selectConnection() {
    joints.clear();
    for (int cf = 0; cf < clusters_f.size(); ++cf) {
        double min_norm = std::numeric_limits<double>::max();
        int cb, df, db;
        for (int cb_ = 0; cb_ < clusters_b.size(); ++cb_) {
            for (int df_ = 0; df_ <= Tf; ++df_) {
                for (int db_ = 0; db_ <= Tb; ++db_) {
                    double norm = (Xf.block(cf * dim_x, df_, dim_x, 1) - Xb.block(cb_ * dim_x, db_, dim_x, 1)).norm();
                    if (norm < min_norm) {
                        min_norm = norm;
                        cb = cb_;
                        df = df_;
                        db = db_;
                    }
                }
            }
        }
        joints.push_back({cf, cb, df, db});
    }
}

template<typename ModelClass>
void BiMPPI<ModelClass>::concatenate() {
    Uc.clear();
    Xc.clear();

    for (std::vector<int> joint : joints) {
        int cf = joint[0];
        int cb = joint[1];
        int df = joint[2];
        int db = joint[3];

        Eigen::MatrixXd U(dim_u, std::max(Tf, df + (Tb - db)));
        Eigen::MatrixXd X(dim_x, std::max(Tf, df + (Tb - db)) + 1);

        if (df == 0) {
            X.leftCols(df+1) = Xf.block(cf * dim_x, 0, dim_x, df+1);
        }
        else {
            U.leftCols(df) =  Uf.block(cf * dim_u, 0, dim_u, df);
            X.leftCols(df+1) = Xf.block(cf * dim_x, 0, dim_x, df+1);
        }

        if (db != Tb) {
            U.middleCols(df+1, Tb - db - 1) = Ub.block(cb * dim_u, db + 1, dim_u, Tb - db - 1);
            X.middleCols(df+2, Tb - db - 1) = Xb.block(cb * dim_x, db + 1, dim_x, Tb - db - 1);
        }

        if (df + (Tb - db) < Tf) {
            U.rightCols(Tf - (df + (Tb - db))).colwise() = dummy_u;
            X.rightCols(Tf - (df + (Tb - db))).colwise() = x_target;
        }

        Uc.push_back(U);
        Xc.push_back(X);
    }
}

template<typename ModelClass>
void BiMPPI<ModelClass>::guideMPPI() {
    Ur.clear();
    Cr.clear();
    Xr.clear();
    
    for (int r = 0; r < joints.size(); ++r) {
        Eigen::MatrixXd Ui = Uc[r].replicate(Nr, 1);
        Eigen::MatrixXd X_ref = Xc[r];
        int Tr = Uc[r].cols();
        Eigen::VectorXd costs(Nr);
        Eigen::VectorXd weights(Nr);

        #pragma omp parallel for
        for (int i = 0; i < Nr; ++i) {
            Eigen::MatrixXd Xi(dim_x, Tr + 1);
            Eigen::MatrixXd noise = getNoise(Tr);
            Ui.middleRows(i * dim_u, dim_u) += noise;
            h(Ui.middleRows(i * dim_u, dim_u));

            Xi.col(0) = x_init;
            double cost = 0.0;
            for (int j = 0; j < Tr; ++j) {
                cost += p(Xi.col(j), x_target);
                Xi.col(j+1) = Xi.col(j) + (dt * f(Xi.col(j), Ui.block(i * dim_u, j, dim_u, 1)));
            }
            cost += p(Xi.col(Tr), x_target);
            
            // Guide Cost
            cost += (Xi - X_ref).colwise().norm().sum();
            
            for (int j = 0; j < Tr + 1; ++j) {
                if (collision_checker->getCollision(Xi.col(j))) { // 수정 완료
                    cost = 1e8;
                    break;
                }
            }
            costs(i) = cost;
        }
        
        double min_cost = costs.minCoeff();
        weights = (-gamma_u * (costs.array() - min_cost)).exp();
        double total_weight = weights.sum();
        weights /= total_weight;

        Eigen::MatrixXd Ures = Eigen::MatrixXd::Zero(dim_u, Tr);
        for (int i = 0; i < Nr; ++i) {
            Ures += Ui.middleRows(i * dim_u, dim_u) * weights(i);
        }
        h(Ures);

        Eigen::MatrixXd Xi(dim_x, Tr + 1);
        Xi.col(0) = x_init;
        double cost = 0.0;
        for (int j = 0; j < Tr; ++j) {
            cost += p(Xi.col(j), x_target);
            Xi.col(j+1) = Xi.col(j) + (dt * f(Xi.col(j), Ures.col(j)));
        }
        cost += p(Xi.col(Tr), x_target);
        
        for (int j = 0; j < Tr + 1; ++j) {
            if (collision_checker->getCollision(Xi.col(j))) { // 수정 완료
                cost = 1e8;
                break;
            }
        }

        Ur.push_back(Ures);
        Cr.push_back(cost);
        Xr.push_back(Xi);
    }

    double min_cost = std::numeric_limits<double>::max();
    int index = 0;
    for (int r = 0; r < joints.size(); ++r) {
        if (Cr[r] < min_cost) {
            min_cost = Cr[r];
            index = r;
        }
    }
    Uo = Ur[index];
    Xo = Xr[index];
    u0 = Uo.col(0);
}

template<typename ModelClass>
void BiMPPI<ModelClass>::partitioningControl() {
    U_f0 = Uo.leftCols(Tf);
    U_b0 = Eigen::MatrixXd::Zero(dim_u, Tb);
}

template<typename ModelClass>
void BiMPPI<ModelClass>::dbscan(std::vector<std::vector<int>> &clusters, const Eigen::MatrixXd &Di, const Eigen::VectorXd &costs, const int &N, const int &T) {
    clusters.clear();
    std::vector<bool> core_points(N, false);
    std::map<int, std::vector<int>> core_tree;

    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        if (costs(i) > 1E7) {continue;}
        for (int j = i + 1; j < N; ++j) {
            if (costs(j) > 1E7) {continue;}
            if (deviation_mu * (Di.col(i) - Di.col(j)).norm() < epsilon) {
                #pragma omp critical
                {
                core_tree[i].push_back(j);
                core_tree[j].push_back(i);
                }
            }
        }
    }

    for (int i = 0; i < N; ++i) {
        if (minpts < core_tree[i].size()) {
            core_points[i] = true;
        }
    }
    
    std::vector<bool> visited(N, false);
    for (int i = 0; i < N; ++i) {
        if (!core_points[i]) {continue;}
        if (visited[i]) {continue;}
        std::deque<int> branch;
        std::vector<int> cluster;
        branch.push_back(i);
        cluster.push_back(i);
        visited[i] = true;
        while (!branch.empty()) {
            int now = branch.front();
            for (int next : core_tree[now]) {
                if (visited[next]) {continue;}
                visited[next] = true;
                cluster.push_back(next);
                if (core_points[next]) {branch.push_back(next);}
            }
            branch.pop_front();
        }
        clusters.push_back(cluster);
    }
}

template<typename ModelClass>
void BiMPPI<ModelClass>::calculateU(Eigen::MatrixXd &U, const std::vector<std::vector<int>> &clusters, const Eigen::VectorXd &costs, const Eigen::MatrixXd &Ui, const int &T) {
    U = Eigen::MatrixXd::Zero(clusters.size() * dim_u, T);
    #pragma omp parallel for
    for (int index = 0; index < clusters.size(); ++index) {
        int pts = clusters[index].size();
        std::vector<double> weights(pts);
        double min_cost = std::numeric_limits<double>::max();
        for (int k : clusters[index]) {
            min_cost = std::min(min_cost, costs(k));
        }

        for (size_t i = 0; i < pts; ++i) {
            weights[i] = std::exp(-gamma_u * (costs(clusters[index][i]) - min_cost));
        }
        double total_weight = std::accumulate(weights.begin(), weights.end(), 0.0);

        for (size_t i = 0; i < pts; ++i) {
            U.middleRows(index * dim_u, dim_u) += (weights[i] / total_weight) * Ui.middleRows(clusters[index][i] * dim_u, dim_u);
        }
        h(U.middleRows(index * dim_u, dim_u));
    }
}

template<typename ModelClass>
void BiMPPI<ModelClass>::savePathToCSV(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) return;
    
    for (const auto& p : visual_traj) {
        if (p.size() == dim_x) {
            for(int i=0; i<dim_x; i++) {
                file << p(i) << (i == dim_x - 1 ? "" : ",");
            }
            file << "\n";
        }
    }
}

template<typename ModelClass>
double BiMPPI<ModelClass>::pathLengthJoint(double eps_ignore) const {
    double L = 0.0;
    const size_t n = visual_traj.size();
    if (n < 2) return 0.0;
    
    Eigen::VectorXd prev_q = visual_traj[0];
    for (size_t i = 1; i < n; ++i) {
        Eigen::VectorXd curr_q = visual_traj[i];
        double d = (curr_q - prev_q).norm();
        if (d > eps_ignore) L += d;
        prev_q = curr_q;
    }
    return L;
}