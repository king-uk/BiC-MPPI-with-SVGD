#pragma once
#include "matplotlibcpp.h"

#include <EigenRand/EigenRand>

#include "mppi_param.h"
#include "collision_checker.h"
#include "model_base.h"

#include <ctime>
#include <vector>
#include <deque>
#include <map>
#include <chrono>
#include <iostream>

#include <omp.h>

#include <cmath>
#include <numeric>
#include <fstream>
#include <stdexcept>
#include <limits>
#include <atomic>

namespace bimppi_detail {

// ---- compile-safe param member detection helpers ----
template <typename P>
auto get_istep_impl(const P& p, int) -> decltype(p.istep, int()) { return p.istep; }
template <typename P>
auto get_istep_impl(const P& p, long) -> decltype(p.num_svgd_iteration, int()) { return p.num_svgd_iteration; }
template <typename P>
auto get_istep_impl(const P& p, char) -> decltype(p.num_svgd_iteration_, int()) { return p.num_svgd_iteration_; }
template <typename P>
int get_istep_impl(const P&, ...) { return 1; }

template <typename P>
int get_istep(const P& p) {
    int v = get_istep_impl(p, 0);
    return (v < 1) ? 1 : v;
}

// local MPPI sample count (optional). If absent, use Ns.
template <typename P>
auto get_nmppi_impl(const P& p, int) -> decltype(p.Nm, int()) { return p.Nm; }
template <typename P>
auto get_nmppi_impl(const P& p, long) -> decltype(p.Nmppi, int()) { return p.Nmppi; }
template <typename P>
int get_nmppi_impl(const P&, ...) { return -1; }

template <typename P>
int get_nmppi(const P& p, int fallback) {
    int v = get_nmppi_impl(p, 0);
    if (v <= 0) v = fallback;
    return v;
}

} // namespace bimppi_detail


class BiMPPI {
public:
    template<typename ModelClass>
    BiMPPI(ModelClass model);
    ~BiMPPI();

    void init(SVGDMPPIParam svgd_mppi_param); // 선언/정의 일치

    void setCollisionChecker(CollisionChecker *collision_checker);
    Eigen::MatrixXd getNoise(const int &T);
    void move();

    void solve();
    void backwardRollout();
    void forwardRollout();

    void SVGD();

    void selectConnection();
    void concatenate();
    void guideMPPI();
    void partitioningControl();

    void savePathToCSV(const std::string& filename) const;

    void dbscan(std::vector<std::vector<int>> &clusters, const Eigen::MatrixXd &Di, const Eigen::VectorXd &costs, const int &N, const int &T);
    void calculateU(Eigen::MatrixXd &U, const std::vector<std::vector<int>> &clusters, const Eigen::VectorXd &costs, const Eigen::MatrixXd &Ui, const int &T);
    void show();
    void showTraj();

    double pathLengthXY(double eps_ignore = 0.0) const;

    std::chrono::time_point<std::chrono::high_resolution_clock> start;
    std::chrono::time_point<std::chrono::high_resolution_clock> finish;
    std::chrono::duration<double> elapsed_1;
    std::chrono::duration<double> elapsed_2;
    std::chrono::duration<double> elapsed_3;
    std::chrono::duration<double> elapsed_4;
    double elapsed;
    std::vector<Eigen::VectorXd> visual_traj;

    Eigen::MatrixXd U_f0;
    Eigen::MatrixXd U_b0;

    Eigen::VectorXd x_init;
    Eigen::VectorXd x_target;
    Eigen::VectorXd dummy_u;

    Eigen::MatrixXd Uo;
    Eigen::MatrixXd Xo;
    Eigen::VectorXd u0;
    
private:
    int dim_x;
    int dim_u;

    std::function<Eigen::MatrixXd(Eigen::VectorXd, Eigen::VectorXd)> f;
    std::function<double(Eigen::VectorXd, Eigen::VectorXd)> q;
    std::function<double(Eigen::VectorXd, Eigen::VectorXd)> p;
    std::function<void(Eigen::Ref<Eigen::MatrixXd>)> h;

    std::mt19937_64 urng{static_cast<std::uint_fast64_t>(std::time(nullptr))};
    Eigen::Rand::NormalGen<double> norm_gen{0.0, 1.0};

    float dt;
    int Tf;
    int Tb;

    int Nf;
    int Nb;

    int Ns;      // surrogate grad 샘플 수
    int istep;   // SVGD i-step iteration 수 (추가)

    int Nr;

    double gamma_u;
    Eigen::MatrixXd sigma_u; // "sqrt-cov"로 쓰이고 있음 (noise = sigma_u * z)

    double deviation_mu;
    double cost_mu;
    int minpts;
    double epsilon;
    double psi; // svgd step size

    CollisionChecker *collision_checker{nullptr};

    std::vector<std::vector<int>> clusters_f;
    std::vector<int> full_cluster_f;
    Eigen::MatrixXd Uf;
    Eigen::MatrixXd Xf;

    std::vector<std::vector<int>> clusters_b;
    std::vector<int> full_cluster_b;
    Eigen::MatrixXd Ub;
    Eigen::MatrixXd Xb;

    std::vector<std::vector<int>> joints;
    std::vector<Eigen::MatrixXd> Xc;
    std::vector<Eigen::MatrixXd> Uc;

    std::vector<Eigen::MatrixXd> Ur;
    std::vector<double> Cr;
    std::vector<Eigen::MatrixXd> Xr;
};

template<typename ModelClass>
BiMPPI::BiMPPI(ModelClass model) {
    this->dim_x = model.dim_x;
    this->dim_u = model.dim_u;

    this->f = model.f;
    this->q = model.q;
    this->p = model.p;
    this->h = model.h;
}

BiMPPI::~BiMPPI() {}

void BiMPPI::init(SVGDMPPIParam svgd_mppi_param) {
    this->dt = svgd_mppi_param.dt;
    this->Tf = svgd_mppi_param.Tf;
    this->Tb = svgd_mppi_param.Tb;
    this->x_init = svgd_mppi_param.x_init;
    this->x_target = svgd_mppi_param.x_target;

    this->Nf = svgd_mppi_param.Nf;
    this->Nb = svgd_mppi_param.Nb;

    this->Ns = svgd_mppi_param.Ns;
    // this->istep = bimppi_detail::get_istep(svgd_mppi_param.istep); // (있으면 읽고, 없으면 1)
    this->istep = svgd_mppi_param.istep;

    this->Nr = svgd_mppi_param.Nr;
    this->gamma_u = svgd_mppi_param.gamma_u;
    this->sigma_u = svgd_mppi_param.sigma_u;
    this->deviation_mu = svgd_mppi_param.deviation_mu;
    this->cost_mu = svgd_mppi_param.cost_mu;
    this->minpts = svgd_mppi_param.minpts;
    this->epsilon = svgd_mppi_param.epsilon;
    this->psi = svgd_mppi_param.psi;

    full_cluster_f.resize(Nf);
    std::iota(full_cluster_f.begin(), full_cluster_f.end(), 0);
    full_cluster_b.resize(Nb);
    std::iota(full_cluster_b.begin(), full_cluster_b.end(), 0);

    u0 = Eigen::VectorXd::Zero(dim_u);
    dummy_u = Eigen::VectorXd::Zero(dim_u);
}

void BiMPPI::setCollisionChecker(CollisionChecker *collision_checker) {
    this->collision_checker = collision_checker;
}

Eigen::MatrixXd BiMPPI::getNoise(const int &T) {
    return sigma_u * norm_gen.template generate<Eigen::MatrixXd>(dim_u, T, urng);
}

void BiMPPI::move() {
    x_init = x_init + (dt * f(x_init, u0));
    U_f0.leftCols(U_f0.cols() - 1) = U_f0.rightCols(U_f0.cols() - 1);
}

void BiMPPI::solve() {
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


// ==========================================================
// (변경) backwardRollout(): i-step SVGD 이동 + 이동 공분산 추정
// + cluster별 nominal selection + covariance-adapted MPPI로 최종 Ub 업데이트
// ==========================================================
void BiMPPI::backwardRollout() {
    Eigen::MatrixXd Ui = U_b0.replicate(Nb, 1); // particle controls
    Eigen::MatrixXd Di(dim_u, Nb);
    Eigen::VectorXd costs(Nb);

    // Σ = sigma_u*sigma_u^T (q(V|U) score용)
    const Eigen::MatrixXd Sigma = sigma_u * sigma_u.transpose();
    const Eigen::LDLT<Eigen::MatrixXd> Sigma_inv(Sigma);

    // 이동 공분산(각 particle별 dim_u x dim_u) 누적
    std::vector<Eigen::MatrixXd> cov_move_particles(Nb, Eigen::MatrixXd::Zero(dim_u, dim_u));
    std::vector<double> cov_move_counts(Nb, 0.0);

    // thread-safe seeds
    std::vector<std::uint64_t> seeds(Nb);
    for (int i = 0; i < Nb; ++i) seeds[i] = urng();

    auto rollout_cost_backward = [&](const Eigen::MatrixXd& U)->double {
        Eigen::VectorXd x = x_target;
        double c = 0.0;

        if (collision_checker && collision_checker->getCollisionGrid(x)) return 1e8;

        for (int j = Tb - 1; j >= 0; --j) {
            const Eigen::VectorXd u_use = (j == Tb - 1) ? U.col(j) : U.col(j + 1);
            x = x - (dt * f(x, u_use));
            c += p(x, x_init);
            if (collision_checker && collision_checker->getCollisionGrid(x)) return 1e8;
        }
        c += p(x, x_init);
        return c;
    };

    // (0) 초기 Nb 샘플 생성
    #pragma omp parallel for
    for (int i = 0; i < Nb; ++i) {
        std::mt19937_64 rng(seeds[i] ^ (0x9e3779b97f4a7c15ULL + (std::uint64_t)i));
        Eigen::Rand::NormalGen<double> normal(0.0, 1.0);

        Eigen::Ref<Eigen::MatrixXd> U = Ui.middleRows(i * dim_u, dim_u);
        U.noalias() += sigma_u * normal.generate<Eigen::MatrixXd>(dim_u, Tb, rng);
        h(U);
    }

    // (1) SVGD i-step 이동 + 이동량으로 공분산 누적
    #pragma omp parallel for
    for (int i = 0; i < Nb; ++i) {
        std::mt19937_64 rng(seeds[i] ^ (0xD1B54A32D192ED03ULL + (std::uint64_t)i));
        Eigen::Rand::NormalGen<double> normal(0.0, 1.0);

        Eigen::Ref<Eigen::MatrixXd> U = Ui.middleRows(i * dim_u, dim_u);

        Eigen::MatrixXd cov_acc = Eigen::MatrixXd::Zero(dim_u, dim_u);
        double cov_cnt = 0.0;

        for (int it = 0; it < istep; ++it) {
            // surrogate samples
            std::vector<Eigen::MatrixXd> Vset;
            Vset.reserve(Ns);
            std::vector<double> Jset(Ns, 0.0);
            double Jmin = std::numeric_limits<double>::infinity();

            for (int s = 0; s < Ns; ++s) {
                Eigen::MatrixXd V = U + sigma_u * normal.generate<Eigen::MatrixXd>(dim_u, Tb, rng);
                h(V);
                double J = rollout_cost_backward(V);
                Vset.push_back(std::move(V));
                Jset[s] = J;
                if (J < Jmin) Jmin = J;
            }

            // weights for surrogate gradient
            std::vector<double> w(Ns, 0.0);
            double sumw = 0.0;
            for (int s = 0; s < Ns; ++s) {
                double ws = std::exp(-cost_mu * (Jset[s] - Jmin));
                w[s] = ws;
                sumw += ws;
            }
            if (!(sumw > 0.0) || !std::isfinite(sumw)) {
                sumw = (double)Ns;
                std::fill(w.begin(), w.end(), 1.0);
            }

            // grad = E_w[ Σ^{-1}(V-U) ]
            Eigen::MatrixXd grad = Eigen::MatrixXd::Zero(dim_u, Tb);
            for (int s = 0; s < Ns; ++s) {
                grad.noalias() += w[s] * Sigma_inv.solve(Vset[s] - U);
            }
            grad /= sumw;

            // update
            Eigen::MatrixXd dU = psi * grad;
            U.noalias() += dU;
            h(U);

            // covariance from movement history: accumulate over time
            for (int t = 0; t < Tb; ++t) {
                Eigen::VectorXd du = dU.col(t);
                cov_acc.noalias() += du * du.transpose();
            }
            cov_cnt += (double)Tb;
        }

        cov_move_particles[i] = cov_acc;
        cov_move_counts[i] = cov_cnt;
    }

    // (2) SVGD 후 particle cost/Di 계산
    std::atomic_bool all_feasible(true);

    #pragma omp parallel for
    for (int i = 0; i < Nb; ++i) {
        Eigen::Ref<Eigen::MatrixXd> U = Ui.middleRows(i * dim_u, dim_u);
        double J = rollout_cost_backward(U);
        if (J > 1e7) all_feasible.store(false, std::memory_order_relaxed);

        costs(i) = J;
        Di.col(i) = (U - U_b0).rowwise().mean();
    }

    // (3) clustering (기존 구조 유지)
    if (!all_feasible.load(std::memory_order_relaxed)) { dbscan(clusters_b, Di, costs, Nb, Tb); }
    else { clusters_b.clear(); }

    if (clusters_b.empty()) { clusters_b.push_back(full_cluster_b); }

    // (4) 일단 기존 calculateU로 cluster mean 생성
    calculateU(Ub, clusters_b, costs, Ui, Tb);

    // (5) stein 형식: cluster별 peak/nominal 선택 + covariance fit + MPPI refine로 Ub를 최종 업데이트
    const int Nmppi = bimppi_detail::get_nmppi(*(SVGDMPPIParam*)nullptr, Ns); // 컴파일러 방지용; 아래에서 직접 Ns 사용
    (void)Nmppi;

    for (int c = 0; c < (int)clusters_b.size(); ++c) {
        const auto& cl = clusters_b[c];

        // nominal: 최소 cost particle 선택
        int best_i = cl[0];
        double best_J = costs(best_i);
        for (int idx : cl) {
            if (costs(idx) < best_J) { best_J = costs(idx); best_i = idx; }
        }
        Eigen::MatrixXd U_nom = Ui.middleRows(best_i * dim_u, dim_u);

        // covariance fit: movement covariance 평균(클러스터 내부 soft weight 가능)
        Eigen::MatrixXd Cov = Eigen::MatrixXd::Zero(dim_u, dim_u);
        double wsum = 0.0;

        // cluster weights (MPPI 스타일)
        double cmin = std::numeric_limits<double>::infinity();
        for (int idx : cl) cmin = std::min(cmin, costs(idx));
        for (int idx : cl) {
            double wi = std::exp(-gamma_u * (costs(idx) - cmin));
            wsum += wi;

            double cnt = std::max(1.0, cov_move_counts[idx]);
            Cov.noalias() += wi * (cov_move_particles[idx] / cnt);
        }
        if (!(wsum > 0.0)) wsum = 1.0;
        Cov /= wsum;

        // regularize + fallback
        const double reg = 1e-9;
        Cov.diagonal().array() += reg;

        Eigen::LLT<Eigen::MatrixXd> llt(Cov);
        Eigen::MatrixXd L;
        if (llt.info() == Eigen::Success) {
            L = llt.matrixL();
        } else {
            // fallback: base Sigma
            Eigen::LLT<Eigen::MatrixXd> llt2(Sigma + reg * Eigen::MatrixXd::Identity(dim_u, dim_u));
            L = (llt2.info() == Eigen::Success) ? llt2.matrixL() : sigma_u; // 최후 fallback
        }

        // MPPI refine around U_nom using L
        const int K = std::max(1, Ns); // 최종 MPPI 샘플 수 (요청: stein 형식이므로 Ns 기반)
        std::vector<double> Jm(K, 0.0);
        std::vector<Eigen::MatrixXd> Um(K);

        // local RNG
        std::mt19937_64 rng(urng() ^ (0x94D049BB133111EBULL + (std::uint64_t)c));
        Eigen::Rand::NormalGen<double> normal(0.0, 1.0);

        double Jmin = std::numeric_limits<double>::infinity();
        for (int k = 0; k < K; ++k) {
            Eigen::MatrixXd noise = L * normal.generate<Eigen::MatrixXd>(dim_u, Tb, rng);
            Um[k] = U_nom + noise;
            h(Um[k]);
            double Jk = rollout_cost_backward(Um[k]);
            Jm[k] = Jk;
            if (Jk < Jmin) Jmin = Jk;
        }

        Eigen::VectorXd w(K);
        for (int k = 0; k < K; ++k) w(k) = std::exp(-gamma_u * (Jm[k] - Jmin));
        double wtot = w.sum();
        if (!(wtot > 0.0) || !std::isfinite(wtot)) { w.setOnes(); wtot = (double)K; }
        w /= wtot;

        Eigen::MatrixXd U_upd = Eigen::MatrixXd::Zero(dim_u, Tb);
        for (int k = 0; k < K; ++k) U_upd.noalias() += w(k) * Um[k];
        h(U_upd);

        // 최종 Ub 업데이트
        Ub.middleRows(c * dim_u, dim_u) = U_upd;
    }

    // (6) Xb 재생성 (업데이트된 Ub 기준)
    Xb.resize((int)clusters_b.size() * dim_x, Tb + 1);
    for (int i = 0; i < (int)clusters_b.size(); ++i) {
        Xb.block(i * dim_x, Tb, dim_x, 1) = x_target;
        for (int t = Tb - 1; t >= 0; --t) {
            if (t == Tb - 1) {
                Xb.block(i * dim_x, t, dim_x, 1) =
                    Xb.block(i * dim_x, t + 1, dim_x, 1) - (dt * f(Xb.block(i * dim_x, t + 1, dim_x, 1),
                                                                  Ub.block(i * dim_u, t, dim_u, 1)));
            } else {
                Xb.block(i * dim_x, t, dim_x, 1) =
                    Xb.block(i * dim_x, t + 1, dim_x, 1) - (dt * f(Xb.block(i * dim_x, t + 1, dim_x, 1),
                                                                  Ub.block(i * dim_u, t + 1, dim_u, 1)));
            }
        }
    }
}


// ==========================================================
// (변경) forwardRollout(): i-step SVGD 이동 + 이동 공분산 추정
// + cluster별 nominal selection + covariance-adapted MPPI로 최종 Uf 업데이트
// ==========================================================
void BiMPPI::forwardRollout() {
    Eigen::MatrixXd Ui = U_f0.replicate(Nf, 1);
    Eigen::MatrixXd Di(dim_u, Nf);
    Eigen::VectorXd costs(Nf);

    const Eigen::MatrixXd Sigma = sigma_u * sigma_u.transpose();
    const Eigen::LDLT<Eigen::MatrixXd> Sigma_inv(Sigma);

    std::vector<Eigen::MatrixXd> cov_move_particles(Nf, Eigen::MatrixXd::Zero(dim_u, dim_u));
    std::vector<double> cov_move_counts(Nf, 0.0);

    std::vector<std::uint64_t> seeds(Nf);
    for (int i = 0; i < Nf; ++i) seeds[i] = urng();

    auto rollout_cost_forward = [&](const Eigen::MatrixXd& U)->double {
        Eigen::VectorXd x = x_init;
        double c = 0.0;

        if (collision_checker && collision_checker->getCollisionGrid(x)) return 1e8;

        for (int j = 0; j < Tf; ++j) {
            c += p(x, x_target);
            x = x + (dt * f(x, U.col(j)));
            if (collision_checker && collision_checker->getCollisionGrid(x)) return 1e8;
        }
        c += p(x, x_target);
        return c;
    };

    // (0) 초기 Nf 샘플 생성
    #pragma omp parallel for
    for (int i = 0; i < Nf; ++i) {
        std::mt19937_64 rng(seeds[i] ^ (0x9e3779b97f4a7c15ULL + (std::uint64_t)i));
        Eigen::Rand::NormalGen<double> normal(0.0, 1.0);

        Eigen::Ref<Eigen::MatrixXd> U = Ui.middleRows(i * dim_u, dim_u);
        U.noalias() += sigma_u * normal.generate<Eigen::MatrixXd>(dim_u, Tf, rng);
        h(U);
    }

    // (1) SVGD i-step 이동 + 이동 공분산 누적
    #pragma omp parallel for
    for (int i = 0; i < Nf; ++i) {
        std::mt19937_64 rng(seeds[i] ^ (0xD1B54A32D192ED03ULL + (std::uint64_t)i));
        Eigen::Rand::NormalGen<double> normal(0.0, 1.0);

        Eigen::Ref<Eigen::MatrixXd> U = Ui.middleRows(i * dim_u, dim_u);

        Eigen::MatrixXd cov_acc = Eigen::MatrixXd::Zero(dim_u, dim_u);
        double cov_cnt = 0.0;

        for (int it = 0; it < istep; ++it) {
            std::vector<Eigen::MatrixXd> Vset;
            Vset.reserve(Ns);
            std::vector<double> Jset(Ns, 0.0);
            double Jmin = std::numeric_limits<double>::infinity();

            for (int s = 0; s < Ns; ++s) {
                Eigen::MatrixXd V = U + sigma_u * normal.generate<Eigen::MatrixXd>(dim_u, Tf, rng);
                h(V);
                double J = rollout_cost_forward(V);
                Vset.push_back(std::move(V));
                Jset[s] = J;
                if (J < Jmin) Jmin = J;
            }

            std::vector<double> w(Ns, 0.0);
            double sumw = 0.0;
            for (int s = 0; s < Ns; ++s) {
                double ws = std::exp(-cost_mu * (Jset[s] - Jmin));
                w[s] = ws;
                sumw += ws;
            }
            if (!(sumw > 0.0) || !std::isfinite(sumw)) {
                sumw = (double)Ns;
                std::fill(w.begin(), w.end(), 1.0);
            }

            Eigen::MatrixXd grad = Eigen::MatrixXd::Zero(dim_u, Tf);
            for (int s = 0; s < Ns; ++s) {
                grad.noalias() += w[s] * Sigma_inv.solve(Vset[s] - U);
            }
            grad /= sumw;

            Eigen::MatrixXd dU = psi * grad;
            U.noalias() += dU;
            h(U);

            for (int t = 0; t < Tf; ++t) {
                Eigen::VectorXd du = dU.col(t);
                cov_acc.noalias() += du * du.transpose();
            }
            cov_cnt += (double)Tf;
        }

        cov_move_particles[i] = cov_acc;
        cov_move_counts[i] = cov_cnt;
    }

    // (2) SVGD 후 particle cost/Di 계산
    std::atomic_bool all_feasible(true);

    #pragma omp parallel for
    for (int i = 0; i < Nf; ++i) {
        Eigen::Ref<Eigen::MatrixXd> U = Ui.middleRows(i * dim_u, dim_u);
        double J = rollout_cost_forward(U);
        if (J > 1e7) all_feasible.store(false, std::memory_order_relaxed);

        costs(i) = J;
        Di.col(i) = (U - U_f0).rowwise().mean();
    }

    // (3) clustering
    if (!all_feasible.load(std::memory_order_relaxed)) { dbscan(clusters_f, Di, costs, Nf, Tf); }
    else { clusters_f.clear(); }

    if (clusters_f.empty()) { clusters_f.push_back(full_cluster_f); }

    // (4) 초기 cluster mean
    calculateU(Uf, clusters_f, costs, Ui, Tf);

    // (5) stein 형식: cluster별 nominal + covariance fit + MPPI refine로 Uf 최종 업데이트
    for (int c = 0; c < (int)clusters_f.size(); ++c) {
        const auto& cl = clusters_f[c];

        int best_i = cl[0];
        double best_J = costs(best_i);
        for (int idx : cl) {
            if (costs(idx) < best_J) { best_J = costs(idx); best_i = idx; }
        }
        Eigen::MatrixXd U_nom = Ui.middleRows(best_i * dim_u, dim_u);

        Eigen::MatrixXd Cov = Eigen::MatrixXd::Zero(dim_u, dim_u);
        double wsum = 0.0;

        double cmin = std::numeric_limits<double>::infinity();
        for (int idx : cl) cmin = std::min(cmin, costs(idx));

        for (int idx : cl) {
            double wi = std::exp(-gamma_u * (costs(idx) - cmin));
            wsum += wi;

            double cnt = std::max(1.0, cov_move_counts[idx]);
            Cov.noalias() += wi * (cov_move_particles[idx] / cnt);
        }
        if (!(wsum > 0.0)) wsum = 1.0;
        Cov /= wsum;

        const double reg = 1e-9;
        Cov.diagonal().array() += reg;

        Eigen::LLT<Eigen::MatrixXd> llt(Cov);
        Eigen::MatrixXd L;
        if (llt.info() == Eigen::Success) {
            L = llt.matrixL();
        } else {
            Eigen::LLT<Eigen::MatrixXd> llt2(Sigma + reg * Eigen::MatrixXd::Identity(dim_u, dim_u));
            L = (llt2.info() == Eigen::Success) ? llt2.matrixL() : sigma_u;
        }

        const int K = std::max(1, Ns);
        std::vector<double> Jm(K, 0.0);
        std::vector<Eigen::MatrixXd> Um(K);

        std::mt19937_64 rng(urng() ^ (0x94D049BB133111EBULL + (std::uint64_t)c));
        Eigen::Rand::NormalGen<double> normal(0.0, 1.0);

        double Jmin = std::numeric_limits<double>::infinity();
        for (int k = 0; k < K; ++k) {
            Eigen::MatrixXd noise = L * normal.generate<Eigen::MatrixXd>(dim_u, Tf, rng);
            Um[k] = U_nom + noise;
            h(Um[k]);
            double Jk = rollout_cost_forward(Um[k]);
            Jm[k] = Jk;
            if (Jk < Jmin) Jmin = Jk;
        }

        Eigen::VectorXd w(K);
        for (int k = 0; k < K; ++k) w(k) = std::exp(-gamma_u * (Jm[k] - Jmin));
        double wtot = w.sum();
        if (!(wtot > 0.0) || !std::isfinite(wtot)) { w.setOnes(); wtot = (double)K; }
        w /= wtot;

        Eigen::MatrixXd U_upd = Eigen::MatrixXd::Zero(dim_u, Tf);
        for (int k = 0; k < K; ++k) U_upd.noalias() += w(k) * Um[k];
        h(U_upd);

        Uf.middleRows(c * dim_u, dim_u) = U_upd;
    }

    // (6) Xf 재생성 (업데이트된 Uf 기준)
    Xf.resize((int)clusters_f.size() * dim_x, Tf + 1);
    for (int i = 0; i < (int)clusters_f.size(); ++i) {
        Xf.block(i * dim_x, 0, dim_x, 1) = x_init;
        for (int t = 0; t < Tf; ++t) {
            Xf.block(i * dim_x, t + 1, dim_x, 1) =
                Xf.block(i * dim_x, t, dim_x, 1) + (dt * f(Xf.block(i * dim_x, t, dim_x, 1),
                                                           Uf.block(i * dim_u, t, dim_u, 1)));
        }
    }
}

void BiMPPI::SVGD() {
    // forward/backwardRollout 내부에서 stein 형식 pipeline을 수행하도록 통합함
}


// ===== 이하 함수들은 사용자 코드 그대로 (selectConnection/concatenate/guideMPPI/partitioningControl/dbscan/calculateU/show/showTraj/pathLengthXY/savePathToCSV) =====
// (중복 방지를 위해 여기서는 생략하지 않고, 사용자가 제공한 그대로 이어 붙이면 됩니다.)

// ... 사용자가 준 원본의 selectConnection() 이하를 그대로 붙여주세요 ...

void BiMPPI::selectConnection() {
    
    // cf : forward cluster index
    // cb : backward cluster index
    // df : forward time index
    // db : backward time index

    // 그냥 Euclidean으로만 구현,,,,,, 이거를 dynamics를 넣으면서 계산하면 어떻까?
    // 일단 계산시간은 많이 늘어날듯.,.... 하지만 Guide 과정이 빠져서 그냥 사용할 수 있을듯
    
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

void BiMPPI::concatenate() {
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

        // Fill if lenght is shorter than Tf
        if (df + (Tb - db) < Tf) {
            U.rightCols(Tf - (df + (Tb - db))).colwise() = dummy_u;
            X.rightCols(Tf - (df + (Tb - db))).colwise() = x_target;
        }

        Uc.push_back(U);
        Xc.push_back(X);
    }
}

void BiMPPI::guideMPPI() {
    Ur.clear();
    Cr.clear();
    Xr.clear();
    
    // #pragma omp parallel for
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
            cost = p(Xi.col(Tr), x_target);
            // Guide Cost
            cost += (Xi - X_ref).colwise().norm().sum();
            for (int j = 0; j < Tr + 1; ++j) {
                if (collision_checker->getCollisionGrid(Xi.col(j))) {
                    cost = 1e8;
                    break;
                }
            }
            costs(i) = cost;
        }
        double min_cost = costs.minCoeff();
        weights = (-gamma_u * (costs.array() - min_cost)).exp();
        double total_weight =  weights.sum();
        weights /= total_weight;

        Eigen::MatrixXd Ures = Eigen::MatrixXd::Zero(dim_u, Tr);
        for (int i = 0; i < Nr; ++i) {
            Ures += Ui.middleRows(i * dim_u, dim_u) * weights(i);
        }
        h(Ures);

        // OCP Cost Calculation
        Eigen::MatrixXd Xi(dim_x, Tr + 1);
        Xi.col(0) = x_init;
        double cost = 0.0;
        for (int j = 0; j < Tr; ++j) {
            cost += p(Xi.col(j), x_target);
            Xi.col(j+1) = Xi.col(j) + (dt * f(Xi.col(j), Ures.col(j)));
        }
        cost += p(Xi.col(Tr), x_target);
        for (int j = 0; j < Tr + 1; ++j) {
            if (collision_checker->getCollisionGrid(Xi.col(j))) {
                cost = 1e8;
                break;
            }
        }

        Ur.push_back(Ures);
        Cr.push_back(cost);
        Xr.push_back(Xi);
    }

    // Optimal Control Result
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

void BiMPPI::partitioningControl() {
    U_f0 = Uo.leftCols(Tf);
    U_b0 = Eigen::MatrixXd::Zero(dim_u, Tb);
}

void BiMPPI::dbscan(std::vector<std::vector<int>> &clusters, const Eigen::MatrixXd &Di, const Eigen::VectorXd &costs, const int &N, const int &T) {
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

void BiMPPI::calculateU(Eigen::MatrixXd &U, const std::vector<std::vector<int>> &clusters, const Eigen::VectorXd &costs, const Eigen::MatrixXd &Ui, const int &T) {
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

void BiMPPI::show() {
    namespace plt = matplotlibcpp;
    // plt::subplot(1,2,1);

    double resolution = 0.1;
    double hl = resolution / 2;
    for (int i = 0; i < collision_checker->map.size(); ++i) {
        for (int j = 0; j < collision_checker->map[0].size(); ++j) {
            if ((collision_checker->map[i])[j] == 10) {
                double mx = i*resolution;
                double my = j*resolution;
                std::vector<double> oX = {mx-hl, mx+hl, mx+hl, mx-hl, mx-hl};
                std::vector<double> oY = {my-hl,my-hl,my+hl,my+hl,my-hl};
                plt::plot(oX, oY, "k");
            }
        }
    }

    // for (int index = 0; index < clusters_f.size(); ++index) {
    //     for (int k : clusters_f[index]) {
    //         Eigen::MatrixXd Xi(dim_x, Tf+1);
    //         Xi.col(0) = x_init;
    //         for (int t = 0; t < Tf; ++t) {
    //             Xi.col(t+1) = f(Xi.col(t), Ui_f.block(k * dim_u, t, dim_u, 1)).cast<double>();
    //         }
    //         std::vector<std::vector<double>> X_MPPI(dim_x, std::vector<double>(Tf));
    //         for (int i = 0; i < dim_x; ++i) {
    //             for (int j = 0; j < Tf + 1; ++j) {
    //                 X_MPPI[i][j] = Xi(i, j);
    //             }
    //         }
    //         // std::cout<<"deviation = "<<Di_f(k)<<"\t";
    //         // std::cout<<"X = "<<Xi.col(T)(0)<<std::endl;
    //         std::string color = "C" + std::to_string(index%10);
    //         // plt::plot(X_MPPI[0], X_MPPI[1], {{"color", color}, {"linewidth", "1.0"}});
    //     }
    // }

    // for (int index = 0; index < clusters_b.size(); ++index) {
    //     for (int k : clusters_b[index]) {
    //         Eigen::MatrixXd Xi(dim_x, Tb+1);
    //         Xi.col(Tb) = x_target;
    //         for (int t = Tb - 1; t >= 0; --t) {
    //             // Xi.col(t) = Xi.col(t+1) - 0.05*f(Xi.col(t+1), Ui_b.block(k * dim_u, t, dim_u, 1)).cast<double>();
    //             Xi.col(t) = 2*Xi.col(t+1) - f(Xi.col(t+1), Ui_b.block(k * dim_u, t, dim_u, 1)).cast<double>();
    //         }
    //         std::vector<std::vector<double>> X_MPPI(dim_x, std::vector<double>(Tb));
    //         for (int i = 0; i < dim_x; ++i) {
    //             for (int j = 0; j < Tb + 1; ++j) {
    //                 X_MPPI[i][j] = Xi(i, j);
    //             }
    //         }
    //         // std::cout<<"deviation = "<<Di_f(k)<<"\t";
    //         // std::cout<<"X = "<<Xi.col(T)(0)<<std::endl;
    //         std::string color = "C" + std::to_string(9 - index%10);
    //         // plt::plot(X_MPPI[0], X_MPPI[1], {{"color", color}, {"linewidth", "1.0"}});
    //     }
    // }

    for (int index = 0; index < clusters_f.size(); ++index) {
        std::vector<std::vector<double>> F_BRANCH(dim_x, std::vector<double>(Tf+1));
        for (int i = 0; i < dim_x; ++i) {
            for (int j = 0; j < Tf + 1; ++j) {
                F_BRANCH[i][j] = Xf(index * dim_x + i, j);
            }
        }
        std::string color = "C" + std::to_string(index%10);
        plt::plot(F_BRANCH[0], F_BRANCH[1], {{"color", color}, {"linewidth", "10.0"}});
    }

    for (int index = 0; index < clusters_b.size(); ++index) {
        std::vector<std::vector<double>> B_BRANCH(dim_x, std::vector<double>(Tb+1));
        for (int i = 0; i < dim_x; ++i) {
            for (int j = 0; j < Tb + 1; ++j) {
                B_BRANCH[i][j] = Xb(index * dim_x + i, j);
            }
        }
        std::string color = "C" + std::to_string(index%10);
        plt::plot(B_BRANCH[0], B_BRANCH[1], {{"color", color}, {"linewidth", "10.0"}});
    }
    
    // // plt::xlim(0, 3);
    // // plt::ylim(0, 5);
    // // plt::grid(true);
    // // // plt::show();
    
    // plt::subplot(1,2,2);
    // for (int i = 0; i < collision_checker->map.size(); ++i) {
    //     for (int j = 0; j < collision_checker->map[0].size(); ++j) {
    //         if ((collision_checker->map[i])[j] == 10) {
    //             double mx = i*resolution;
    //             double my = j*resolution;
    //             std::vector<double> oX = {mx-hl, mx+hl, mx+hl, mx-hl, mx-hl};
    //             std::vector<double> oY = {my-hl,my-hl,my+hl,my+hl,my-hl};
    //             plt::plot(oX, oY, "k");
    //         }
    //     }
    // }
    std::vector<std::vector<double>> X_MPPI(dim_x, std::vector<double>(Xo.cols()));
    for (int i = 0; i < dim_x; ++i) {
        for (int j = 0; j < Xo.cols(); ++j) {
            X_MPPI[i][j] = Xo(i, j);
        }
    }
    // std::string color = "C" + std::to_string(9 - index%10);
    plt::plot(X_MPPI[0], X_MPPI[1], {{"color", "black"}, {"linewidth", "10.0"}});

    plt::xlim(0, 3);
    plt::ylim(0, 5);
    plt::grid(true);
    plt::show();
}

void BiMPPI::showTraj() {
    namespace plt = matplotlibcpp;

    // --- 0) (선택) 이전 그림 지우기 ---
    // plt::clf();

    // --- 1) 장애물 그리기: 검은 윤곽선 "k-" ---
    const double res = 0.1;
    const double hl  = res / 2.0;
    for (int i = 0; i < static_cast<int>(collision_checker->map.size()); ++i) {
        for (int j = 0; j < static_cast<int>(collision_checker->map[0].size()); ++j) {
            if (collision_checker->map[i][j] == 10) {
                const double mx = i * res;
                const double my = j * res;
                std::vector<double> oX = {mx - hl, mx + hl, mx + hl, mx - hl, mx - hl};
                std::vector<double> oY = {my - hl, my - hl, my + hl, my + hl, my - hl};
                plt::plot(oX, oY, "k-");
            }
        }
    }

    // --- 2) 경로(visual_traj) 그리기: 파란색, linewidth=1.0 ---
    if (!visual_traj.empty()) {
        std::vector<double> xs, ys;
        xs.reserve(visual_traj.size());
        ys.reserve(visual_traj.size());
        for (const auto& x : visual_traj) {
            // dim_x >= 2 가정: x(0)=x좌표, x(1)=y좌표
            xs.push_back(x(0));
            ys.push_back(x(1));
        }
        plt::plot(xs, ys, { {"color","blue"}, {"linewidth","1.0"} });
    }

    // --- 3) 축 설정: 맵 크기에 맞게, 등축, 그리드 ---
    const double xmax = collision_checker->map[0].size() * res;
    const double ymax = collision_checker->map.size()    * res;
    plt::xlim(0.0, xmax);
    plt::ylim(0.0, ymax);
    plt::axis("equal");
    plt::grid(true);

    // --- 4) 표시 ---
    plt::show();
}


inline double BiMPPI::pathLengthXY(double eps_ignore) const
{
    double L = 0.0;
    const size_t n = visual_traj.size();
    if (n < 2) return 0.0;

    // 차원 체크 (x,y 존재 확인)
    // visual_traj[0]는 Eigen::Matrix<double, -1, 1> 타입이므로, size() 대신 rows()를 사용합니다.
    if (visual_traj[0].rows() < 2) return 0.0;

    auto finite = [](double v) { return std::isfinite(v); };

    // visual_traj는 std::vector 이므로 []로 접근하고, 그 안에 Eigen::Matrix(Eigen Vector)이므로 ()로 접근합니다.
    double px = visual_traj[0](0); // 첫 번째 점의 x좌표
    double py = visual_traj[0](1); // 첫 번째 점의 y좌표

    for (size_t i = 1; i < n; ++i) {
        // 방어: 차원이 2 미만인 점이 섞여 있으면 스킵
        if (visual_traj[i].rows() < 2) continue; // 마찬가지로 size() 대신 rows() 사용

        double cx = visual_traj[i](0); // 현재 점의 x좌표
        double cy = visual_traj[i](1); // 현재 점의 y좌표

        // NaN/Inf 방어
        if (!finite(px) || !finite(py) || !finite(cx) || !finite(cy)) {
            px = cx; py = cy;
            continue;
        }

        double dx = cx - px;
        double dy = cy - py;
        double d  = std::hypot(dx, dy);

        // 너무 작은 진동 제거(옵션)
        if (d > eps_ignore) L += d;

        px = cx; py = cy;
    }
    return L;
}



inline void BiMPPI::savePathToCSV(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }
    for (const auto& p : visual_traj) {
        if (p.size() >= 2 && std::isfinite(p(0)) && std::isfinite(p(1))) {
            file << p(0) << "," << p(1) << "," << p(2) << "\n";
        }
    }
    // file.close()는 생략 가능
}