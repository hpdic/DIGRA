/**
 * digra_churn.cpp — DIGRA dynamic-maintenance benchmark, aligned to ETALE's
 * bench_churn workload for the ETALE-vs-DIGRA comparison.
 *
 * DIGRA (Jiang et al., SIGMOD 2025) is a dynamic graph index built on hnswlib
 * with a B-tree overlay for range-filtered ANNS. This driver exercises its
 * INCREMENTAL UPDATE path (addPoint / erase) under the same churn protocol
 * ETALE uses: build on `active` points, then R rounds of (delete `churn`
 * random live points + insert `churn` fresh points), timing the per-round
 * maintenance and measuring Recall@10 over the live set.
 *
 * Alignment notes (see conversation): DIGRA's update path always maintains its
 * value-sorted B-tree. To keep that tree well-formed (not degenerate), each
 * point gets a UNIQUE value = its global index. We do not use DIGRA's range
 * filter (queries cover the full range), so this measures DIGRA's pure
 * dynamic-maintenance cost with its native update machinery intact. This is
 * the fairest alignment that does not amputate DIGRA's own update logic.
 *
 * Build: see CMakeLists in this dir (links DIGRA's hnswlib + TreeHNSW.hpp).
 * Run:   ./digra_churn <base.fvecs> <query.fvecs> <N> <dim> <active> <churn>
 *                      <rounds> <M> <ef_con> <out.csv>
 */

#include "TreeHNSW.hpp"
#include "utils.hpp"

#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_set>
#include <algorithm>
#include <random>
#include <chrono>
#include <cstring>
#include <climits>

static int stringTonum(const char* ch){ return std::atoi(ch); }

// brute-force Recall@10 over the current live set (not timed)
static double eval_recall(const float* base, const float* query,
                          const std::vector<int>& live,
                          RangeHNSW& index, int dim, int nq, int k, int ef) {
    // host GT over the live set
    double rs = 0.0;
    for (int qi = 0; qi < nq; ++qi) {
        const float* q = query + (size_t)qi * dim;
        // exact top-k over live
        std::vector<std::pair<float,int>> gt;
        gt.reserve(live.size());
        for (int id : live) {
            const float* v = base + (size_t)id * dim;
            float s = 0;
            for (int d = 0; d < dim; ++d) { float t = q[d]-v[d]; s += t*t; }
            gt.push_back({s, id});
        }
        std::partial_sort(gt.begin(), gt.begin()+k, gt.end());
        std::unordered_set<int> gtset;
        for (int j = 0; j < k; ++j) gtset.insert(gt[j].second);

        // DIGRA query: full range [min_value, max_value] so the filter is a no-op.
        // queryRange returns keys (we used key==value==global id at insert).
        auto res = index.queryRange(const_cast<float*>(q),
                                    INT_MIN/2, INT_MAX/2, k, ef);
        int hit = 0;
        while (!res.empty()) {
            int id = (int)res.top().second; res.pop();
            if (gtset.count(id)) ++hit;
        }
        rs += (double)hit / k;
    }
    return rs / nq;
}

int main(int argc, char** argv) {
    if (argc < 11) {
        std::cerr << "usage: " << argv[0]
                  << " base.fvecs query.fvecs N dim active churn rounds M ef_con out.csv\n";
        return 1;
    }
    const char* baseFile = argv[1];
    const char* queryFile = argv[2];
    int N      = stringTonum(argv[3]);
    int dim    = stringTonum(argv[4]);
    int active = stringTonum(argv[5]);
    int churn  = stringTonum(argv[6]);
    int rounds = stringTonum(argv[7]);
    int M      = stringTonum(argv[8]);
    int ef_con = stringTonum(argv[9]);
    const char* outCsv = argv[10];
    int k = 10, nq = 1000, ef_search = 300;

    int cap = active + rounds * churn;
    if (cap > N) { std::cerr << "need active+rounds*churn <= N\n"; return 1; }

    // load vectors
    float* base = nullptr; float* query = nullptr;
    load_data(baseFile, base, N, dim);
    load_data(queryFile, query, /*num*/0, dim);
    nq = std::min(nq, 1000);

    std::cout << "DIGRA churn: N=" << N << " dim=" << dim
              << " active=" << active << " churn=" << churn
              << " rounds=" << rounds << " M=" << M << " ef_con=" << ef_con << "\n";

    // value = key = global index (unique => well-formed B-tree, full-range query = no-op filter)
    std::vector<int> keyArr(cap), valArr(cap);
    for (int i = 0; i < cap; ++i) keyArr[i] = valArr[i] = i;

    // build initial index on first `active` points.
    // maxEleNum = cap so addPoint never overflows across all churn rounds.
    auto t_build0 = std::chrono::high_resolution_clock::now();
    RangeHNSW index(dim, /*eleNum*/active, /*maxEleNum*/cap,
                    base, keyArr.data(), valArr.data(), M, ef_con);
    auto t_build1 = std::chrono::high_resolution_clock::now();
    double build_s = std::chrono::duration<double>(t_build1 - t_build0).count();
    std::cout << "initial build: " << build_s << " s on " << active << " points\n";

    std::unordered_set<int> liveSet;
    for (int i = 0; i < active; ++i) liveSet.insert(i);
    int next_free = active;

    std::ofstream csv(outCsv);
    csv << "round,maintenance_ms,recall,live\n";

    auto live_vec = [&]() {
        return std::vector<int>(liveSet.begin(), liveSet.end());
    };

    double r0 = eval_recall(base, query, live_vec(), index, dim, nq, k, ef_search);
    std::cout << "initial recall@10 = " << r0 << " (live=" << liveSet.size() << ")\n";
    csv << "0,0.0," << r0 << "," << liveSet.size() << "\n";

    std::mt19937 rng(123);
    std::cout << "\nround   maintenance_ms   recall@10   live\n";
    double sum_ms = 0, sum_r = 0;

    for (int rd = 1; rd <= rounds; ++rd) {
        // pick `churn` random live ids to delete (never id 0, keep an anchor)
        std::vector<int> lv = live_vec();
        std::shuffle(lv.begin(), lv.end(), rng);
        std::vector<int> del;
        for (int id : lv) { if (id == 0) continue; del.push_back(id); if ((int)del.size() >= churn) break; }

        int ins0 = next_free, n = std::min(churn, N - next_free);

        // ---- TIMED maintenance: erase `churn` then addPoint `churn` ----
        auto m0 = std::chrono::high_resolution_clock::now();
        for (int id : del) index.erase(/*key*/id);
        for (int i = ins0; i < ins0 + n; ++i) {
            index.addPoint(/*key*/i, /*value*/i,
                           (char*)(base + (size_t)i * dim));
        }
        auto m1 = std::chrono::high_resolution_clock::now();
        double maint_ms = std::chrono::duration<double, std::milli>(m1 - m0).count();

        for (int id : del) liveSet.erase(id);
        for (int i = ins0; i < ins0 + n; ++i) liveSet.insert(i);
        next_free += n;

        double r = eval_recall(base, query, live_vec(), index, dim, nq, k, ef_search);
        sum_ms += maint_ms; sum_r += r;
        printf("%5d   %14.2f   %9.4f   %zu\n", rd, maint_ms, r, liveSet.size());
        csv << rd << "," << maint_ms << "," << r << "," << liveSet.size() << "\n";
    }

    csv.close();
    printf("\nmean maintenance = %.2f ms/round   mean recall@10 = %.4f\n",
           sum_ms / rounds, sum_r / rounds);
    printf("CSV: %s\n", outCsv);
    return 0;
}