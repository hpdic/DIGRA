/**
 * hnsw_churn.cpp — vanilla hnswlib dynamic-maintenance benchmark.
 *
 * Identical churn protocol to digra_churn.cpp (same SIFT1M, same active/churn/
 * rounds, same brute-force GT over the live set, same timing boundaries) so
 * ETALE / DIGRA / vanilla-HNSW maintenance numbers are directly comparable.
 *
 * vanilla HNSW dynamic path:
 *   - insert: addPoint(vec, label)
 *   - delete: markDelete(label)   (lazy; hnswlib's native deletion)
 *   - search: searchKnn over the live set
 *
 * This is the cleanest pure-vector CPU dynamic baseline (Malkov & Yashunin,
 * TPAMI 2020). It carries NO range-filter / B-tree overhead, unlike DIGRA, so
 * comparing DIGRA-vs-HNSW here is also a sanity check on the DIGRA repro: a
 * range-filter system should not be faster than the bare HNSW it is built on.
 *
 * Build: g++ -O3 -march=native -fopenmp -I<hnswlib_parent> hnsw_churn.cpp -o hnsw_churn
 *   where <hnswlib_parent> contains the hnswlib/ header dir (DIGRA ships one).
 * Run:   ./hnsw_churn base.fvecs query.fvecs N dim active churn rounds M ef_con out.csv
 */

#include "hnswlib/hnswlib.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_set>
#include <algorithm>
#include <random>
#include <chrono>
#include <cstring>
#include <cstdlib>

// fvecs loader matching DIGRA's utils.hpp semantics (4-byte dim header per row)
static float* load_fvecs(const char* path, int& num, int dim) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) { std::cerr << "open file error: " << path << "\n"; exit(1); }
    int d = 0; in.read((char*)&d, 4);
    in.seekg(0, std::ios::end);
    size_t fsize = (size_t)in.tellg();
    num = (int)(fsize / ((dim + 1) * 4));
    float* data = new float[(size_t)num * dim];
    in.seekg(0, std::ios::beg);
    for (int i = 0; i < num; ++i) {
        in.seekg(4, std::ios::cur);                       // skip per-row dim
        in.read((char*)(data + (size_t)i * dim), dim * 4);
    }
    in.close();
    return data;
}

static double eval_recall(const float* base, const float* query,
                          const std::vector<int>& live,
                          hnswlib::HierarchicalNSW<float>* index,
                          int dim, int nq, int k) {
    double rs = 0.0;
    for (int qi = 0; qi < nq; ++qi) {
        const float* q = query + (size_t)qi * dim;
        std::vector<std::pair<float,int>> gt; gt.reserve(live.size());
        for (int id : live) {
            const float* v = base + (size_t)id * dim;
            float s = 0; for (int d = 0; d < dim; ++d){ float t=q[d]-v[d]; s+=t*t; }
            gt.push_back({s, id});
        }
        std::partial_sort(gt.begin(), gt.begin()+k, gt.end());
        std::unordered_set<int> gtset;
        for (int j = 0; j < k; ++j) gtset.insert(gt[j].second);

        auto res = index->searchKnn(q, k);   // labels of approx top-k
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
    int N      = std::atoi(argv[3]);
    int dim    = std::atoi(argv[4]);
    int active = std::atoi(argv[5]);
    int churn  = std::atoi(argv[6]);
    int rounds = std::atoi(argv[7]);
    int M      = std::atoi(argv[8]);
    int ef_con = std::atoi(argv[9]);
    const char* outCsv = argv[10];
    int k = 10, nq = 1000, ef_search = 300;

    int cap = active + rounds * churn;
    if (cap > N) { std::cerr << "need active+rounds*churn <= N\n"; return 1; }

    int nb = 0, nqf = 0;
    float* base = load_fvecs(baseFile, nb, dim);
    float* query = load_fvecs(queryFile, nqf, dim);
    nq = std::min(nq, nqf);

    std::cout << "HNSW churn: N=" << N << " dim=" << dim
              << " active=" << active << " churn=" << churn
              << " rounds=" << rounds << " M=" << M << " ef_con=" << ef_con << "\n";

    hnswlib::L2Space space(dim);
    // max_elements = cap so we never need to resize during churn.
    auto* index = new hnswlib::HierarchicalNSW<float>(
        &space, cap, M, ef_con);
    index->setEf(ef_search);

    // initial build on first `active` points
    auto tb0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < active; ++i)
        index->addPoint(base + (size_t)i * dim, i);
    auto tb1 = std::chrono::high_resolution_clock::now();
    std::cout << "initial build: "
              << std::chrono::duration<double>(tb1 - tb0).count()
              << " s on " << active << " points\n";

    std::unordered_set<int> liveSet;
    for (int i = 0; i < active; ++i) liveSet.insert(i);
    int next_free = active;
    auto live_vec = [&]{ return std::vector<int>(liveSet.begin(), liveSet.end()); };

    std::ofstream csv(outCsv);
    csv << "round,maintenance_ms,recall,live,used_slabs\n";

    double r0 = eval_recall(base, query, live_vec(), index, dim, nq, k);
    std::cout << "initial recall@10 = " << r0 << " (live=" << liveSet.size() << ")\n";
    // cur_element_count = HNSW's structural footprint: live + accumulated
    // tombstones (markDelete never reclaims). Analogous to ETALE's used_slabs.
    csv << "0,0.0," << r0 << "," << liveSet.size() << ","
        << index->cur_element_count << "\n";

    std::mt19937 rng(123);
    std::cout << "\nround   maintenance_ms   recall@10   live\n";
    double sum_ms = 0, sum_r = 0;

    for (int rd = 1; rd <= rounds; ++rd) {
        std::vector<int> lv = live_vec();
        std::shuffle(lv.begin(), lv.end(), rng);
        std::vector<int> del;
        for (int id : lv) { if (id == 0) continue; del.push_back(id); if ((int)del.size() >= churn) break; }
        int ins0 = next_free, n = std::min(churn, N - next_free);

        // ---- TIMED maintenance: markDelete `churn` then addPoint `churn` ----
        auto m0 = std::chrono::high_resolution_clock::now();
        for (int id : del) index->markDelete(id);
        for (int i = ins0; i < ins0 + n; ++i)
            index->addPoint(base + (size_t)i * dim, i);
        auto m1 = std::chrono::high_resolution_clock::now();
        double maint_ms = std::chrono::duration<double, std::milli>(m1 - m0).count();

        for (int id : del) liveSet.erase(id);
        for (int i = ins0; i < ins0 + n; ++i) liveSet.insert(i);
        next_free += n;

        double r = eval_recall(base, query, live_vec(), index, dim, nq, k);
        sum_ms += maint_ms; sum_r += r;
        printf("%5d   %14.2f   %9.4f   %zu\n", rd, maint_ms, r, liveSet.size());
        csv << rd << "," << maint_ms << "," << r << "," << liveSet.size() << ","
            << index->cur_element_count << "\n";
    }

    csv.close();
    printf("\nmean maintenance = %.2f ms/round   mean recall@10 = %.4f\n",
           sum_ms / rounds, sum_r / rounds);
    printf("CSV: %s\n", outCsv);
    return 0;
}
