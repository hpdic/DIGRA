/**
 * hnsw_reclaim_probe.cpp — does hnswlib reclaim deleted slots, or accumulate?
 *
 * The whole memory-footprint argument hinges on this: if markDelete + addPoint
 * REUSES freed slots, HNSW memory stays bounded and our "lazy delete leaks
 * memory" claim is wrong. If it does NOT reuse (pure tombstone), every insert
 * grows the structure and memory climbs monotonically under churn.
 *
 * This probe builds a small index, then runs churn rounds (delete K, insert K)
 * and prints, each round:
 *   - cur_element_count : total slots ever used (live + tombstoned)
 *   - deleted count     : tombstoned slots
 *   - max_elements      : capacity
 *
 * If cur_element_count climbs while live stays flat -> tombstones accumulate
 * (our claim holds). If cur_element_count stays flat -> hnswlib reuses slots.
 *
 * Build: g++ -O3 -march=native -fopenmp -I. hnsw_reclaim_probe.cpp -o hnsw_reclaim_probe
 * Run:   ./hnsw_reclaim_probe
 */

#include "hnswlib/hnswlib.h"
#include <iostream>
#include <vector>
#include <random>

int main() {
    int dim = 16, base = 1000, churn = 200, rounds = 10;
    int cap = base + rounds * churn;   // generous capacity

    hnswlib::L2Space space(dim);
    auto* index = new hnswlib::HierarchicalNSW<float>(&space, cap, 16, 100);

    std::mt19937 rng(0);
    std::uniform_real_distribution<float> ud(0, 1);
    auto randvec = [&](std::vector<float>& v){ for (auto& x : v) x = ud(rng); };

    // build
    std::vector<int> live;
    for (int i = 0; i < base; ++i) {
        std::vector<float> v(dim); randvec(v);
        index->addPoint(v.data(), i);
        live.push_back(i);
    }
    int next = base;

    std::cout << "round  cur_element_count  deleted  live  max_elements\n";
    auto report = [&](int rd){
        // cur_element_count and deleted count via public hnswlib API
        size_t cur = index->cur_element_count;
        size_t del = index->getDeletedCount();
        std::cout << rd << "      " << cur
                  << "                 " << del
                  << "      " << live.size()
                  << "     " << index->max_elements_ << "\n";
    };
    report(0);

    for (int rd = 1; rd <= rounds; ++rd) {
        // delete `churn` random live, insert `churn` fresh
        std::shuffle(live.begin(), live.end(), rng);
        for (int j = 0; j < churn; ++j) {
            index->markDelete(live[j]);
        }
        live.erase(live.begin(), live.begin() + churn);
        for (int j = 0; j < churn; ++j) {
            std::vector<float> v(dim); randvec(v);
            index->addPoint(v.data(), next);
            live.push_back(next);
            ++next;
        }
        report(rd);
    }

    std::cout << "\nInterpretation:\n";
    std::cout << "  cur_element_count climbs while live flat -> tombstones "
                 "accumulate, memory leaks (our claim HOLDS).\n";
    std::cout << "  cur_element_count stays flat -> hnswlib reuses slots "
                 "(claim needs rethinking).\n";
    return 0;
}