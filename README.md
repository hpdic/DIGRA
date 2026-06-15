# DIGRA — churn-maintenance benchmark fork, by HPDIC

This fork of [CUHK-DBGroup/DIGRA](https://github.com/CUHK-DBGroup/DIGRA)
(Jiang et al., *DIGRA: A Dynamic Graph Indexing for Approximate Nearest
Neighbor Search with Range Filter*, SIGMOD 2025) adds a **dynamic churn
benchmark** so DIGRA — and the vanilla hnswlib it is built on — can be compared
against ETALE on identical insert/delete workloads.

The upstream `main.cpp` only measures range-filtered query throughput; it never
calls the incremental `addPoint`/`erase` path. The drivers here exercise that
update path directly under a streaming-churn protocol.

## What was added

| file | purpose |
|------|---------|
| `digra_churn.cpp` | drives DIGRA's `addPoint`/`erase` under a churn workload, times per-round maintenance, measures Recall@10 over the live set, writes a CSV |
| `hnsw_churn.cpp`  | same workload on **vanilla hnswlib** (`addPoint`/`markDelete`) — the pure-vector CPU dynamic baseline, and a sanity check on the DIGRA repro |
| `CMakeLists.txt`  | builds both drivers into `bin/`; **no forced `-mavx512f`** (see Gotchas) |
| `.gitignore`      | ignores `bin/`, CMake artifacts, `results/`, `*.csv`, the `data` symlink |

## Build

```bash
rm -rf CMakeCache.txt CMakeFiles    # if you changed CMakeLists, clear the cache
cmake . && make
# binaries land in ./bin
```

Requires a C++17 compiler with OpenMP. The upstream build forced AVX-512
(`-mavx512f`); this fork does not — see Gotchas.

## Data

Point the drivers at standard `.fvecs` files (the SIFT1M base + query set).
Easiest is to symlink an existing data dir:

```bash
ln -s /path/to/sift/data ./data       # must contain sift_base.fvecs, sift_query.fvecs
mkdir -p results
```

The loader reads `.fvecs` (4-byte little-endian dim header per row, then
`dim` float32s). It does **not** need the `.data` key-value file that upstream
`main.cpp` uses — the churn drivers assign `value = key = global index` so the
internal B-tree stays well-formed without external attributes.

## Run the churn benchmark

Common arguments (both drivers, in order):

```
base.fvecs  query.fvecs  N  dim  active  churn  rounds  M  ef_con  out.csv
```

- `N`       total base vectors available (e.g. 1000000 for SIFT1M)
- `active`  size of the initial index (working set)
- `churn`   points deleted + inserted per round (1% churn = active/100)
- `rounds`  number of churn rounds
- `M`       graph degree (use 8 for DIGRA's recommended config; 32 to match
            an out-degree-32 graph)
- `ef_con`  construction beam width

### DIGRA (recommended config, M=8)

```bash
./bin/digra_churn data/sift_base.fvecs data/sift_query.fvecs \
    1000000 128 200000 2000 5 8 100 results/churn_digra_m8.csv
```

### DIGRA (out-degree-matched, M=32)

```bash
./bin/digra_churn data/sift_base.fvecs data/sift_query.fvecs \
    1000000 128 200000 2000 5 32 100 results/churn_digra_m32.csv
```

### vanilla hnswlib (same workload)

```bash
./bin/hnsw_churn data/sift_base.fvecs data/sift_query.fvecs \
    1000000 128 200000 2000 5 8 100 results/churn_hnsw_m8.csv
./bin/hnsw_churn data/sift_base.fvecs data/sift_query.fvecs \
    1000000 128 200000 2000 5 32 100 results/churn_hnsw_m32.csv
```

Each run prints a per-round table and writes a CSV with columns
`round,maintenance_ms,recall,live`.

## Reference numbers (SIFT1M, 200k active, 1% churn, 5 rounds, A100 host)

| system        | platform | maintenance/round | Recall@10 |
|---------------|----------|-------------------|-----------|
| DIGRA (M=8)   | CPU      | ~4940 ms          | ~0.975    |
| DIGRA (M=32)  | CPU      | ~7210 ms          | ~0.992    |
| hnswlib (M=8) | CPU      | ~282 ms           | ~0.994    |
| hnswlib (M=32)| CPU      | ~423 ms           | ~0.9996   |

Notes:
- **DIGRA is ~17x slower than the vanilla hnswlib it builds on.** That gap is
  the B-tree overlay (which serves range-filtered queries, unused here). This
  ratio is the sanity check: a range-filter system should not beat its own bare
  HNSW on pure churn, and it doesn't.
- **hnswlib's `markDelete` is a lazy tombstone**: it flips a bit, does not
  repair connectivity, and never reclaims memory. Its low maintenance time
  reflects deferred (not avoided) deletion cost — memory grows monotonically
  under sustained churn. Keep this in mind when reading its "fast" numbers.

## Gotchas (hit during setup)

1. **`Illegal instruction (core dumped)` at startup** — upstream `CMakeLists`
   forces `-mavx512f`. If the host CPU lacks AVX-512, the binary SIGILLs before
   `main`. This fork drops the explicit flag and relies on `-march=native`,
   which enables exactly what the CPU supports. After changing CMake flags you
   must `rm -rf CMakeCache.txt CMakeFiles` or the old flags persist.

2. **`open file error`** — the loader uses the path you pass verbatim. Run from
   the repo root with a `data/` symlink, or pass absolute paths.

3. **`INT_MIN/INT_MAX not declared`** — needs `#include <climits>` (already in
   the drivers here).

4. **`erase` is barely exercised upstream.** `main.cpp` never calls it, so the
   B-tree delete/merge/refresh path is lightly tested. Large-scale random
   deletion can be slow and its per-round time is bursty (periodic B-tree
   restructuring); report mean *and* range, not a single round.

5. **`M=32` inflates DIGRA's edge count** (~207 edges/node observed, vs ~85 at
   M=8) because edges are built per B-tree layer. Report both M=8 (DIGRA's
   recommended config) and M=32 (out-degree-matched) for a fair picture.



# DIGRA

## Quick Start

### Compile and Run

```bash
cmake . && make
```

Running example benchmark on SIFT dataset:
```bash
./Digra 10000 1000 128 10 100 8 [path_to_sift_base.fvecs] [path_to_sift_query.fvecs] [path_to_sift_base.data]
```

Parameter description:

1. `10000`: Number of base vectors
2. `1000`: Number of query vectors
3. `128`: Vector dimension
4. `10`: Number of nearest neighbors to return
5. `100`: The size of result set during index building
6. `8`: The degree of the graph index 
7. `[path_to_sift_base.fvecs]`: Path to the SIFT base vectors file
8. `[path_to_sift_query.fvecs]`: Path to the SIFT query vectors file
9. `[path_to_sift_base.data]`: Path to the SIFT base data file

The `.data` file is a text file containing key-value pairs. Each line represents an entry in the format:

```
key value
```
