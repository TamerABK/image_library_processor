# Blockwise unknown-face clustering memory

## Scope

The source revision inspected before editing was `34267d3fe7459dd3aa5047f6c144899a04f75308`. The implementation commit is `b26ec94ff1257332cbfae8ca21ecf3df863a3043`. This changes only unknown-face clustering's temporary similarity calculation. The 16,000-face and 4,000,000-edge ceilings, thresholds, recognition, ranking, models, and stored identities are unchanged. The Windows CI run is in the delivery report.

## Allocation policy

The old float32 matrix required `4 * N * N` bytes for N faces. The new similarity block requires at most `4 * min(256, floor(16 MiB / (4N))) * N` bytes, with a minimum of one row. The actual block can be smaller because only columns from the current block onward are computed; earlier columns cannot contain `j > i` edges. The embedding matrix remains `4 * N * D` bytes. Representative selection computes cluster similarity sums in blocks and chooses the first maximum, matching the previous tie rule. It never allocates a full cluster submatrix.

At 8,219 faces, the old matrix is 257.7 MiB and the default maximum block is 8.0 MiB. A 4 MiB target gives 127 rows. The 16 MiB setting is a maximum buffer allowance, not preallocated RAM. The graph still uses Python adjacency lists and weak-edge tuples, and may grow substantially until the unchanged edge ceiling. The algorithm remains quadratic in comparison count. Its total memory is not linear because of the graph.

The profiler records calculated NumPy buffer sizes, block count, edge counts, largest component, timing, and before/after process RSS and private bytes where available. Before/after process samples are not exact allocation measurements. The isolated benchmark samples process memory every 5 ms.

## Real cached embeddings

The microbenchmark read 8,219 NSC Meeting cached ArcFace embeddings from `image_analysis_cache.sqlite3` in SQLite read-only mode. It did not decode images, run ONNX inference, or modify the project or cache. Each variant ran three times in fresh processes with no concurrent test suite. The table gives medians. The dense reference and all block variants had the same ordered membership and representative SHA-256. All produced 978 clusters, 233,756 strong edges, and 207,220 weak edges. The largest component had 405 faces. This verifies the actual cached embeddings, not a new full scan.

| Method | Effective rows | Similarity buffer MiB | Sampled peak RSS MiB | Total seconds | Similarity seconds | Representative seconds |
|---|---:|---:|---:|---:|---:|---:|
| Dense reference | 8,219 | 257.7 | 412.6 | 0.257 | 0.107 | 0.015 |
| Blockwise | 64 | 2.0 | 147.1 | 0.284 | 0.086 | 0.017 |
| Blockwise | 128 | 4.0 | 147.5 | 0.271 | 0.076 | 0.013 |
| Blockwise, default | 256 | 8.0 | 147.3 | 0.253 | 0.065 | 0.007 |
| Blockwise, 4 MiB target | 127 | 4.0 | 146.4 | 0.277 | 0.079 | 0.015 |

An initial blockwise implementation multiplied every block against all embeddings and took 1.42 seconds at 256 rows. Skipping earlier columns, which cannot contribute `j > i` edges, brought the median to 0.253 seconds, close to the dense reference's 0.257 seconds. Timing varied across runs, including one 0.689-second 64-row outlier, so this is not a guaranteed speed advantage. The previous full cold scan was about 693 seconds, but no new full-scan impact was measured. The isolated sampled peak RSS median difference was 265.3 MiB. Process allocation and 5 ms sampling affect this measurement, so it is not an exact NumPy allocation measurement.

## Synthetic scaling

Deterministic normalized 512-dimensional random embeddings avoided creating a dense graph. These single runs establish buffer scaling, not a reliable speed curve.

| Faces | Embedding MiB | Max block MiB | Graph edges | Sampled peak RSS MiB | Seconds |
|---:|---:|---:|---:|---:|---:|
| 1,000 | 2.0 | 1.0 | 0 | 69.3 | 0.013 |
| 4,000 | 7.8 | 3.9 | 0 | 96.3 | 0.049 |
| 8,000 | 15.6 | 7.8 | 0 | 121.3 | 0.132 |
| 12,000 | 23.4 | 11.7 | 0 | 152.8 | 0.258 |
| 16,000 | 31.3 | 15.6 | 0 | 180.0 | 0.360 |

## Numerical compatibility and limitations

Blocks process rows and edges in original index order. Strong graph, weak-edge order, component traversal, weak attachment, cluster ordering, and representative objective are unchanged. Tests use a dense float32 reference with independent edge construction and cover block sizes 1, 2, 16, 64, 128, and 256, threshold boundaries, uneven final blocks, cancellation, and the edge ceiling. No disagreement was observed on the tested inputs or 8,219 real cached vectors. A different BLAS implementation could differ at a threshold or near a representative tie because matrix shapes can alter float32 rounding. No arbitrary threshold rounding was introduced.

The final benchmark data are in `audit_results/cluster_blocks_final_20261105/`. Reproduce with `python -B -m app.tools.benchmark_cluster_blocks --cache image_analysis_cache.sqlite3 --out audit_results/cluster_blocks_<date>` using the project's Python environment. The source cache must contain the NSC Meeting embeddings. This is a clustering-only benchmark.
