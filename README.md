# PyMatching 2

![Continuous Integration](https://github.com/oscarhiggott/PyMatching/workflows/ci/badge.svg)
[![codecov](https://codecov.io/gh/oscarhiggott/PyMatching/branch/master/graph/badge.svg)](https://codecov.io/gh/oscarhiggott/PyMatching)
[![docs](https://readthedocs.org/projects/pymatching/badge/?version=latest&style=plastic)](https://readthedocs.org/projects/pymatching/builds/)
[![PyPI version](https://badge.fury.io/py/PyMatching.svg)](https://badge.fury.io/py/PyMatching)
[![Unitary Fund](https://img.shields.io/badge/Supported%20By-UNITARY%20FUND-brightgreen.svg?style=for-the-badge)](http://unitary.fund)

PyMatchingは、量子誤り訂正（QEC）符号の復号のための高速なPython/C++ライブラリで、MWPM（Minimum Weight Perfect Matching、最小重み完全マッチング問題）デコーダを使用しています。
量子誤り訂正回路からのシンドローム測定値が与えられると、
MWPMデコーダは、誤りの発生は独立であり、かつグラフ的であるという仮定（各誤りは1つまたは2つの検出イベントを引き起こす）のもと、最も可能性の高い誤り箇所の集合を探索します。
MWPMデコーダは、[表面符号](https://arxiv.org/abs/quant-ph/0110143)を復号するための最も一般的なデコーダです。
また、
[サブシステム符号](https://arxiv.org/abs/1207.1443)、
[ハニカム符号](https://quantum-journal.org/papers/q-2021-10-19-564/)、
[2次元双曲線符号](https://arxiv.org/abs/1506.04029)
といった、他のさまざまな符号のデコードに使用できます。

バージョン2にはblossomアルゴリズムの新しい実装が含まれており、以前のバージョンよりも**100-1000倍**高速になりました。
PyMatching は、境界のあるなしに関わらず、任意の重み付きグラフを使用して設定することができ、
Craig Gidneyの[Stim](https://github.com/quantumlib/Stim)ライブラリと組み合わせて、回路レベルのノイズがある場合のエラー訂正回路のシミュレーションとデコードを行うことができます。
と組み合わせて、回路レベルのノイズがある場合のエラー訂正回路をシミュレートし、デコードすることができます。[sinter](https://pypi.org/project/sinter/) パッケージでは、Stim と 
PyMatching を組み合わせて、量子エラー訂正回路のモンテカルロ・サンプリングを高速・並列に実行します。

Documentation for PyMatching can be found at: [pymatching.readthedocs.io](https://pymatching.readthedocs.io/en/stable/)

To see how stim, sinter and pymatching can be used to estimate the threshold of an error correcting code with 
circuit-level noise, try out the [stim getting started notebook](https://github.com/quantumlib/Stim/blob/main/doc/getting_started.ipynb).

## Version 2の100倍以上高速な新実装について

Version 2 features a new implementation of the blossom algorithm, which I wrote with Craig Gidney.
Our new implementation, which we refer to as the _sparse blossom_ algorithm, can be seen as a generalisation of the 
blossom algorithm to handle the decoding problem relevant to QEC. 
We solve the problem of finding minimum-weight paths between detection events in a detector graph 
_directly_, which avoids the need to use costly all-to-all Dijkstra searches to find a MWPM in a derived 
graph using the original blossom algorithm.
The new version is also exact - unlike previous versions of PyMatching, no approximation is made.

Our new implementation is **over 100x faster** than previous versions of PyMatching, and is 
**over 100,000x faster** than NetworkX (benchmarked with surface code circuits). At 0.1% circuit-noise, PyMatching can 
decode both X and Z basis measurements of surface code circuits up to distance 13 in under 1 microsecond per round 
of syndrome extraction on a single core (or up to distance 19 if only X-basis measurements are processed - however 
both X and Z basis measurements must be decoded at scale). Furthermore, the runtime is roughly linear in the number 
of nodes in the graph.

The plot below compares the performance of PyMatching v2 with the previous 
version (v0.7) as well as with NetworkX for decoding surface code circuits with circuit-level depolarising noise. 
All decoders were run on a single core of an M1 processor, processing both the X and Z basis measurements.
The equations T=N^x in the legend (and plotted as dashed lines) are 
obtained from a fit to the same dataset for 
distance > 10, where N is the number of detectors (nodes) per round, and T is the decoding time per round.
See the [benchmarks](https://github.com/oscarhiggott/PyMatching/raw/master/benchmarks) folder in the repository 
for the data and stim circuits, as well as additional benchmarks.


![PyMatching new vs old vs NetworkX](https://github.com/oscarhiggott/PyMatching/raw/master/benchmarks/surface_codes/surface_code_rotated_memory_x_p_0.001_d_5_7_9_13_17_23_29_39_50_both_bases/pymatching_v0.7_vs_pymatching_v2_vs_networkx_timing_p=0.001_per_round_both_bases_decoded.png)


Sparse blossom is conceptually similar to the approach described in [this paper](https://arxiv.org/abs/1307.1740) 
by Austin Fowler, although our approach differs in many of the details (which will be explained in our upcoming paper).
There are even more similarities with the very nice independent work by Yue Wu, who recently released the 
[fusion-blossom](https://pypi.org/project/fusion-blossom/) library.
One of the differences with our approach is that fusion-blossom grows the exploratory regions of alternating trees 
in a similar way to how clusters are grown in Union-Find, whereas our approach instead progresses along a timeline, 
and uses a global priority queue to grow alternating trees.
Yue also has a paper coming soon, so stay tuned for that as well.

## インストール

The latest version of PyMatching can be downloaded and installed from [PyPI](https://pypi.org/project/PyMatching/) 
with the command:

```
pip install pymatching --upgrade
```


## 使い方

PyMatching can load matching graphs from a check matrix, a `stim.DetectorErrorModel`, a `networkx.Graph`, a 
`retworkx.PyGraph` or by adding edges individually with `pymatching.Matching.add_edge` and 
`pymatching.Matching.add_boundary_edge`.

### Stim回路のデコード

PyMatching can be combined with [Stim](https://github.com/quantumlib/Stim). Generally, the easiest and fastest way to 
do this is using [sinter](https://pypi.org/project/stim/) (use v1.10.0 or later), which uses PyMatching and Stim to run 
parallelised monte carlo simulations of quantum error correction circuits.
However, in this section we will use Stim and PyMatching directly, to demonstrate how their Python APIs can be used.
To install stim, run `pip install stim --upgrade`.

First, we generate a stim circuit. Here, we use a surface code circuit included with stim:

```python
import numpy as np
import stim
import pymatching
circuit = stim.Circuit.generated("surface_code:rotated_memory_x", 
                                 distance=5, 
                                 rounds=5, 
                                 after_clifford_depolarization=0.005)
```

Next, we use stim to generate a `stim.DetectorErrorModel` (DEM), which is effectively a 
[Tanner graph](https://en.wikipedia.org/wiki/Tanner_graph) describing the circuit-level noise model.
By setting `decompose_errors=True`, stim decomposes all error mechanisms into _edge-like_ error 
mechanisms (which cause either one or two detection events).
This ensures that our DEM is graphlike, and can be loaded by pymatching:

```python
model = circuit.detector_error_model(decompose_errors=True)
matching = pymatching.Matching.from_detector_error_model(model)
```

Next, we will sample 1000 shots from the circuit. Each shot (a row of `shots`) contains the full syndrome (detector 
measurements), as well as the logical observable measurements, from simulating the noisy circuit:

```python
sampler = circuit.compile_detector_sampler()
syndrome, actual_observables = sampler.sample(shots=1000, separate_observables=True)
```

Now we can decode! We compare PyMatching's predictions of the logical observables with the actual observables sampled 
with stim, in order to count the number of mistakes and estimate the logical error rate:

```python
num_errors = 0
for i in range(syndrome.shape[0]):
    predicted_observables = matching.decode(syndrome[i, :])
    num_errors += not np.array_equal(actual_observables[i, :], predicted_observables)

print(num_errors)  # prints 8
```

### Loading from a parity check matrix

We can also load a `pymatching.Matching` object from a binary
[parity check matrix](https://en.wikipedia.org/wiki/Parity-check_matrix), another representation of a Tanner graph.
Each row in the parity check matrix `H` corresponds to a parity check, and each column corresponds to an 
error mechanism.
The element `H[i,j]` of `H` is 1 if parity check `i` is flipped by error mechanism `j`, and 0 otherwise.
To be used by PyMatching, the error mechanisms in `H` must be _graphlike_.
This means that each column must contain either one or two 1s (if a column has a single 1, it represents a half-edge 
connected to the boundary).

We can give each edge in the graph a weight, by providing PyMatching with a `weights` numpy array.
Element `weights[j]` of the `weights` array sets the edge weight for the edge corresponding to column `j` of `H`.
If the error mechanisms are treated as independent, then we typically want to set the weight of edge `j` to 
the log-likelihood ratio `log((1-p_j)/p_j)`, where `p_j` is the error probability associated with edge `j`.
With this setting, PyMatching will find the most probable set of error mechanisms, given the syndrome.

With PyMatching configured using `H` and `weights`, decoding a binary syndrome vector `syndrome` (a numpy array 
of length `H.shape[0]`) corresponds to finding a set of errors defined in a binary `predictions` vector 
satisfying `H@predictions % 2 == syndrome` while minimising the total solution weight `predictions@weights`.

In quantum error correction, rather than predicting which exact set of error mechanisms occurred, we typically want to 
predict the outcome of _logical observable_ measurements, which are the parities of error mechanisms.
These can be represented by a binary matrix `observables`. Similar to the check matrix, `observables[i,j]` is 1 if 
logical observable `i` is flipped by error mechanism `j`.
For example, suppose our syndrome `syndrome`, was the result of a set of errors `noise` (a binary array of 
length `H.shape[1]`), such that `syndrome = H@noise % 2`.
Our decoding is successful if `observables@noise % 2 == observables@predictions % 2`.

Putting this together, we can decode a distance 5 repetition code as follows:

```python
import numpy as np
from scipy.sparse import csc_matrix
import pymatching
H = csc_matrix([[1, 1, 0, 0, 0],
                 [0, 1, 1, 0, 0],
                 [0, 0, 1, 1, 0],
                 [0, 0, 0, 1, 1]])
weights = np.array([4, 3, 2, 3, 4])   # Set arbitrary weights for illustration
matching = pymatching.Matching(H, weights=weights)
prediction = matching.decode(np.array([0, 1, 0, 1]))
print(prediction)  # prints: [0 0 1 1 0]
# Optionally, we can return the weight as well:
prediction, solution_weight = matching.decode(np.array([0, 1, 0, 1]), return_weight=True)
print(prediction)  # prints: [0 0 1 1 0]
print(solution_weight)  # prints: 5.0
```

And in order to estimate the logical error rate for a physical error rate of 10%, we can sample 
as follows:

```python
import numpy as np
from scipy.sparse import csc_matrix
import pymatching
H = csc_matrix([[1, 1, 0, 0, 0],
                [0, 1, 1, 0, 0],
                [0, 0, 1, 1, 0],
                [0, 0, 0, 1, 1]])
observables = csc_matrix([[1, 0, 0, 0, 0]])
error_probability = 0.1
weights = np.ones(H.shape[1]) * np.log((1-error_probability)/error_probability)
matching = pymatching.Matching.from_check_matrix(H, weights=weights)
num_errors = 0
for i in range(1000):
    noise = (np.random.random(H.shape[1]) < error_probability).astype(np.uint8)
    syndrome = H@noise % 2
    prediction = matching.decode(syndrome)
    predicted_observables = observables@prediction % 2
    actual_observables = observables@noise % 2
    num_errors += not np.array_equal(predicted_observables, actual_observables)
print(num_errors)  # prints 4
```

Note that we can also ask PyMatching to predict the logical observables directly, by supplying them 
to the `faults_matrix` argument when constructing the `pymatching.Matching` object. This allows the decoder to make 
some additional optimisations, that speed up the decoding procedure a bit. The following example uses this approach, 
and is equivalent to the example above:

```python
import numpy as np
from scipy.sparse import csc_matrix
import pymatching

H = csc_matrix([[1, 1, 0, 0, 0],
                [0, 1, 1, 0, 0],
                [0, 0, 1, 1, 0],
                [0, 0, 0, 1, 1]])
observables = csc_matrix([[1, 0, 0, 0, 0]])
error_probability = 0.1
weights = np.ones(H.shape[1]) * np.log((1-error_probability)/error_probability)
matching = pymatching.Matching.from_check_matrix(H, weights=weights, faults_matrix=observables)
num_errors = 0
for i in range(1000):
    noise = (np.random.random(H.shape[1]) < error_probability).astype(np.uint8)
    syndrome = H@noise % 2
    predicted_observables = matching.decode(syndrome)
    actual_observables = observables@noise % 2
    num_errors += not np.array_equal(predicted_observables, actual_observables)

print(num_errors)  # prints 6
```

Instead of using a check matrix, the Matching object can also be constructed using
the [`Matching.add_edge`](https://pymatching.readthedocs.io/en/stable/api.html#pymatching.matching.Matching.add_edge)
and 
[`Matching.add_boundary_edge`](https://pymatching.readthedocs.io/en/stable/api.html#pymatching.matching.Matching.add_boundary_edge) 
methods, or by loading from a NetworkX or retworkx graph. 

For more details on how to use PyMatching,
see [the documentation](https://pymatching.readthedocs.io).

## Attribution

PyMatching version 2で使用されている新しい実装（sparse blossom）に関する論文は近日中に発表される予定です。それまでの間、以下の論文の引用をお願いします。

```
@misc{pymatchingv2,
  author = {Higgott, Oscar and Gidney, Craig},
  title = {PyMatching v2},
  year = {2022},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/oscarhiggott/PyMatching}}
}
```

注釈: 既存のPyMatchingの[論文](https://arxiv.org/abs/2105.13082)では、バージョン0.7以前のPyMatchingの実装について記述されています。

## Acknowledgements

We are grateful to the Google Quantum AI team for supporting the development of PyMatching v2. Earlier versions of 
PyMatching were supported by Unitary Fund and EPSRC.
