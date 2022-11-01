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
[sinter](https://pypi.org/project/sinter/) パッケージでは、Stim と 
PyMatching を組み合わせて、量子エラー訂正回路のモンテカルロ・サンプリングを高速・並列に実行します。

PyMatchingのドキュメントは[pymatching.readthedocs.io](https://pymatching.readthedocs.io/en/stable/) をご覧ください。

stim, sinter, pymatchingによって回路レベルのノイズを含む誤り訂正符号の閾値を推定する方法については、[stim getting started notebook](https://github.com/quantumlib/Stim/blob/main/doc/getting_started.ipynb)を試してみてください。

## Version 2の100倍以上高速な新実装について

バージョン2は、私がCraig Gidneyと共同で書いたblossomアルゴリズムの新しい実装を特徴としています。
私たちの新しい実装は、_sparse blossom_ アルゴリズムと呼ばれ、QECに関連する復号問題を扱うためのblossomアルゴリズムの一般化として見ることができます。
blossomアルゴリズムを一般化して、QECに関連する復号化問題を扱えるようにしたものです。
我々は、検出グラフの検出イベント間の最小重みの経路を見つける問題を _直接_ 解決します。
これによって、元のblossomアルゴリズムを使用した際に発生する派生グラフのMWPMを見つけるためのコストのかかる全対全ダイクストラ探索を回避することができます。
新しいバージョンはまたexactです - 以前のバージョンのPyMatchingとは異なり、近似は行われません。

我々の新しい実装は、以前のバージョンのPyMatchingと比較して**100倍以上**高速です。
NetworkXに比べて**100,000倍**の速度です(表面符号回路でベンチマークした場合)。回路ノイズが0.1%の場合、PyMatchingは
距離13までの表面符号回路のXおよびZ基底を、シングルコアでのシンドローム解析1ラウンドあたり1マイクロ秒未満でデコードすることができます（X基底の測定値のみを処理する場合は距離19まで、ただし大規模な状況ではX基底とZ基底の両方を処理する必要があります）。
さらに、実行時間はグラフのノード数にほぼ比例しています。

以下のグラフでは、回路レベルの脱分極ノイズを含む表面符号回路のデコードにおいて、PyMatching v2と旧バージョン(v0.7)、NetworkXを比較したものです。
すべてのデコーダはM1プロセッサのシングルコアで実行され、XとZの両方の基底の測定値を処理しました。
凡例にあるT=N^xの式（および破線でプロット）は、同じデータに対するフィットから得られたものです。
ここで、Nは1ラウンドあたりの検出器（ノード）数、Tは1ラウンドあたりの復号化時間です。
データおよびstim回路、他のベンチマークについてはリポジトリの[benchmarks](https://github.com/oscarhiggott/PyMatching/raw/master/benchmarks)フォルダを参照してください。

![PyMatching new vs old vs NetworkX](https://github.com/oscarhiggott/PyMatching/raw/master/benchmarks/surface_codes/surface_code_rotated_memory_x_p_0.001_d_5_7_9_13_17_23_29_39_50_both_bases/pymatching_v0.7_vs_pymatching_v2_vs_networkx_timing_p=0.001_per_round_both_bases_decoded.png)

Sparse blossomはAustin Fowlerの[論文](https://arxiv.org/abs/1307.1740)に記載されているものと概念的に似たアプローチとなっていますが、我々のアプローチは細部が色々と異なっています(これについては我々の次の論文で説明されます)。
また、最近[fusion-blossom](https://pypi.org/project/fusion-blossom/)ライブラリーをリリースしたYue Wuによる非常に素晴らしい独立した研究にも類似点があります。
我々のアプローチとの違いの1つは、fusion-blossomは交互木の探索領域を、Union-Findでクラスタが成長するのと同じような方法で成長させるのに対し、我々のアプローチは時系列に沿って進行することです。
そして、グローバルな優先順位キューを使用して、交互木を成長させます。
Yueは近々論文も発表する予定ですので、そちらもご期待ください。

## インストール

PyMatchingの最新版は[PyPI](https://pypi.org/project/PyMatching/)からダウンロードしてインストールすることができます。
以下のコマンドでインストールできます。

```
pip install pymatching --upgrade
```

## 使い方

PyMatching はチェックマトリックス、 `stim.DetectorErrorModel` 、 `networkx.Graph` 、 `retworkx.PyGraph` からマッチンググラフを読み込むことができます。
あるいは、`pymatching.Matching.add_edge` と `pymatching.Matching.add_boundary_edge` を使って個別にエッジを追加することができます。

### Stim回路のデコード

PyMatching can be combined with [Stim](https://github.com/quantumlib/Stim). Generally, the easiest and fastest way to 
do this is using [sinter](https://pypi.org/project/stim/) (use v1.10.0 or later), which uses PyMatching and Stim to run 
parallelised monte carlo simulations of quantum error correction circuits.
However, in this section we will use Stim and PyMatching directly, to demonstrate how their Python APIs can be used.
To install stim, run `pip install stim --upgrade`.

PyMatchingは[Stim](https://github.com/quantumlib/Stim)と組み合わせることができます。
[sinter](https://pypi.org/project/stim/) (v1.10.0、あるいはより最新のバージョンを使用してください)は、PyMatchingとStimを使って、量子エラー訂正回路の並列モンテカルロシミュレーションを実行するもので、一般に最も簡単で速い方法です。
しかし、このセクションでは、StimとPyMatchingを直接使用し、Python APIの使用方法について説明します。
Stimをインストールするには、`pip install stim --upgrade` を実行してください。

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
