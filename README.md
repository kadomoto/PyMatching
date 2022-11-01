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
また、最近[fusion-blossom](https://pypi.org/project/fusion-blossom/)ライブラリーをリリースしたYue Wuによる素晴らしい独自研究とも類似点があります。
違いの1つは、fusion-blossomは交互木の探索領域を、Union-Findでクラスタが成長するのと同じような方法で成長させるのに対し、我々のアプローチは時系列に沿って進行することです。
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

PyMatchingは[Stim](https://github.com/quantumlib/Stim)と組み合わせることができます。
[sinter](https://pypi.org/project/stim/) (v1.10.0、あるいはより最新のバージョンを使用してください)は、PyMatchingとStimを使って、量子エラー訂正回路の並列モンテカルロシミュレーションを実行するもので、一般に最も簡単で速い方法です。
しかし、このセクションでは、StimとPyMatchingを直接使用し、Python APIの使用方法について説明します。
Stimをインストールするには、`pip install stim --upgrade` を実行してください。

まず、stimの回路を生成します。ここでは、stimに付属するサーフェスコード回路を使用します。

```python
import numpy as np
import stim
import pymatching
circuit = stim.Circuit.generated("surface_code:rotated_memory_x", 
                                 distance=5, 
                                 rounds=5, 
                                 after_clifford_depolarization=0.005)
```

次に、stim を用いて `stim.DetectorErrorModel` (DEM) を生成します。これは、実質的には 
[Tannerグラフ](https://en.wikipedia.org/wiki/Tanner_graph)であり、回路レベルのノイズモデルを記述します。
`decompose_errors=True` を設定することにより、stim はすべてのエラー機構を _edge-like_ エラー機構（1つまたは2つの検出イベントを引き起こす）に分解します。
これにより、DEMはグラフ的になり、pymatchingで読み込むことができるようになります。

```python
model = circuit.detector_error_model(decompose_errors=True)
matching = pymatching.Matching.from_detector_error_model(model)
```

次に、回路から1000ショットをサンプリングします。各ショット（`shots`の列）は、完全なシンドローム（検出器 
の測定値）と、ノイズの多い回路をシミュレートして得られた論理的な観測値からなります。

```python
sampler = circuit.compile_detector_sampler()
syndrome, actual_observables = sampler.sample(shots=1000, separate_observables=True)
```

これでデコードができるようになりました！PyMatchingの論理的観測値の予測と、stimでサンプリングした実際の観測値を比較し、誤りの数を数え、論理エラー率を推定します。

```python
num_errors = 0
for i in range(syndrome.shape[0]):
    predicted_observables = matching.decode(syndrome[i, :])
    num_errors += not np.array_equal(actual_observables[i, :], predicted_observables)

print(num_errors)  # prints 8
```

### パリティチェック行列からの読み込み

また、バイナリの[パリティチェック行列](https://en.wikipedia.org/wiki/Parity-check_matrix)、Tannerグラフの別の表現、から `pymatching.Matching` オブジェクトをロードすることもできます。
パリティチェック行列 `H` の各行はパリティチェックに対応し、各列はエラーメカニズムに対応します。
H` の要素 `H[i,j]` は、パリティチェック `i` がエラーメカニズム `j` によって反転された場合に 1 となり、そうでない場合に 0 となります。
PyMatching で使用するためには、`H` のエラーメカニズムは _graphlike_ （グラフ的）である必要があります。
これは、各列には1つまたは2つの1が含まれていなければならないことを意味します（列が1つの1を持つ場合、それは境界に接続されたハーフエッジを表します）。

PyMatchingに `weights` というnumpy配列を与えることで、グラフ内の各エッジに重みを与えることができます。
`weights` 配列の要素 `weights[j]` は、 `H` の列 `j` に対応するエッジの重みを設定します。
エラー要因が独立したものとして扱われる場合、通常、エッジ `j` の重みは `log((1-p_j)/p_j)` として設定されます。
ここで、 `p_j` はエッジ `j` に関連する誤り確率です。
この設定により、PyMatchingはシンドロームが与えられたときに、最も確率の高いエラーメカニズムの集合を見つけます。

With PyMatching configured using `H` and `weights`, decoding a binary syndrome vector `syndrome` (a numpy array 
of length `H.shape[0]`) corresponds to finding a set of errors defined in a binary `predictions` vector 
satisfying `H@predictions % 2 == syndrome` while minimising the total solution weight `predictions@weights`.

PyMatchingが `H` と `weights` を用いて設定されている場合、バイナリシンドロームベクトル `syndrome` (長さ `H.shape[0]` のnumpy配列) のデコードは、 `H@predictions % 2 == syndrome` を満たし、かつ解の総重み `predictions@weights` を最小にするような、バイナリ `predictions` ベクトルで定義されるエラーのセットを見つけることに対応します。

In quantum error correction, rather than predicting which exact set of error mechanisms occurred, we typically want to 
predict the outcome of _logical observable_ measurements, which are the parities of error mechanisms.
These can be represented by a binary matrix `observables`. Similar to the check matrix, `observables[i,j]` is 1 if 
logical observable `i` is flipped by error mechanism `j`.
For example, suppose our syndrome `syndrome`, was the result of a set of errors `noise` (a binary array of 
length `H.shape[1]`), such that `syndrome = H@noise % 2`.
Our decoding is successful if `observables@noise % 2 == observables@predictions % 2`.

量子誤り訂正では、どのエラーメカニズムが発生したかを正確に予測するのではなく、通常、「論理的に観測可能な」測定結果、すなわちエラーメカニズムのパリティ、を予測したいのです。
これらはバイナリ行列の `observables` で表現されます。チェック行列と同様に、論理観測値 `i` がエラーメカニズム `j` によって反転された場合、`observables[i,j]` は 1 になります。
例えば、シンドローム `syndrome` が、一連のエラー `noise` （長さ `H.shape[1]` のバイナリ配列）の結果、たとえば、`syndrome = H@noise % 2` であったとします。
もし、 `observables@noise % 2 == observables@predictions % 2` ならば、我々のデコードは成功です。

これをまとめると、距離5の繰り返し符号を次のようにデコードすることができます。

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

また、物理エラー率10%に対する論理エラー率を推定するために、次のようにサンプルします。

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

`pymatching.Matching` オブジェクト構築の際に `faults_matrix` 引数を与えることで、PyMatching に直接論理的な観測値を予測させることもできます。これによってデコーダは 
いくつか追加の最適化を行い、デコード処理を少し速くすることができます。次の例では、この方法を使用しており、上で示した例と等価な操作になっています。

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

チェック行列を使う代わりに、Matching オブジェクトを[`Matching.add_edge`](https://pymatching.readthedocs.io/en/stable/api.html#pymatching.matching.Matching.add_edge)
メソッドと
[`Matching.add_boundary_edge`](https://pymatching.readthedocs.io/en/stable/api.html#pymatching.matching.Matching.add_boundary_edge) 
メソッドを用いて構築することもできます。または NetworkX や retworkx のグラフからロードすることによっても構築できます。

PyMatching の使い方の詳細については
[ドキュメント](https://pymatching.readthedocs.io)を参照してください。

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
