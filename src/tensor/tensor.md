# Hiyoko Tensor operation rule




```
tensor axis at Tensor2d
->axis1
[[,,,,],  ↓axis0
 [,,,,],]
because tensor[axis0][axis1] is natural 

so
let array = [[T; axis1]; axis0]
reverse styele

```
```
両方Vec<row>だとtensor(a, N), tensor(N, b)のdotはrowの加算を毎回するか、
a * N * b回のメモリアクセスになる。
が、左をVec<col>にするとtensor(N, a), tensor(N, b)のdotは結果を書き込む時の
a * b 回のメモリアクセスで済む。
ただし、結果を加算した方が計算量は増えるが
→あんま違い無いっぽいので通常スタイルでいく。

```


