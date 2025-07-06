## 1. 置信区间的基本计算公式

\*\*最常见场景：\*\*估计总体均值（比如“某地居民的平均身高”），你只拿到样本。

### 基本公式

对于**正态分布**（样本量大时近似成立）：

$$
\text{置信区间} = \bar{x} \pm z_{\alpha/2} \times \frac{s}{\sqrt{n}}
$$

* $\bar{x}$：样本均值
* $s$：样本标准差
* $n$：样本容量
* $z_{\alpha/2}$：置信系数，比如95%置信度时，查标准正态分布表是**1.96**
* $\frac{s}{\sqrt{n}}$：标准误差

**解读：** 用样本均值，两边各加上“z倍的标准误差”，得到置信区间的上下界。

---

## 2. 举例

假如：

* 你抽样100人，身高均值170cm，样本标准差8cm
* 95%置信区间是多少？

计算：

* 标准误差：$8 / \sqrt{100} = 0.8$
* 区间：$170 \pm 1.96 \times 0.8 = 170 \pm 1.568$
* 答案：**\[168.432, 171.568]**

---

## 3. 对于比例（如“通过率”、“准确率”）

假设n次实验，p是样本比例，常用**二项分布**置信区间，比如Clopper-Pearson方法（用在投票占比时）。

* 公式（正态近似）：

  $$
  \text{置信区间} = p \pm z_{\alpha/2} \sqrt{\frac{p(1-p)}{n}}
  $$

  例如：

  * 100人中有90人通过，p=0.9
  * 置信区间：$0.9 \pm 1.96 \times \sqrt{0.9 \times 0.1 / 100} \approx 0.9 \pm 0.0589$
  * 答案：\[0.841, 0.959]

  更严谨时用Clopper-Pearson，见下方代码。

---

## 4. Python 代码实现

用scipy计算比例的置信区间（Clopper-Pearson）：

```python
from scipy.stats import beta

def clopper_pearson_ci(k, n, alpha=0.05):
    lower = beta.ppf(alpha/2, k, n-k+1)
    upper = beta.ppf(1 - alpha/2, k+1, n-k)
    return lower, upper

# 90/100例子
print(clopper_pearson_ci(90, 100, alpha=0.05))  # 95%置信区间
```

---

## 5. 其他场景

* **方差未知、样本小**：用t分布替换z分布，查t表即可。
* **复杂模型下（如AI认证投票）**：仍然是统计比例的置信区间计算，常用Clopper-Pearson等方法。

---

## 小结

* **核心思路**：用样本数据估计总体，估算误差范围，乘以对应置信系数，得到区间上下界。
* **工具**：z表、t表、beta函数、scipy/statsmodels等库。

---
