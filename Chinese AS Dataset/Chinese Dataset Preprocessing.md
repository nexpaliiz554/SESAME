## Chinese Dataset Preprocessing

The Chinese datasets are based on the work of [[Peng et al. @KBS2018]](http://sentic.net/chinese-review-datasets.zip). Each sentence is annotated with only one aspect-sentiment (a, s) pair.

To support learning-based models that require aspect index ranges, we convert the original text-annotated aspects into character span indices.
 For example:

- **Text**: 外观亮丽 (The appearance is good)
- **Aspect**: 外观 (appearance)
- **Converted Index Range**: `[0, 2]`

During this process, we handle the following special cases:

- **Aspect not a substring**:
   If the annotated aspect is not a substring of the sentence (e.g.,
  - **Text**: 小米的是让我感觉最舒心的 (Xiaomi's makes me feel most comfortable)
  - **Aspect**: UI
     ), we exclude the instance. This accounts for approximately **2%** of the data.
- **Multiple occurrences of aspect**:
   If the aspect appears more than once (e.g.,
  - **Text**: 外观外观还行 (appearance appearance is acceptable)
     ), we manually define the span. In most cases, we select the **first occurrence** as the final aspect span.
  - **Selected Index Range**: `[0, 2]`
     Such cases account for **less than 1%** of the data.