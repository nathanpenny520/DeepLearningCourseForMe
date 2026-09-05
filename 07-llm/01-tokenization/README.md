# 01 Tokenization

BPE 分词：从字符到子词，完整实现训练（最高频相邻对合并）与编解码，用 BPE 论文原例语料验证。

**运行**：打开 `tokenization.ipynb`，内核 **Python (llm)**。

**关键结论**：子词是固定词表与短序列的最优折中；BPE 按"最高频对合并"压缩文本；`low`+`est` 等形态复用是核心收益。
