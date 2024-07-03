---
title: "Recall From Memory"
description: "Memory-of-Thought (MoT) is a two-step framework that helps an LLM improve its responses through self-thinking and memory."
---

How do we improve an LLM's reasoning without new data or parameter updates?

Memory-of-Thought (MoT) is a two-step framework that helps an LLM improve its responses through self-thinking and memory, mimicing humans. The steps are:

1. **Before the query**: 
    1. the LLM "pre-thinks" on an unlabeled dataset and
    2. saves the highest confidence thoughts as external memory
2. **During the query**: the LLM recalls relevant memory to help answer the query

MoT has shown to improve ChatGPT's reasoning abilities and improve results from other CoT methods<sup><a href="https://arxiv.org/abs/2305.05181">1</a></sup>.

```python
# Add code here
```

## References

<sup id="ref-1">1</sup>: [MoT: Memory-of-Thought Enables ChatGPT to Self-Improve](https://arxiv.org/abs/2305.05181)

<sup id="ref-asterisk">\*</sup>: [The Prompt Report: A Systematic Survey of Prompting Techniques](https://arxiv.org/abs/2406.06608)