
## PowerInfer: Fast Large Language Model Serving with a Consumer-grade GPU 

https://arxiv.org/pdf/2312.12456.pdf


## Summary
This paper introduces a compute engine to use cpu and 
gpu memory together to do matrix/neuron computation in LLM inference.
Two key insights:
1. 80% of neurons are cold, 20% of neurons are hot in LLM inference. -> load hot neurons to gpu.
2. cpu direct matrix calculatoin is faster than load and execute on gpu with few computation.

- Is is true that all neurons of neural network follow skewed law? The paper only mentions LLM inference
- Quality loss when skip cold neuron calculation, what's the numerical impact of this one? How does 
PowerInfer solve this one? 


