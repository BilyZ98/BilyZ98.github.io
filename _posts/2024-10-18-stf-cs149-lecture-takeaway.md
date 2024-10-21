


## Lecture 5
[Video](https://youtu.be/mmO2Ri_dJkk?si=CCG3Tf9dDYZiExq6)
Deciding granularity is important for dynamic scheduling in 
parallel programming.

Small granularity leads to better workload distribution
but comes with higher synchronization overhead.


## Lecture 6
Performance optimization: locality, communication and contention.

Reduce costs of communication between:
1. processors.
2. between processors and memory.


Shared memory communication.
Numa: non-uniform memory access


Message passing
blocking send and non-blocking send

Reduce communication is important to achieve max 
utilization of cpu. Just to keep cpu busy


Roofline model:
To achieve maximum computation throughput GFLOPS/s
of cpu or gpus one has to have algorithm that has high
operation intensity -> high flops/bytes.

Need to has many computation per byte access unit.


