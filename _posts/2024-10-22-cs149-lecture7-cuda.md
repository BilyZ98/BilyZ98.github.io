

## CUDA programming model ( abstraction )

Three execution unit and memory address
1.  thread
2.  thread block
3.  cuda kernel


A thread block contains bunch of threads.

A cuda kernal contains all the thread blocks.

Memory address space
1. Each thread has its own memory address space
2. Each thread block has its own shared memory address space for all 
threads in the thread block
3. All threads across all thread blocks share a process memory address space


Why this 3 level hierachy adress space ? 
For efficient memory access when threads in thread block are scheduled in
the same core.

## Nvidia gpu (implementation)

A warp in nvidia gpu is a gropu of 32 threads in thread block.
