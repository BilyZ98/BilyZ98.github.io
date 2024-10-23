

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

Different CUDA thread has it own PC(Program counter)
even though they are in the same warp.

However, since all threads in the same warp is likely to execute the 
same code and same instructions it effectively looks like that there are only 
4 unique PCs even though in reality there are 4 * 32 = 128 PCs.

Difference between warp and thread block.

A thread block is an programming model abstraction.

A warp in hardware implementation.

Both represent the concept of group of threads . 


sub-core has 4 warp in the diagram below. 

Each SM(streaming multi-processor)


Instruction execution.

Since we have more execution context than ALUs, each instructions is finished 
half of the work in one cycle and another half of the work in the next cycle.

