
[Lecture 7 slides](https://gfxcourses.stanford.edu/cs149/fall23/lecture/gpucuda/)

[Video lecture](https://www.youtube.com/watch?v=qQTDF0CBoxE&list=PLoROMvodv4rMp7MTFr4hQsDEcX7Bx6Odp&index=7&pp=iAQB)


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

![image](https://github.com/user-attachments/assets/851fe0f2-52ec-4b7b-8a23-d3870982c520)


Why this 3 level hierachy adress space ? 
For efficient memory access when threads in thread block are scheduled in
the same core.

## Nvidia gpu (implementation)

A warp in nvidia gpu is a gropu of 32 threads in thread block.
![image](https://github.com/user-attachments/assets/e2f4aa55-103c-404b-828b-28d693b9c72b)


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

Each SM(streaming multi-processor) has 4 sub-core.

V100 has 80 SMs in total.

For V100, each SM(streaming multi-processor) has 4 sub-cores. 
![image](https://github.com/user-attachments/assets/dbca936d-0da6-42fa-82d9-ceaf3d91596d)


Instruction execution.

Since we have more execution context than ALUs, each instructions is finished 
half of the work in one cycle and another half of the work in the next cycle.

![image](https://github.com/user-attachments/assets/0531761c-6aef-437d-814a-095990d67950)


