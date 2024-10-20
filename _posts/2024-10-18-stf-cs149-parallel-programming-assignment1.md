---
layout: post
title:  Parallel programming - round robin assignment
date: 2024-10-17 07:59:00-0400
description:  
tags:  parallel programming  
categories: parallel programming
featured: false
---



## Program1: generate image with multiple threads.

Hardware:
hyperthreads? 


Code: 
partition the image generation task.


Plot speedup line with number of threads as x axis.


```
(base) ➜  prog1_mandelbrot_threads git:(master) ✗ ./mandelbrot --threads 2
[mandelbrot serial]:            [525.650] ms
Wrote image file mandelbrot-serial.ppm
[mandelbrot thread]:            [280.569] ms
Wrote image file mandelbrot-thread.ppm
                                (1.87x speedup from 2 threads)
(base) ➜  prog1_mandelbrot_threads git:(master) ✗ ./mandelbrot --threads 3
Wrote image file mandelbrot-serial.ppm
[mandelbrot thread]:            [341.063] ms
Wrote image file mandelbrot-thread.ppm                                                                                                                                                       (1.56x speedup from 3 threads)
(base) ➜  prog1_mandelbrot_threads git:(master) ✗ ./mandelbrot --threads 4
[mandelbrot serial]:            [531.202] ms
Wrote image file mandelbrot-serial.ppm
[mandelbrot thread]:            [237.027] ms
Wrote image file mandelbrot-thread.ppm                                                                                                                                                       (2.24x speedup from 4 threads)
(base) ➜  prog1_mandelbrot_threads git:(master) ✗ ./mandelbrot --threads 5
[mandelbrot serial]:            [532.980] ms
Wrote image file mandelbrot-serial.ppm
[mandelbrot thread]:            [213.517] ms
Wrote image file mandelbrot-thread.ppm                                                                                                                                                       (2.50x speedup from 5 threads)
(base) ➜  prog1_mandelbrot_threads git:(master) ✗ ./mandelbrot --threads 6
[mandelbrot serial]:            [531.480] ms
Wrote image file mandelbrot-serial.ppm
[mandelbrot thread]:            [182.457] ms
Wrote image file mandelbrot-thread.ppm                                                                                                                                                       (2.91x speedup from 6 threads)
(base) ➜  prog1_mandelbrot_threads git:(master) ✗ ./mandelbrot --threads 7
[mandelbrot serial]:            [530.595] ms
Wrote image file mandelbrot-serial.ppm
[mandelbrot thread]:            [173.007] ms
Wrote image file mandelbrot-thread.ppm
Mismatch : [1197][142], Expected : 1, Actual : 0
Error : Output from threads does not match serial output
(base) ➜  prog1_mandelbrot_threads git:(master) ✗ ./mandelbrot --threads 8
[mandelbrot serial]:            [534.651] ms
Wrote image file mandelbrot-serial.ppm
[mandelbrot thread]:            [150.080] ms
Wrote image file mandelbrot-thread.ppm                                                                                                                                                       (3.56x speedup from 8 threads)
```


Time for each thread

Not all threads run with same finish time.


Why is that?
```
(base) ➜  prog1_mandelbrot_threads git:(master) ✗ ./mandelbrot --threads 2
[mandelbrot serial]:            [529.644] ms
Wrote image file mandelbrot-serial.ppm
exe time: 275.106996 ms
exe time: 294.014979 ms
exe time: 278.083589 ms
exe time: 280.188169 ms
exe time: 268.240355 ms
exe time: 288.978558 ms
exe time: 274.672702 ms
exe time: 285.212621 ms
exe time: 275.313959 ms
exe time: 291.359153 ms
[mandelbrot thread]:            [280.327] ms
Wrote image file mandelbrot-thread.ppm                                                                                                                                                       
(1.89x speedup from 2 threads)


(base) ➜  prog1_mandelbrot_threads git:(master) ✗ ./mandelbrot --threads 3
[mandelbrot serial]:            [532.551] ms
Wrote image file mandelbrot-serial.ppm
exe time: 120.048959 ms
exe time: 127.574176 ms
exe time: 346.292444 ms
exe time: 122.336693 ms
exe time: 122.885458 ms
exe time: 342.198521 ms
exe time: 123.949669 ms
exe time: 123.917334 ms
exe time: 343.334582 ms
exe time: 121.276554 ms
exe time: 121.796411 ms
exe time: 339.319866 ms
exe time: 122.690346 ms
exe time: 123.405423 ms
exe time: 341.921013 ms
[mandelbrot thread]:            [339.491] ms
Wrote image file mandelbrot-thread.ppm
                                (1.57x speedup from 3 threads)


(base) ➜  prog1_mandelbrot_threads git:(master) ✗ ./mandelbrot --threads 4
[mandelbrot serial]:            [532.573] ms
Wrote image file mandelbrot-serial.ppm
exe time: 66.314548 ms
exe time: 69.146506 ms
exe time: 236.007646 ms
exe time: 236.860119 ms
exe time: 67.293212 ms
exe time: 68.643764 ms
exe time: 235.531762 ms
exe time: 235.957604 ms
exe time: 67.872606 ms
exe time: 68.252590 ms
exe time: 231.048137 ms
exe time: 236.915503 ms
exe time: 68.757534 ms
exe time: 70.160590 ms
exe time: 219.524853 ms
exe time: 238.315675 ms
exe time: 66.293016 ms
exe time: 66.733379 ms
exe time: 233.316239 ms
exe time: 234.051295 ms
[mandelbrot thread]:            [234.236] ms
Wrote image file mandelbrot-thread.ppm
                                (2.27x speedup from 4 threads)


(base) ➜  prog1_mandelbrot_threads git:(master) ✗ ./mandelbrot --threads 8
[mandelbrot serial]:            [533.956] ms
Wrote image file mandelbrot-serial.ppm
exe time: 21.111727 ms
exe time: 21.316495 ms
exe time: 59.669584 ms
exe time: 59.972882 ms
exe time: 101.592321 ms
exe time: 104.972839 ms
exe time: 144.783191 ms
exe time: 145.647489 ms
[mandelbrot thread]:            [133.506] ms
Wrote image file mandelbrot-thread.ppm
                                (4.00x speedup from 8 threads)
```


## More efficient implementation

cacheline aware ? 

In this code image is divided by `numThreads` blocks and 
each thread with `threadId`   accesses idx:threadId of each block.

This means that it's highly likely that at the same moment each
thread access the same memory block that is in cache.

This is just my understanding.


The reason for some threads taking much longer time to finish the job
is that some adjacent rows need much more time to compute.
If we use round-robin assignment strategy then we can distribute the 
computation job evenly and each thread can get equal amount of computation job.

GPT give me following answer when I asked it why to use round-robin assignment:

Thread 2 has one more row to process than the other threads, which can lead to a slight imbalance. However, the imbalance becomes more pronounced if the work done per row is not uniform. For example, if the computation for the Mandelbrot set is more complex for certain rows, the threads processing those rows will take longer to complete, leading to idle time for other threads.

This imbalance can be avoided by using a better load balancing strategy, such as the round-robin assignment used in the original code, which distributes the rows more evenly across the threads. 😊


```cpp
void workerThreadStart(WorkerArgs * const args) {

  // TODO FOR CS149 STUDENTS: Implement the body of the worker
  // thread here. Each thread should make a call to mandelbrotSerial()
  // to compute a part of the output image.  For example, in a
  // program that uses two threads, thread 0 could compute the top
  // half of the image and thread 1 could compute the bottom half.
  //
  double startTime = CycleTimer::currentSeconds();
  float x0 = args->x0;
  float y0 = args->y0;
  float x1 = args->x1;
  float y1 = args->y1;
  int width = args->width;
  int height = args->height;

  // printf("cur start row: %d, cur total row:%d\n", cur_start_row, cur_total_rows);
  for(int i=0; i < height/args->numThreads; i++) {
    int start_row = args->threadId + i * args->numThreads;
    int num_rows = 1;
    mandelbrotSerial(x0, y0, x1, y1,
                     width, height,
                     start_row, num_rows,
                     args->maxIterations, args->output);


  }
  double endTime = CycleTimer::currentSeconds();
  double exe_time = endTime - startTime;
  printf("exe time: %f ms\n", exe_time*1000);


  // printf("Hello world from thread %d\n", args->threadId);
}
```

```
(base) ➜  prog1_mandelbrot_threads git:(master) ✗ ./mandelbrot --threads 4
[mandelbrot serial]:            [533.651] ms
Wrote image file mandelbrot-serial.ppm
exe time: 142.692391 ms
exe time: 157.330707 ms
exe time: 157.454997 ms
exe time: 157.499295 ms
exe time: 141.290620 ms
exe time: 150.249667 ms
exe time: 150.373559 ms
exe time: 150.334526 ms
exe time: 138.450164 ms
exe time: 151.096858 ms
exe time: 151.156478 ms
exe time: 151.210513 ms
exe time: 138.270788 ms
exe time: 150.668491 ms
exe time: 150.774766 ms
exe time: 150.800020 ms
exe time: 138.079636 ms
exe time: 150.737014 ms
exe time: 150.741972 ms
exe time: 150.848471 ms
[mandelbrot thread]:            [150.471] ms
Wrote image file mandelbrot-thread.ppm
                                (3.55x speedup from 4 threads)
```


### Comparison between inefficient assignment and round-robin assignment
Naive sequential assignment:

Thread running time for another image genration 
```
(base) ➜  prog1_mandelbrot_threads git:(master) ✗ ./mandelbrot --view 2 --threads 4
[mandelbrot serial]:            [311.051] ms
Wrote image file mandelbrot-serial.ppm
exe time: 84.468454 ms
exe time: 86.912777 ms
exe time: 87.307919 ms
exe time: 133.808278 ms
[mandelbrot thread]:            [119.725] ms
Wrote image file mandelbrot-thread.ppm                                                                                                                                                       
(2.60x speedup from 4 threads)
```

Round-robin assignment:
```
(base) ➜  prog1_mandelbrot_threads git:(master) ✗ ./mandelbrot --view 2 --threads 4
[mandelbrot serial]:            [310.842] ms
Wrote image file mandelbrot-serial.ppm
exe time: 83.830711 ms
exe time: 93.051653 ms
exe time: 93.096461 ms
exe time: 93.373701 ms
[mandelbrot thread]:            [93.562] ms
Wrote image file mandelbrot-thread.ppm                                                                                                                                                       (3.32x speedup from 4 threads)
```
