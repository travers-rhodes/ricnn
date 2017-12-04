This model was trained very late monday night, early Tuesday morning, running code found at 
commit d68beec344b6fd17e0d05b161948f064ea416bd7
Author: Travers Rhodes <traversr@andrew.cmu.edu>
Date:   Mon Dec 4 12:09:49 2017 -0500

    10,100,5, trained overnight

Its output is shown below

(tensorflow) coral@peppercmu:~/travers/ricnn/code$ python main.py 
Extracting MNIST_data/train-images-idx3-ubyte.gz
Extracting MNIST_data/train-labels-idx1-ubyte.gz
Extracting MNIST_data/t10k-images-idx3-ubyte.gz
Extracting MNIST_data/t10k-labels-idx1-ubyte.gz
the size of the last conv layer is (?, 46, 46, 100)
the shape of our layer after maxpool (?, 1, 1, 100)
yconv has shape (?, 10)
2017-12-04 00:16:16.452808: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX AVX2 FMA
2017-12-04 00:16:16.536341: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:892] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2017-12-04 00:16:16.536702: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1030] Found device 0 with properties: 
name: GeForce GTX 1080 Ti major: 6 minor: 1 memoryClockRate(GHz): 1.683
pciBusID: 0000:01:00.0
totalMemory: 10.91GiB freeMemory: 10.04GiB
2017-12-04 00:16:16.624406: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:892] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2017-12-04 00:16:16.624721: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1030] Found device 1 with properties: 
name: GeForce GTX 1080 Ti major: 6 minor: 1 memoryClockRate(GHz): 1.683
pciBusID: 0000:02:00.0
totalMemory: 10.91GiB freeMemory: 10.75GiB
2017-12-04 00:16:16.625229: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1045] Device peer to peer matrix
2017-12-04 00:16:16.625280: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1051] DMA: 0 1 
2017-12-04 00:16:16.625285: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1061] 0:   Y Y 
2017-12-04 00:16:16.625301: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1061] 1:   Y Y 
2017-12-04 00:16:16.625306: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1120] Creating TensorFlow device (/device:GPU:0) -> (device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:01:00.0, compute capability: 6.1)
2017-12-04 00:16:16.625325: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1120] Creating TensorFlow device (/device:GPU:1) -> (device: 1, name: GeForce GTX 1080 Ti, pci bus id: 0000:02:00.0, compute capability: 6.1)
checking confusion
[[  0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0]
 [479 563 488 493 535 434 501 550 462 495]
 [  0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0]
 [  0   0   0   0   0   0   0   0   0   0]]
checking accuracy...
(0, 0.1002, 0.1002)
Model saved in file: /tmp/model.ckpt
checking accuracy...
(10000, 0.85460000000000003, 0.85460000000000003)
Model saved in file: /tmp/model.ckpt
checking confusion
[[477   0   0   2   0   0   1   0   2   8]
 [  0 557   3   2   2   0   0   5   0   1]
 [  1   2 374  11  14  26  10  14   0   4]
 [  0   0  10 465   0  17   1   1   7   2]
 [  0   4  32   2 485   1   9  44   3  31]
 [  0   0  26   2   4 363   2   0   1  11]
 [  0   0  10   1   4   8 440   8   1 126]
 [  1   0  16   3  20   2  12 476   0   1]
 [  0   0  16   4   4   2   2   0 447   4]
 [  0   0   1   1   2  15  24   2   1 307]]
checking accuracy...
(20000, 0.91059999999999997, 0.87819999999999998)
Model saved in file: /tmp/model.ckpt
checking accuracy...
(30000, 0.92179999999999995, 0.87619999999999998)
Model saved in file: /tmp/model.ckpt
checking confusion
[[471   0   0   0   0   0   1   0   1   6]
 [  0 550   1   0   1   0   0   1   1   0]
 [  0   3 356   5  15   9   7   9   2   1]
 [  1   0  11 472   3  10   2   1   7   0]
 [  0   1  15   3 453   1   3  18   4  12]
 [  0   0  42   3   4 346   2   2   1   4]
 [  1   3  13   2   3   7 431  11   3 155]
 [  4   6  35   4  43   6  12 504   1   0]
 [  0   0   8   1   1   2   1   0 434   1]
 [  2   0   7   3  12  53  42   4   8 316]]
checking accuracy...
(40000, 0.9224, 0.86660000000000004)
Model saved in file: /tmp/model.ckpt
checking accuracy...
(50000, 0.93440000000000001, 0.88419999999999999)
Model saved in file: /tmp/model.ckpt
checking confusion
[[464   0   0   0   0   0   1   0   0   2]
 [  0 551   1   0   0   0   0   1   0   0]
 [  2   3 390   6  18  14   3   8   3   3]
 [  0   0   8 463   1   4   2   3   8   1]
 [  0   3  16   0 486   1   3  37   2  10]
 [  1   1  32  12   5 389   5   5   5  10]
 [  3   0  17   2   3   0 432  20   1 124]
 [  5   5   7   4   9   1   6 475   0   0]
 [  1   0   5   2   0   0   1   0 435   2]
 [  3   0  12   4  13  25  48   1   8 343]]
checking accuracy...
(60000, 0.92459999999999998, 0.88560000000000005)
Model saved in file: /tmp/model.ckpt
checking accuracy...
(70000, 0.92800000000000005, 0.89459999999999995)
Model saved in file: /tmp/model.ckpt
checking confusion
[[467   0   0   0   0   0   0   0   0   3]
 [  0 554   2   0   0   0   0   2   0   0]
 [  1   1 368   8   8   9   3   6   0   1]
 [  0   0   6 473   0   4   2   2   2   0]
 [  0   5  24   0 503   1   5  27   1   3]
 [  0   0  47   5   6 394   4   5   2   8]
 [  0   0   8   0   0   0 392  16   1  85]
 [  4   3  12   2   8   1   7 489   0   0]
 [  1   0  10   3   0   1   3   0 451   3]
 [  6   0  11   2  10  24  85   3   5 392]]
checking accuracy...
(80000, 0.93540000000000001, 0.89659999999999995)
Model saved in file: /tmp/model.ckpt
checking accuracy...
(90000, 0.93540000000000001, 0.88319999999999999)
Model saved in file: /tmp/model.ckpt
checking confusion
[[473   0   1   0   0   0   1   0   1   6]
 [  0 556   4   1   0   1   0   6   1   0]
 [  1   1 411   5  17  18   5   4   3   3]
 [  0   0   7 485   7  16   2   4   7   2]
 [  0   3   8   0 475   0   3  12   0   6]
 [  0   0  27   0   2 378   3   2   2   4]
 [  0   0  10   0   2   0 434  10   1 156]
 [  4   3  15   1  25   2   9 509   0   3]
 [  0   0   4   0   2   2   2   1 444   6]
 [  1   0   1   1   5  17  42   2   3 309]]
checking accuracy...
(100000, 0.93979999999999997, 0.89480000000000004)
Model saved in file: /tmp/model.ckpt
checking accuracy...
(110000, 0.93879999999999997, 0.8972)
Model saved in file: /tmp/model.ckpt
checking confusion
[[471   0   0   0   0   2   2   0   0   7]
 [  1 560   5   2   0   2   2  14   1   1]
 [  1   0 374   6   2  15   7   4   1   1]
 [  0   0   7 473   0  25   2   3   1   0]
 [  0   2  37   3 511   9   7  24   2  24]
 [  0   0  20   0   1 309   1   4   0   1]
 [  0   0   7   0   0   2 378   2   1  73]
 [  3   1  21   1  17   9  19 498   0   3]
 [  0   0  14   8   1   6   5   0 453   7]
 [  3   0   3   0   3  55  78   1   3 378]]
checking accuracy...
(120000, 0.92920000000000003, 0.88100000000000001)
Model saved in file: /tmp/model.ckpt
checking accuracy...
(130000, 0.94699999999999995, 0.8992)
Model saved in file: /tmp/model.ckpt
checking confusion
[[460   1   0   0   0   0   0   0   0   2]
 [  0 552   2   0   1   0   0   5   0   0]
 [  1   1 386   4  10   7   4   9   2   4]
 [  0   0  13 482   2  19   5   1   4   3]
 [  0   4  17   0 472   1   5  10   0   2]
 [  1   0  30   2   8 381   3   3   2  18]
 [  5   0  14   0   1   5 427  22   2 125]
 [  4   4  17   1  32   2   6 498   0   2]
 [  0   1   7   4   4   2   2   0 449   6]
 [  8   0   2   0   5  17  49   2   3 333]]
checking accuracy...
(140000, 0.93679999999999997, 0.88800000000000001)
Model saved in file: /tmp/model.ckpt
checking accuracy...
(150000, 0.93899999999999995, 0.88880000000000003)
Model saved in file: /tmp/model.ckpt
checking confusion
[[471   0   0   0   0   0   1   0   0   4]
 [  0 553   3   0   0   1   1   2   0   0]
 [  1   2 394   5  17  24   9  15   1  11]
 [  0   0   8 478   1   8   0   0   8   0]
 [  0   5  14   0 492   1   4  20   2  11]
 [  1   0  29   3   3 372   3   8   0   8]
 [  0   0  19   1   2   4 434  21   2 132]
 [  2   3  13   3  14   1   4 482   0   2]
 [  0   0   3   3   0   2   2   0 445   0]
 [  4   0   5   0   6  21  43   2   4 327]]
checking accuracy...
(160000, 0.93979999999999997, 0.88959999999999995)
Model saved in file: /tmp/model.ckpt
checking accuracy...
(170000, 0.93959999999999999, 0.89059999999999995)
Model saved in file: /tmp/model.ckpt
checking confusion
[[470   1   0   1   0   0   1   0   1   3]
 [  0 546   2   0   0   0   0   2   0   0]
 [  1   1 375   6  10   8   4  10   1   5]
 [  0   0   6 476   1  10   4   0   5   1]
 [  0   3  16   1 483   1   2   6   0   4]
 [  0   0  35   2   4 377   4   6   2   5]
 [  0   0  17   0   1   3 419   7   1 103]
 [  3  12  19   3  21   4   9 517   0   3]
 [  0   0  13   4   5   2   2   0 448   5]
 [  5   0   5   0  10  29  56   2   4 366]]
checking accuracy...
(180000, 0.94340000000000002, 0.89539999999999997)
Model saved in file: /tmp/model.ckpt
checking accuracy...
(190000, 0.94279999999999997, 0.90080000000000005)
Model saved in file: /tmp/model.ckpt
checking confusion
[[470   0   0   0   0   1   1   0   0   7]
 [  0 553   3   0   0   0   1   1   0   0]
 [  1   0 390   6   9  14   4  11   1   7]
 [  0   0   3 477   1   6   1   1   4   1]
 [  0   4  15   1 491   1   5  20   4  12]
 [  0   0  31   3   7 399   2   5   0   8]
 [  0   0  19   0   3   0 432  15   3 143]
 [  4   6  18   2  18   1  11 496   0   2]
 [  0   0   7   4   2   1   1   0 446   5]
 [  4   0   2   0   4  11  43   1   4 310]]
checking accuracy...
(200000, 0.94120000000000004, 0.89280000000000004)
Model saved in file: /tmp/model.ckpt
checking accuracy...
(210000, 0.93999999999999995, 0.88119999999999998)
Model saved in file: /tmp/model.ckpt
checking confusion
[[469   0   0   0   0   0   2   0   0   2]
 [  0 554   3   0   1   1   0   5   0   0]
 [  1   0 365   3  10  20   7  14   2   9]
 [  0   0  23 485   4  18   4   7  11   6]
 [  0   3  26   0 480   0   2  17   0   7]
 [  0   1  29   3   6 381   7   6   3  23]
 [  1   0  11   1   2   3 416   6   1 102]
 [  3   5  23   1  20   1  16 494   0   6]
 [  0   0   5   0   6   2   4   0 442   6]
 [  5   0   3   0   6   8  43   1   3 334]]
checking accuracy...
(220000, 0.93500000000000005, 0.88400000000000001)
Model saved in file: /tmp/model.ckpt
checking accuracy...
(230000, 0.94099999999999995, 0.89200000000000002)
Model saved in file: /tmp/model.ckpt
checking confusion
[[474   0   0   1   0   0   3   0   0   8]
 [  0 555   3   0   3   1   1   5   0   0]
 [  1   0 399   4  13  12  20  16   1   8]
 [  0   0   8 479   2  12   5   2   5   1]
 [  0   5  14   2 500   1   3  19   2   8]
 [  0   0  34   1   3 393   6   5   2  10]
 [  0   0   4   0   0   1 389   5   1 115]
 [  2   3  12   2   7   1   6 494   0   4]
 [  0   0   8   4   2   1   6   1 451   7]
 [  2   0   6   0   5  12  62   3   0 334]]
checking accuracy...
(240000, 0.93720000000000003, 0.89359999999999995)
Model saved in file: /tmp/model.ckpt
checking accuracy...
(250000, 0.93720000000000003, 0.88280000000000003)
Model saved in file: /tmp/model.ckpt
checking confusion
[[470   0   0   0   0   0   1   0   0   5]
 [  0 554   2   0   2   0   0   6   0   0]
 [  1   1 384   8  11  19   6   8   2   1]
 [  0   0   4 471   1   9   1   0   3   0]
 [  0   3  13   0 485   0   4  18   1   9]
 [  0   0  30   1   2 367   2   5   1   2]
 [  0   0  11   2   3   6 428  12   2 144]
 [  3   4  29   6  20   4  11 498   0   1]
 [  0   1   8   5   4   2   2   1 449   5]
 [  5   0   7   0   7  27  46   2   4 328]]
checking accuracy...
(260000, 0.94279999999999997, 0.88680000000000003)
Model saved in file: /tmp/model.ckpt
checking accuracy...
(270000, 0.93740000000000001, 0.89600000000000002)
Model saved in file: /tmp/model.ckpt
checking confusion
[[471   0   0   0   0   0   1   0   0   5]
 [  0 556   4   0   1   1   1   9   0   0]
 [  0   0 371   5   9  11   4   4   2   6]
 [  0   0   5 473   1   9   2   0   3   1]
 [  0   2  16   1 463   1   4  19   2   4]
 [  0   1  35   6   7 383   8   5   1  11]
 [  1   0  18   1   2   1 416   8   2 139]
 [  1   4  21   2  37   2  14 501   0   2]
 [  0   0   9   5   3   2   2   0 448   4]
 [  6   0   9   0  12  24  49   4   4 323]]
checking accuracy...
(280000, 0.93540000000000001, 0.88100000000000001)
Model saved in file: /tmp/model.ckpt
checking accuracy...
(290000, 0.93379999999999996, 0.88460000000000005)
Model saved in file: /tmp/model.ckpt
 
