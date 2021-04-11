# SOLOV2 ncnn

## the C++ version of SOLOV2 with ncnn

## [知乎](https://zhuanlan.zhihu.com/p/361900997?utm_source=qq&utm_medium=social&utm_oi=872955404320141312)


## ncnn 模型
链接: https://pan.baidu.com/s/1W1AiKdI4JJq2LW50uGOVng  密码: phh8


```
mkdir build
cd build 
cmake ..
make 
./solov2 ../imgs/horses.jpg
./solov2_fast ../imgs/horses.jpg
```

## 结果展示
![avatar](imgs/result.jpg)

## mbv2_slov2 coco val result (short size=448)

Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.254
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.412
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.266
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.062
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.267
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.429
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.251
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.389
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.407
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.140
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.467
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.647

