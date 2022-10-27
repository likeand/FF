| 实验                          |结果     | 备注     |
| ----                          | ---- | ---- |
|  resnet18 320聚类             |  {'resnet18_320_mIoU': 6.165891140699387, 'resnet18_320_Accuracy': 21.70196622610092, 'resnet18_320_Nonzero%': 99.99999980484363}    |  embed dim 256  bs 16   |
|  resnet50 320聚类             |  {'resnet50_320_mIoU': 6.812487542629242, 'resnet50_320_Accuracy': 24.76217895746231, 'resnet50_320_Nonzero%': 99.99999980484363}    |  embed dim 256  bs 16   |
|  resnet18 640聚类             |  {'resnet18_640_mIoU': 9.267111122608185, 'resnet18_640_Accuracy': 34.338751435279846, 'resnet18_640_Nonzero%': 99.99999980484363}    |   embed dim 256   bs 8 |
|  resnet50 640聚类             |  {'resnet50_640_mIoU': 9.48382318019867, 'resnet50_640_Accuracy': 29.006344079971313, 'resnet50_640_Nonzero%': 99.99999980484363}    |   embed dim 256  bs 8  |
|  dino 320 聚类                |  {'dino_320_mIoU': 12.777101993560791, 'dino_320_Accuracy': 42.42326021194458, 'dino_320_Nonzero%': 99.99999980484363}    |  vit base embed dim = 768  bs 4  |
|  dino 640 聚类                |      |      |
|  resnet不同层的聚类           |      |      |
|  resnet18使用cam的方法 聚类         |  {'resnet18_320_cammIoU': 6.627695262432098, 'resnet18_320_camAccuracy': 23.551766574382782, 'resnet18_320_camNonzero%': 99.99999980484363}    |   改进前，尝试多次运行取较好结果，每次结果随机的   |
|resnet50使用cam的方法 聚类|{'resnet50_320_cammIoU': 6.995102018117905, 'resnet50_320_camAccuracy': 25.048112869262695, 'resnet50_320_camNonzero%': 99.99999980484363}|改进后|
|  resnet18 使用多尺度的方法 聚类    |   {'resnet18_320_multiscalemIoU': 9.91184338927269, 'resnet18_320_multiscaleAccuracy': 35.589879751205444, 'resnet18_320_multiscaleNonzero%': 77.56618908473575}  |   bs 4 embed dim 256   |
|resnet 50 使用多尺度的方法 聚类| {'resnet50_320_multiscalemIoU': 10.782629996538162, 'resnet50_320_multiscaleAccuracy': 37.124186754226685, 'resnet50_320_multiscaleNonzero%': 77.56618908473575}      {'gen_files/resnet50_320_multiscalemIoU': 10.68727895617485, 'gen_files/resnet50_320_multiscaleAccuracy': 39.72204923629761, 'gen_files/resnet50_320_multiscaleNonzero%': 77.56618908473575}| bs 4 embed dim 256|
|  resnet使用cam+多尺度的方法 聚类    |   {'resnet18_320_cam_multiscalemIoU': 5.245412141084671, 'resnet18_320_cam_multiscaleAccuracy': 19.674259424209595, 'resnet18_320_cam_multiscaleNonzero%': 99.99999980484363}   |    改进前  |
|resnet50使用cam+多尺度的方法 聚类|{'resnet50_320_cam_multiscalemIoU': 7.2747498750686646, 'resnet50_320_cam_multiscaleAccuracy': 28.292104601860046, 'resnet50_320_cam_multiscaleNonzero%': 99.99999980484363}| 改进后|
| swinv2的方法 384 | {'swinv2_384_mIoU': 8.15533623099327, 'swinv2_384_Accuracy': 27.52838432788849, 'swinv2_384_Nonzero%': 99.99999980484363} | 384x |
|  swinv2使用cam的方法 聚类         |  {'swinv2_384_cammIoU': 11.391163617372513, 'swinv2_384_camAccuracy': 38.67080509662628, 'swinv2_384_camNonzero%': 99.99999980484363}    |      |
|  swinv2使用多尺度的方法 聚类    |   {'swinv2_384_multiscalemIoU': 10.983266681432724, 'swinv2_384_multiscaleAccuracy': 35.64129173755646, 'swinv2_384_multiscaleNonzero%': 77.56618908473575}   |      |
|  swinv2使用cam+多尺度的方法 聚类    |   {'swinv2_384_cam_multiscalemIoU': 9.470033645629883, 'swinv2_384_cam_multiscaleAccuracy': 30.42462170124054, 'swinv2_384_cam_multiscaleNonzero%': 99.99999980484363}   |      |
|  dino 320 聚类                  |  {'test_on_valmIoU': 14.25454467535019, 'test_on_valAccuracy': 42.68613159656525}    |      |

