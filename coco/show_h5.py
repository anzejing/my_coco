import h5py
# 导入工具包
import numpy as np

# HDF5的读取：
f = h5py.File(r'D:\1file\Mask_RCNN\mask_rcnn_coco.h5', 'r')  # 打开h5文件
if len(f.attrs.items()):
    print("{} contains: ".format('D:/net.h5'))
    print("Root attributes:")
for key, value in f.attrs.items():
    print(" {}: {}".format(key, value))  # 输出储存在File类中的attrs信息，一般是各层的名称

for layer, g in f.items():  # 读取各层的名称以及包含层信息的Group类
    print(" {}".format(layer))
    print("  Attributes:")
    for key, value in g.attrs.items():  # 输出储存在Group类中的attrs信息，一般是各层的weights和bias及他们的名称
        print("   {}: {}".format(key, value))

    print("  Dataset:")
    for name, d in g.items():  # 读取各层储存具体信息的Dataset类
        print("   {}: {}".format(name, d.value.shape))  # 输出储存在Dataset中的层名称和权重，也可以打印dataset的attrs，但是keras中是空的
        print("   {}: {}".format(name.d.value))
f.close()