# DeepCGNCE
2022本科毕业论文. 

用于根据蛋白质天然态下C$\alpha$结构预测残基对接触能. DeepCGNCE是指Deep learning-based Coarse-Grained Native Contact Energy. 

模型基于一个简单的残差网络结构，不大准. 在SCOPe2.08上选了12276个蛋白，8:1:1分割. 对最好的模型，测试集上Pearson相关系数是0.87. 

各文件夹内容如下：
- CGmap：用于做粗粒化映射，ProCG.py用于读入蛋白结构，对其检查、粗粒化映射或编码. 检查和粗粒化映射需要一个规则文件，文件夹中那些``.cgin``文件就是不同的规则文件. 
- models：包含用tensorflow2.1.0 api实现的层、激活函数、优化器等，用于搭建神经网络模型的组件. 以及模型的封装、训练与参数文件. 参数文件定义了模型结构与训练参数等，一个参数文件就是一个模型架构. 参数文件存放在architectures/目录下. 
