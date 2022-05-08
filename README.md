# DeepCGNCE
2022本科毕业论文. 

用于根据蛋白质天然态下C$\alpha$结构预测残基对接触能. DeepCGNCE是指Deep learning-based Coarse-Grained Native Contact Energy. 

模型基于一个简单的残差网络结构，不大准. 在SCOPe2.08上选了12276个蛋白，8:1:1分割. 对最好的模型，测试集上Pearson相关系数是0.87. 

各文件夹内容如下：
- CGmap：用于做粗粒化映射，ProCG.py用于读入蛋白结构，对其检查、粗粒化映射或编码. 检查和粗粒化映射需要一个规则文件，文件夹中那些``.cgin``文件就是不同的规则文件. 
- ResNet6-18：以卷积残差网络为backbone，包含6个一维卷积层和18个二维卷积层. 
- ResNet12-25：以卷积残差网络为backbone，包含12个一维卷积层和25个二维卷积层. 
- LambdaResNet12-25：和ResNet12-25差不多，一些卷积层用Lambda卷积层替代，Lambda卷积层参考LambdaNet设计. 
