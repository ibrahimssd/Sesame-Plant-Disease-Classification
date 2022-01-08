# Sesame-Plant-Disease-Classification

In  this  project we explore  the  plant  disease  classification77problem on a dataset of sesame leaves of 1453 images, containing three88classes,  including  healthy  plants.  Resnet50  architecture  pre-trained  on99ImageNet was used, in combination with other methods and techniques1010to increase the test accuracy and overcome overfitting and the challenge1111of  small  dataset  size.  Experiments  using  data  augmentation,  transfer1212learning and self-supervised pre-training were carried out. Empirical re-1313sults  showed  that  data  augmentation  and  multi-step  transfer  learning1414with a dataset that shares similarity (both are leaves in this case) showed1515the best performance (pre-trained weights on ImageNet, followed by pre-1616training on the Cassava Leaf Challenge dataset), reaching an accuracy of171796.71%. Adding a self-supervised angle classification pre-training phase1818decreased the performance, which can be explained by the learned fea-1919tures being less useful than the ones from transfer learning. Experiments2020on freezing the model and training only the last layer were also performed2121and analyzed, achieving a result of 90.9%. The significantly high success2222rate makes the model a very useful advisory or early warning tool, and an2323approach that could be further expanded to support an integrated plant2424disease identification system to operate in real cultivation conditions

Sesame Dataset:
https://drive.google.com/drive/folders/1f3YgVhelsXuyeCosMlKFpiCwrSzcLPxl?usp=sharing

Code used to prepare dataset:
https://colab.research.google.com/drive/1WzaqmxBlF5AETmqDR1IxrmE19qPuvwx1?usp=sharing
