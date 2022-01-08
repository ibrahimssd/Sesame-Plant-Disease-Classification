# Sesame-Plant-Disease-Classification

In this project we explore the plant disease classification problem on a dataset of sesame leaves of 1453 images, containing three classes, including healthy plants. Resnet50 architecture pre-trained on ImageNet was used, in combination with other methods and techniques to increase the test accuracy and overcome overfitting and the challenge of small dataset size. Experiments using data augmentation, transfer learning and self-supervised pre-training were carried out. Empirical results showed that data augmentation and multi-step transfer learning with a dataset that shares similarity (both are leaves in this case) showed the best performance (pre-trained weights on ImageNet, followed by pre-training on the Cassava Leaf Challenge dataset), reaching an accuracy of 96.71\%. Adding a self-supervised angle classification pre-training phase decreased the performance, which can be explained by the learned features being less useful than the ones from transfer learning. Experiments on freezing the model and training only the last layer were also performed and analyzed, achieving a result of 90.9\%. The significantly high success rate makes the model a very useful advisory or early warning tool, and an approach that could be further expanded to support an integrated plant disease identification system to operate in real cultivation conditions.

Sesame Dataset:
https://drive.google.com/drive/folders/1f3YgVhelsXuyeCosMlKFpiCwrSzcLPxl?usp=sharing

Code used to prepare dataset:
https://colab.research.google.com/drive/1WzaqmxBlF5AETmqDR1IxrmE19qPuvwx1?usp=sharing
