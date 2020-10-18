# NAIC2020
NAIC2020 code

在初赛题目中，我们首先尝试了Lu Z , Wang J , Song J . Multi-resolution CSI Feedback with deep learning in Massive MIMO System[J]. arXiv, 2019. 等论文给出的多分辨率的神经网络结构。 发现多分辨率且包含Residu 结构的神经网络的效果比较好。 其次，由于初赛训练集的规模较大，不太可能出现过拟合。因此，增大网络结构，例如加大卷积核规模和数量，增大卷积层层数等等。 最后， 许多论文中给出了不对称的Autoencoder结构，认为decoder比encoder更为重要，这一点我们也做过一些尝试，发现确实如此。
