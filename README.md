# 3D-TransU-Net-framwork
The 3D TransU-Net framwork can predict the multi-component stress fields for the RVE geometries of fiber composites with diverse fiber volume fractions and different input load paths.
# How to work 
1.The first step is to create image and mask for the training and crop them to the specified matrix size (N, 128, 128, T, C).
2.The second step is to use data augmentation to expand training set and normalize the data.
3.The final step is to train the 3D TransU-Net framework using paired samples.
# Presentation of training results
![image](https://github.com/pengxiangbobin/3D-TransU-Net-framwork/blob/main/Figure.png)
