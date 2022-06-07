# Dis-PU-pytorch
Pytorch unofficial implementation of Dis-PU

(Point Cloud Upsampling via Disentangled Refinement)

https://arxiv.org/abs/2106.04779




# Model Structure
**Model**  
![ex_screenshot](./imgs/model.png)

**Detail**  
![ex_screenshot](./imgs/detail.png)


# Evaluation
**Input : 2048**  
**Output : 8192**  
**Test Dataset : PU-GAN dataset**    

| X4 | Chamfer Distance(10<sup>-3</sup>)|HD(10<sup>-3</sup>)|P2F(10<sup>-3</sup>)|
|:--------|:--------:|:--------:|:--------:|
| This code | **0.2702**|**5.50**|**4.35**|
|Paper|**0.315**|**4.201**|**4.141**| 


# Visualize
**Ground Truth**  
![ex_screenshot](./imgs/cat_gt.png)

**Model Prediction**  
![ex_screenshot](./imgs/cat_predict.png)
