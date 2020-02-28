
### Fast Neural Style Transfer project
This project is my attempt to solve the problem of quick style transfer by myself. While creating it, I moved step by step, gradually studying new papers. <br>
Jupyter notebooks for each step you can find in the [research_notebooks](https://github.com/Gooogr/Keras_Fast_Style_Transfer/tree/master/research_notebooks) folder

**Step 1: Simple algorithm based on original paper "A Neural Algorithm of Artistic Style"**<br>
* Original article: ["A Neural Algorithm of Artistic Style"](https://arxiv.org/abs/1508.06576)

My first implementation was based on the article "A Neural Algorithm of Artistic Style".
I took code from  Francois Chollet's book 'Deep Learning with Python' and adapted for Google Collab.<br>
It workes pretty well, but it takes about 2 hours to get one image.

<img src = "https://github.com/Gooogr/Keras_Fast_Style_Transfer/blob/master/img/dogs.jpg" width = "250" /> <img src = "https://github.com/Gooogr/Keras_Fast_Style_Transfer/blob/master/img/night.jpg" width = "250" /> 
<img src = "https://github.com/Gooogr/Keras_Fast_Style_Transfer/blob/master/img/Result%20(iteration_999).png" width = "250" />


**Step 2: Setting up autoencoder from "Perceptual Losses for Real-Time Style Transfer and Super-Resolution"**<br>
* Original article: ["Perceptual Losses for Real-Time Style Transfer and Super-Resolution"](https://arxiv.org/abs/1603.08155)<br>
* Supplementary Material: ["Perceptual Losses for Real-Time Style Transfer
and Super-Resolution: Supplementary Material"](https://cs.stanford.edu/people/jcjohns/papers/fast-style/fast-style-supp.pdf)

The approach described in the article turned out to be rather complicated, that's why I singled out a subtask - the launch of the autoencoder architecture proposed in the article. Its specific feature is the absence of pooling layers.<br>
It works, and in the process of creating it, I thought why not use it separately, without using any specific per-pixel or perceptual losses.<br>
So, the plan was simple: generate some style transfer examples from the first step algorithm and train autoencoder on them with data augmentation.

**Step 3-4: Naive fast style transfer based on the autoencoder**

The original input/output size of the autoencoder was 256x256, but I changed it on 512x512 for better picture quality. I prepared 15 pairs of images and trained my autoencoder with data augmentation. Train dataset was created by generator, based on the algorithm from the step 1<br>

Train examples:<br>

<img src = "https://github.com/Gooogr/Keras_Fast_Style_Transfer/blob/master/img_encoders_pairs/original_images/dummy_folder/bridge.jpg" width = "250" /> <img src = "https://github.com/Gooogr/Keras_Fast_Style_Transfer/blob/master/img_encoders_pairs/original_images/dummy_folder/castle.jpg" width = "250" /> 
<img src = "https://github.com/Gooogr/Keras_Fast_Style_Transfer/blob/master/img_encoders_pairs/original_images/dummy_folder/sea.jpg" width = "250" />

<img src = "https://github.com/Gooogr/Keras_Fast_Style_Transfer/blob/master/img_encoders_pairs/generated_results/dummy_folder/bridge_gen.png" width = "250" /> <img src = "https://github.com/Gooogr/Keras_Fast_Style_Transfer/blob/master/img_encoders_pairs/generated_results/dummy_folder/castle_gen.png" width = "250" /> 
<img src = "https://github.com/Gooogr/Keras_Fast_Style_Transfer/blob/master/img_encoders_pairs/generated_results/dummy_folder/sea_gen.png" width = "250" />



Test examples:<br>

<img src = "https://github.com/Gooogr/Keras_Fast_Style_Transfer/blob/master/img_encoders_pairs/test_images/dummy_folder/red_bridge.png" width = "250" /> <img src = "https://github.com/Gooogr/Keras_Fast_Style_Transfer/blob/master/img_encoders_pairs/test_images/dummy_folder/bridge.jpg" width = "250" /> 
<img src = "https://github.com/Gooogr/Keras_Fast_Style_Transfer/blob/master/img_encoders_pairs/test_images/dummy_folder/branches.jpg" width = "250" />

<img src = "https://github.com/Gooogr/Keras_Fast_Style_Transfer/blob/master/img_encoders_pairs/test_results/red_bridge.png" width = "250" /> <img src = "https://github.com/Gooogr/Keras_Fast_Style_Transfer/blob/master/img_encoders_pairs/test_results/bridge.jpg" width = "250" /> 
<img src = "https://github.com/Gooogr/Keras_Fast_Style_Transfer/blob/master/img_encoders_pairs/test_results/branches.jpg" width = "250" />

Model was trained for 20 epochs per 500 iterations in the each. 

**Step 4:  Implementation of Perceptual Losses for Real-Time Style Transfer and Super-Resolution**
