
## Fast Neural Style Transfer project
This project is my attempt to solve the problem of quick style transfer by myself. While creating it, I moved step by step, gradually studying new papers. <br>
Jupyter notebooks for each step you can find in the [research_notebook](https://github.com/Gooogr/Keras_Fast_Style_Transfer/tree/master/research_notebooks) page.<br>
Also you can find all notebooks in my [Google Collab](https://drive.google.com/drive/folders/1t81jrL_823o9ZCqwGAgKfmHPSHiM4Nsr?usp=sharing) project folder. The source of the development was there.

Original article: ["Perceptual Losses for Real-Time Style Transfer and Super-Resolution"](https://arxiv.org/abs/1603.08155)<br>
Supplementary Material: ["Perceptual Losses for Real-Time Style Transfer
and Super-Resolution: Supplementary Material"](https://cs.stanford.edu/people/jcjohns/papers/fast-style/fast-style-supp.pdf)

### Result

| | | | |
|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|
|  |<img width="1604" src="https://github.com/Gooogr/Keras_Fast_Style_Transfer/blob/master/research_notebooks/img_fst_results/bridge.jpg">|<img width="1604" src="https://github.com/Gooogr/Keras_Fast_Style_Transfer/blob/master/research_notebooks/img_fst_results/dear.png">|<img width="1604" src="https://github.com/Gooogr/Keras_Fast_Style_Transfer/blob/master/research_notebooks/img_fst_results/red_bridge.png">|
| <img width="1604" src="https://github.com/Gooogr/Keras_Fast_Style_Transfer/blob/master/research_notebooks/img_fst_results/night.jpg"> |<img width="1604" src="https://github.com/Gooogr/Keras_Fast_Style_Transfer/blob/master/research_notebooks/img_fst_results/bridge_n.png">|<img width="1604" src="https://github.com/Gooogr/Keras_Fast_Style_Transfer/blob/master/research_notebooks/img_fst_results/dear_n.png">|<img width="1604" src="https://github.com/Gooogr/Keras_Fast_Style_Transfer/blob/master/research_notebooks/img_fst_results/red_bridge_n.png">|
| <img width="1604" src="https://github.com/Gooogr/Keras_Fast_Style_Transfer/blob/master/research_notebooks/img_fst_results/draft.jpg"> |<img width="1604" src="https://github.com/Gooogr/Keras_Fast_Style_Transfer/blob/master/research_notebooks/img_fst_results/bridge_d.png">|<img width="1604" src="https://github.com/Gooogr/Keras_Fast_Style_Transfer/blob/master/research_notebooks/img_fst_results/dear_d.png">|<img width="1604" src="https://github.com/Gooogr/Keras_Fast_Style_Transfer/blob/master/research_notebooks/img_fst_results/red_bridge_d.png">|
| <img width="1604" src="https://github.com/Gooogr/Keras_Fast_Style_Transfer/blob/master/research_notebooks/img_fst_results/kandinskiy.jpg"> |<img width="1604" src="https://github.com/Gooogr/Keras_Fast_Style_Transfer/blob/master/research_notebooks/img_fst_results/bridge_k.png">|<img width="1604" src="https://github.com/Gooogr/Keras_Fast_Style_Transfer/blob/master/research_notebooks/img_fst_results/dear_k.png">|<img width="1604" src="https://github.com/Gooogr/Keras_Fast_Style_Transfer/blob/master/research_notebooks/img_fst_results/red_bridge_k.png">|

### Environment (main packages):
* Keras==2.2.5
* opencv-python==4.1.2.30
* Pillow==7.0.0
* tensorboard==1.15.0
* tensorflow==1.5.2
* tensorflow-estimator==1.15.1

### How to use it
There  are two ways to use this neural network. You can train or predict in [this](https://github.com/Gooogr/Keras_Fast_Style_Transfer/blob/master/research_notebooks/5.2\)%20Fast%20Style%20Transfer.ipynb) google collab file or use a python script.

If you want to train your version of neural  network, you have to download train part of the [COCO 2014 dataset](http://images.cocodataset.org/zips/train2014.zip). 

For training the network:<br>
```
python main.py train --conf [path to config json file, optional]
```
For predicting: <br>
```
python main.py predict -w [path to the pre-trained weights] -i [path to the content image] -r [directory where to save result, optional]
```

Explanation some of the config.json parametres:
* `"net_name"` - It will be the name of saved weights after training
* `"height", "weight"` - Size for re-shaped images during the training. It were 256 x 256 in the original article, but I used 512x512 for better results
* `"verbose_iter"` - Determine how often you will print training info and save test image

Default files structure
```
project
│   README.md
|   LICENSE
│   main.py
|   model_functions.py
|   model_zoo.py  
|   utilities.py
│
└───img
│   │   style_img.jpg
│   │   content_img.jpg
|   |   test_content_img.jpg
│   │
│   └───iteration_results
│   
└───dataset
|   |
|   └───train2014
|       |   # 82 000 images
|
└───saved_weights
    |   fst_draft_512_weights.h5
    |   fst_kandinskiy_512_weights.h5
    |   fst_night_256_weights.h5
    |   fst_night_512_weights.h5
```

### Perfomance
It takes about 10 hours to train a network on a Nvidia K80 GPU (Google Collab).
