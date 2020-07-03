import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.preprocessing.image import load_img, img_to_array
from keras.applications import vgg16

### Mathematical functions ###
def gram_matrix(x):
    '''
    Calculate Gram matrix 
    http://pmpu.ru/vf4/dets/gram
    '''
    # K.permute_dimensions - Permutes axes in a tensor. https://www.tensorflow.org/api_docs/python/tf/keras/backend/permute_dimensions
    # K.batch_flatten - Turn a nD tensor into a 2D tensor with same 0th dimension. https://www.tensorflow.org/api_docs/python/tf/keras/backend/batch_flatten
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram

### Loss functions ###

def get_content_loss(activations, **kwargs):
  '''
  Calculate content (feature) loss of vgg16 content activation and 
  correspond layer from our model.The loss is the Euclidian distance (L2 norm).
  '''
  layer_activation, content_activation = activations
  batch_size = kwargs['batch_size']

  shape = content_activation.shape
  H, W, C = shape[1], shape[2], shape[3]

  loss = K.variable(0)
  for idx in range(batch_size):
    loss = loss + (K.sum(K.square(layer_activation[idx] - content_activation[idx])) / (H * W * C).value)
  return loss


def get_style_loss(activations, **kwargs):
  '''
  Calculate style loss.
  '''
  layer_activation, style_activation = activations
  batch_size = kwargs['batch_size']

  shape = style_activation.shape.dims
  H, W, C = shape[1], shape[2], shape[3]

  loss = K.variable(0)
  for idx in range(batch_size):
    layer_gram = gram_matrix(layer_activation[idx])
    style_gram = gram_matrix(style_activation[idx])
    loss = loss + K.sum(K.square(layer_gram - style_gram)) / ((H * W * C).value**2)
  return loss


def get_tv_loss(x, **kwargs):
  '''
   Calculate total variation loss. Reduce puzzling effect
  '''
  height = kwargs['height']
  width = kwargs['width']
  a = K.square(x[:, :height - 1, :width - 1, :] - 
               x[:, 1:, :width - 1, :])
  b = K.square(x[:, :height - 1, :width - 1, :] - 
               x[:, :height - 1, 1:, :])
  print(a.shape, b.shape)
  return K.sum(K.pow(a+b, 1.25))
  
def dummy_loss(y_true, y_pred):
  '''
  Use to initialize loss functions and compile model.
  Input variables y_true, y_pred are demanded by keras, even if they doesn`t return
  Read more here https://towardsdatascience.com/advanced-keras-constructing-complex-custom-losses-and-metrics-c07ca130a618
  '''
  return y_pred

def zero_loss(y_true, y_pred):
  '''
  Initialize VGG-16 inputs by zeros
  '''
  return K.variable(np.zeros(1,))
  
### Image processing/deprocessing ###

def preprocess_img(img_file, img_height, img_width, resize_img = True):
  ''' 
  Convert raw image to [1, img_width, imh_height, 3] image prepared to VGG16
  img_file could be:
   file path - is used when running from terminal
   PIL image - is used when running  under Streamlit web app
  '''
  if isinstance(img_file, str):
    if resize_img:
      img_file = load_img(img_file, target_size = (img_height, img_width))
    else:
      img_file = load_img(img_file)
  img = img_to_array(img_file)
  img = np.expand_dims(img, axis = 0)
  img = vgg16.preprocess_input(img)
  return img


def deprocess_img(x, height, width): 
  '''
  Deprocess  model output into image. Image compatible with OpenCV.
  Input:
    x - raw model output
    height - wishful image height
    width - wishful image width
  Output:
    img - result image, ready for OpenCV
  '''
  
  x = x.reshape([height, width, 3])
  # Remove zero-pixel mean value
  # Read more here: https://github.com/keras-team/keras-applications/blob/master/keras_applications/imagenet_utils.py
  x[:, :, 0] += 103.939
  x[:, :, 1] += 116.779
  x[:, :, 2] += 123.68
  # BGR -> RGB
  #x = x[:, :, ::-1] # Made a color error in OpenCV
  img = np.clip(x, 0, 255).astype('uint8')
  return img
  
def make_padding(img):
  '''
  Pad image to proper size
  '''
  img_height = img.shape[1]
  img_width = img.shape[2]
  print('Original image size:', img.shape)

  # Calculate paddings for our image. We have to provide divisibility on 4 (based on autoncoder architecture)
  double_pad_height = (img_height//128 + 2) * 128 - img_height # found constant "128" experimentally
  double_pad_width = (img_width//128 + 2) * 128 - img_width

  pad_height_1 = int(np.floor(double_pad_height / 2.0))
  pad_height_2 = int(np.ceil(double_pad_height / 2.0))
  pad_width_1 = int(np.floor(double_pad_width / 2.0))
  pad_width_2 = int(np.ceil(double_pad_width / 2.0))

  # Pad image
  padding = ((0,0), (pad_height_1, pad_height_2), (pad_width_1, pad_width_2), (0,0))
  padded_img = np.pad(img, padding, 'reflect')
  print('Padded image size:', padded_img.shape)
  return padded_img, padding

def remove_padding(img, original_height, original_width, paddings):
  '''
  Unpad image to the original size
  '''
  height_paddings = paddings[1]
  width_paddings = paddings[2]

  unpadded_image = img[height_paddings[0]:original_height + height_paddings[0], 
                       width_paddings[0]:original_width + width_paddings[0], :] 
  return unpadded_image

def url_to_image(url, preprocess_image = True):
  '''
  Download the image by url-link, convert it to a NumPy array. Encode it into OpenCV format or preprocess it for VGG16 architecture
  input:
    url - direct url link to image 
    preprocess_image - boolean flag. Determine if you want to get simple image ready for OpenCV or encoded matrix for VGG16
  output:
    OpenCV image or matrix encoded for VGG16 input.
  '''
  resp = urllib.request.urlopen(url)
  image = np.asarray(bytearray(resp.read()), dtype="uint8")

  image = cv2.imdecode(image, cv2.IMREAD_COLOR)
  if preprocess_image:
    img = img_to_array(image)
    img = np.expand_dims(img, axis = 0)
    img = vgg16.preprocess_input(img)
    return img
  else:
    return image

### Inner neuralnetwork functions ###
 
def get_func_extract_vgg_activations(layer_name, width, height):
  '''
  Return function which input is a placeholder with shape [1, height, weight, 3]
  and output is selected layer output of VGG16 newtwork.
  This layer can be feeded through the placeholder.
  '''
  tensor = K.placeholder([1, height, width, 3])
  temp_model = vgg16.VGG16(input_tensor = tensor, weights = 'imagenet', 
                           include_top = False)
  for layer in temp_model.layers:
    if layer.name == layer_name:
      layer.trainable = False
      return K.function([tensor], [layer.output])

def expand_batch_input(batch_size, initial_input):
  '''
  Expand batch dimension of selected input by copy-paste initial matrix.
  [1, 256, 256, 3] -> [4, 256,  256, 3] if butch_size = 4
  Input:
    "batch_size" - nessesary amount of batches 
    "initial_input" - input that should be expanded
  Output:
    expanded input
  '''
  expanded_input = initial_input.copy()
  for step in range(batch_size - 1):
    expanded_input = np.append(expanded_input, initial_input, axis = 0)
  return expanded_input


### Printing functions ###
def print_training_loss(history_data):
  '''
  Print out current loss.
  input:
    history_data  - keras History object. Get it from model_name.history .
  output:
    None
  '''
  print('Training loss details')
  print('Content loss: {}'.format(history.history['content_loss_loss'][0]))
  print('Style loss_1: {}, Style loss_2: {}'.format(history.history['style_loss1_loss'][0], history.history['style_loss2_loss'][0]))
  print('Style loss_3: {}, Style loss_4: {}'.format(history.history['style_loss3_loss'][0], history.history['style_loss4_loss'][0]))
  print('Total variation loss: {}'.format(history.history['tv_loss_loss'][0]))
  print('----------------------------------------')


def print_test_info(verbose_result):
  '''
  Print out loss information of the test image during the training loop.
  input:
    verbose_result - current model`s prediction. Contain losses information and output image.
  output:
    None
  '''
  loss_list = []
  loss_list.append(verbose_result[0][0] * content_w) 
  loss_list.append(verbose_result[1][0] * style_w1)
  loss_list.append(verbose_result[2][0] * style_w2) 
  loss_list.append(verbose_result[3][0] * style_w3) 
  loss_list.append(verbose_result[4][0] * style_w4) 
  loss_list.append(verbose_result[5][0] * tv_w) 
  current_loss = sum(loss_list)
  print('----------------------------------------')
  print('Current test image losses')
  print('Current sum loss: {}'.format(current_loss))
  print('Losses parts:')
  print('Content loss: {}'.format(loss_list[0]))
  print('Style loss_1: {}, Style loss_2: {}, \nStyle loss_3: {}, Style loss_4: {}'.format(loss_list[1], loss_list[2], loss_list[3], loss_list[4]))
  print('Total variation loss: {}'.format(loss_list[5]))
