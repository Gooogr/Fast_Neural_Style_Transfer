from keras.layers import Input, Conv2D, Conv2DTranspose, Activation,\
                         MaxPooling2D, BatchNormalization
from keras import layers
from keras.models import Model
from utilities import get_style_loss, get_content_loss, get_tv_loss

#### Custom layers ###

def residual_block(y, _nb_channels = 128, _strides = (1, 1), name_number = 1):
  shortcut = y

  #forward pass
  y = Conv2D(filters = _nb_channels, kernel_size = (3, 3), 
             strides = _strides, padding = 'same', 
             name = 're_conv1_{}'.format(name_number))(y)
  y = BatchNormalization(name = 're_batch1_{}'.format(name_number))(y)
  # ~ y = ReLU(name = 're_relu1_{}'.format(name_number))(y)
  y = Activation('relu', name = 're_relu1_{}'.format(name_number))(y)

  y = Conv2D(filters = _nb_channels, kernel_size = (3, 3), 
             strides = _strides, padding = 'same',
              name = 're_conv2_{}'.format(name_number))(y)
  y = BatchNormalization(name = 're_batch2_{}'.format(name_number))(y)

  #adding shortcut
  y = layers.add([shortcut, y])
  # ~ y = ReLU(name = 're_relu2_{}'.format(name_number))(y)
  y = Activation('relu')(y)

  return y
  
class OutputScale(layers.Layer):

    def __init__(self, **kwargs):
        super(OutputScale, self).__init__(**kwargs)

    def build(self, input_shape):
        pass

    def call(self, x, mask=None):
        return x * 150

    def compute_output_shape(self, input_shape):
        return input_shape


### Models builders ###

def get_training_model(height, width, batch_size):
  '''
  Create training model (with VGG-16 part).
  Input:
    height - input image height
    width - input image width
    batch_size - batch size
  Output:
    Non-compiled Keras model
  '''
  model_input = Input(shape = (height, width, 3), name = 'model_input')

  ### Autoencoder block ###
  # Convolution sub-blocks
  c1 = Conv2D(32, (9, 9), strides = 1, padding = 'same', name = 'conv_1')(model_input)
  c1 = BatchNormalization(name = 'norm_1')(c1)
  # ~ c1 = ReLU(name = 'relu_1')(c1)
  c1 = Activation('relu', name = 'relu_1')(c1)
  
  c2 = Conv2D(64, (3, 3), strides = 2, padding = 'same', name = 'conv_2')(c1)
  c2 = BatchNormalization(name = 'norm_2')(c2)
  # ~ c2 = ReLU(name = 'relu_2')(c2)
  c2 = Activation('relu', name = 'relu_2')(c2)

  c3 = Conv2D(128, (3, 3), strides = 2, padding = 'same', name = 'conv_3')(c2)
  c3 = BatchNormalization(name = 'norm_3')(c3)
  # ~ c3 = ReLU(name = 'relu_3')(c3)
  c3 = Activation('relu', name = 'relu_3')(c3)

  # Residual sub-blocks
  r1 = residual_block(c3, name_number = 1)
  r2 = residual_block(r1, name_number = 2)
  r3 = residual_block(r2, name_number = 3)
  r4 = residual_block(r3, name_number = 4)
  r5 = residual_block(r4, name_number = 5)

  # Deconvolution sub-blocks
  d1 = Conv2DTranspose(64, (3, 3), strides = 2, padding='same', name = 'conv_4')(r5)
  d1 = BatchNormalization(name = 'norm_4')(d1)
  # ~ d1 = ReLU(name = 'relu_4')(d1)
  d1 = Activation('relu', name = 'relu_4')(d1)

  d2 = Conv2DTranspose(32, (3, 3), strides = 2, padding='same', name = 'conv_5')(d1)
  d2 = BatchNormalization(name = 'norm_5')(d2)
  # ~ d2 = ReLU(name = 'relu_5')(d2)
  d2 = Activation('relu', name = 'relu_5')(d2)

  c4 = Conv2D(3, (9, 9), strides = 1, padding = 'same', name = 'conv_6')(d2)
  c4 = BatchNormalization(name = 'norm_6')(c4)
  c4 = Activation(activation = 'hard_sigmoid', name = 'hard_sigmoid_1')(c4) #hard sigmoid computes faster then sigmoid
  c4 = OutputScale(name = 'model_output')(c4)
   

  ### Content and style activation inputs
  #shapes correspond to VGG16 layers. Look on scheme of activations exits.
  content_activation = Input(shape = (height//4, width//4, 256))  #(64, 64, 256)

  style_activation_1 = Input(shape = (height, width, 64))         #(256, 256, 64)
  style_activation_2 = Input(shape = (height//2, width//2, 128))  #(128, 128, 128)
  style_activation_3 = Input(shape = (height//4, width//4, 256))  #(64, 64, 256)
  style_activation_4 = Input(shape = (height//8, width//8, 512))  #(32, 32, 512)

  ### VGG-16 block ###
  # Sub-block 1
  vgg_c1 = Conv2D(64, (3, 3), activation = 'relu', padding = 'same', name = 'block1_conv1')(c4) # use the same names like in the vgg16 network!
  vgg_c1 = Conv2D(64, (3, 3), activation = 'relu', padding = 'same', name = 'block1_conv2')(vgg_c1)
  style_loss1 = layers.Lambda(get_style_loss, output_shape = (1,), name = 'style_loss1', 
                              arguments = {'batch_size': batch_size})([vgg_c1, style_activation_1])
  vgg_c1 = MaxPooling2D((2, 2), strides = (2, 2), name = 'block1_pool')(vgg_c1)

  # Sub-block 2
  vgg_c2 = Conv2D(128, (3, 3), activation = 'relu', padding = 'same', name = 'block2_conv1')(vgg_c1)
  vgg_c2 = Conv2D(128, (3, 3), activation = 'relu', padding = 'same', name = 'block2_conv2')(vgg_c2)
  style_loss2 = layers.Lambda(get_style_loss, output_shape = (1,), name = 'style_loss2', 
                              arguments = {'batch_size': batch_size})([vgg_c2, style_activation_2])
  vgg_c2 = MaxPooling2D((2, 2), strides = (2, 2), name = 'block2_pool')(vgg_c2)


  # Sub-block 3
  vgg_c3 = Conv2D(256, (3, 3), activation = 'relu', padding = 'same', name = 'block3_conv1')(vgg_c2)
  vgg_c3 = Conv2D(256, (3, 3), activation = 'relu', padding = 'same', name = 'block3_conv2')(vgg_c3)
  vgg_c3 = Conv2D(256, (3, 3), activation = 'relu', padding = 'same', name = 'block3_conv3')(vgg_c3)
  style_loss3 = layers.Lambda(get_style_loss, output_shape = (1,), name = 'style_loss3', 
                              arguments = {'batch_size': batch_size})([vgg_c3, style_activation_3])


  content_loss = layers.Lambda(get_content_loss, output_shape = (1,), name = 'content_loss', 
                              arguments = {'batch_size': batch_size})([vgg_c3, content_activation])

  vgg_c3 = MaxPooling2D((2, 2), strides = (2, 2), name = 'block3_pool')(vgg_c3)

  # Sub-block 4
  vgg_c4 = Conv2D(512, (3, 3), activation = 'relu', padding = 'same', name = 'block4_conv1')(vgg_c3)
  vgg_c4 = Conv2D(512, (3, 3), activation = 'relu', padding = 'same', name = 'block4_conv2')(vgg_c4)
  vgg_c4 = Conv2D(512, (3, 3), activation = 'relu', padding = 'same', name = 'block4_conv3')(vgg_c4)
  style_loss4 = layers.Lambda(get_style_loss, output_shape = (1,), name = 'style_loss4', 
                              arguments = {'batch_size': batch_size})([vgg_c4, style_activation_4])
  vgg_c4 = MaxPooling2D((2, 2), strides = (2, 2), name = 'block4_pool')(vgg_c4)

  ### Total variation loss ###

  total_variation_loss = layers.Lambda(get_tv_loss, output_shape = (1,), name = 'tv_loss',
                                      arguments = {'width': width, 'height': height})(c4)

  ### Create model ###
  model = Model(inputs = [model_input, content_activation, style_activation_1, 
                          style_activation_2, style_activation_3, style_activation_4], 
                outputs = [content_loss, style_loss1, style_loss2, style_loss3, 
                          style_loss4, total_variation_loss, c4])
  return model

def get_pred_model(height, width):
  '''
  Create simplified autoencoder model, only for predictions
  Input:
    height - input image height
    width - input image width
    batch_size - batch size
  Output:
    Non-compiled Keras model
  '''
  model_input = Input(shape = (height, width, 3), name = 'model_input')

  ### Autoencoder block ###
  # Convolution sub-blocks
  c1 = Conv2D(32, (9, 9), strides = 1, padding = 'same', name = 'conv_1')(model_input)
  c1 = BatchNormalization(name = 'norm_1')(c1)
  # ~ c1 = ReLU(name = 'relu_1')(c1)
  c1 = Activation('relu', name = 'relu_1')(c1)
  
  c2 = Conv2D(64, (3, 3), strides = 2, padding = 'same', name = 'conv_2')(c1)
  c2 = BatchNormalization(name = 'norm_2')(c2)
  # ~ c2 = ReLU(name = 'relu_2')(c2)
  c2 = Activation('relu', name = 'relu_2')(c2)

  c3 = Conv2D(128, (3, 3), strides = 2, padding = 'same', name = 'conv_3')(c2)
  c3 = BatchNormalization(name = 'norm_3')(c3)
  # ~ c3 = ReLU(name = 'relu_3')(c3)
  c3 = Activation('relu', name = 'relu_3')(c3)

  # Residual sub-blocks
  r1 = residual_block(c3, name_number = 1)
  r2 = residual_block(r1, name_number = 2)
  r3 = residual_block(r2, name_number = 3)
  r4 = residual_block(r3, name_number = 4)
  r5 = residual_block(r4, name_number = 5)

  # Deconvolution sub-blocks
  d1 = Conv2DTranspose(64, (3, 3), strides = 2, padding='same', name = 'conv_4')(r5)
  d1 = BatchNormalization(name = 'norm_4')(d1)
  # ~ d1 = ReLU(name = 'relu_4')(d1)
  d1 = Activation('relu', name = 'relu_4')(d1)

  d2 = Conv2DTranspose(32, (3, 3), strides = 2, padding='same', name = 'conv_5')(d1)
  d2 = BatchNormalization(name = 'norm_5')(d2)
  # ~ d2 = ReLU(name = 'relu_5')(d2)
  d2 = Activation('relu', name = 'relu_5')(d2)

  c4 = Conv2D(3, (9, 9), strides = 1, padding = 'same', name = 'conv_6')(d2)
  c4 = BatchNormalization(name = 'norm_6')(c4)
  c4 = Activation(activation = 'hard_sigmoid', name = 'hard_sigmoid_1')(c4) #hard sigmoid computes faster then sigmoid
  c4 = OutputScale(name = 'model_output')(c4) 

  model = Model(inputs = model_input, outputs = c4)

  return model


# ~ test_train_model = get_training_model(height = 100, width = 100, batch_size = 4)
# ~ test_train_model.summary()

# ~ test_pred_model = get_pred_model(height = 100, width = 100)
# ~ test_pred_model.summary()
