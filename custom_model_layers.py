from keras.layers import Input, Conv2D, Conv2DTranspose, Activation,\
                         MaxPooling2D, BatchNormalization
from keras import layers

def residual_block(y, _nb_channels = 128, _strides = (1, 1), name_number = 1):
  shortcut = y

  #forward pass
  y = Conv2D(filters = _nb_channels, kernel_size = (3, 3), 
             strides = _strides, padding = 'same', 
             name = 're_conv1_{}'.format(name_number))(y)
  y = BatchNormalization(name = 're_batch1_{}'.format(name_number))(y)
  # ~ y = ReLU(name = 're_relu1_{}'.format(name_number))(y)
  y = Activation('relu')(y)

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

print('file compiled!')
