from model_zoo import get_training_model, get_pred_model
from utilities import *
from keras.applications import vgg16
from keras import optimizers
import cv2
import os


def train(options):
	height = options['height']
	width = options['width']
	batch_size = options['batch_size']
	
	learning_rate = options['learning_rate']
	
	content_w = options['content_w'] 
	style_w1 = options['style_w1'] 
	style_w2 = options['style_w2'] 
	style_w3 = options['style_w3'] 
	style_w4 = options['style_w4'] 
	tv_w =  options['tv_w'] 
	output_w = options['output_w']
	
	style_img_path = options['style_img_path']
	test_content_img_path = options['test_content_img_path']
	
	style_layers = options['style_layers']
	content_layer = options['content_layer']
	
	# Build model
	model = get_training_model(height, width, batch_size)
	
	# Loading imagenet weights in our VGG16 model's part and frozing
	model_layers = {layer.name: layer for layer in model.layers}
	vgg_imagenet = vgg16.VGG16(weights = 'imagenet', include_top = False)
	vgg_imagenet_layers = {layer.name: layer for layer in vgg_imagenet.layers}
	for layer in vgg_imagenet.layers:
		if layer.name in model_layers:
			# load imagenet weights in model layer
			model_layers[layer.name].set_weights(vgg_imagenet_layers[layer.name].get_weights())
			# froze layer
			model_layers[layer.name].trainable = False
		
	# Compile model
	model_loss_weights = [content_w, style_w1, style_w2, style_w3, style_w4, tv_w, output_w]
	model_optimizer = optimizers.Adam(lr = learning_rate) 							  
	model_loss = {'content_loss': dummy_loss, 'style_loss1': dummy_loss, 'style_loss2': dummy_loss,
				  'style_loss3': dummy_loss, 'style_loss4': dummy_loss, 'tv_loss': dummy_loss, 
				  'model_output': zero_loss}
	model.compile(loss = model_loss, optimizer = model_optimizer, loss_weights = model_loss_weights)
	print('Model succesfully compiled!')
	
	#Getting activation to fit them in our model
	
	# Get style activations
	style_tensor = preprocess_img(style_img_path, height, width)
	style_act = []
	for layer_name in style_layers:
		style_function = get_func_extract_vgg_activations(layer_name, width, height)
		style_activation = expand_batch_input(batch_size, style_function([style_tensor])[0])
		style_act.append(style_activation)

	# Get content activations for test image
	content_test_tensor = preprocess_img(test_content_img_path, height, width)
	content_function = get_func_extract_vgg_activations(content_layer, width, height)
	content_test_activation = expand_batch_input(batch_size, content_function([content_test_tensor])[0])
	content_test = expand_batch_input(batch_size, content_test_tensor)
	
	# ADD HERE TRAIN LOOP AND MODEL SAVING!
	
		 
def predict(options):
	img_path = options['img_path']
	weights_path = options['weights_path']
	result_dir = options['result_dir']
	
	img = preprocess_img(img_path, img_height = None, img_width = None, resize_img = False) 
	original_img_height = img.shape[1]
	original_img_width = img.shape[2]

	# Make image padding 
	padded_img, paddings = make_padding(img)
	print('padings', paddings)
	padded_img_height = padded_img.shape[1]
	padded_img_width = padded_img.shape[2]

	# Create autoencoder with properly shapes
	pred_model = get_pred_model(padded_img_height,padded_img_width)
	pred_model.load_weights(weights_path)

	# Make prediction
	prediction = pred_model.predict(padded_img)
	print('Prediction shape', prediction.shape)
	result_img = deprocess_img(prediction, padded_img_height, padded_img_width)

	result_img = remove_padding(result_img, original_img_height, original_img_width, paddings)
	print('Shape after unpadding', result_img.shape)
	
	cv2.imwrite(os.path.join(result_dir, 'result.png'), result_img)
	# ~ cv2_imshow(result_img)
