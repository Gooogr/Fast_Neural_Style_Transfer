from model_zoo import get_training_model, get_pred_model
from utilities import *
from keras.applications import vgg16
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import cv2
import os
import time
import datetime


def train(options):
	height = options['height']
	width = options['width']
	batch_size = options['batch_size']
	
	learning_rate = options['learning_rate']
	steps_per_epoch = options['steps_per_epoch']
	epochs = options['epochs']
	batch_size = options['batch_size']
	verbose_iter = options['verbose_iter']
	
	content_w = options['content_w'] 
	style_w1 = options['style_w1'] 
	style_w2 = options['style_w2'] 
	style_w3 = options['style_w3'] 
	style_w4 = options['style_w4'] 
	tv_w =  options['tv_w'] 
	output_w = options['output_w']
	
	style_img_path = options['style_img_path']
	test_content_img_path = options['test_content_img_path']
	model_saving_path = options['model_saving_path']
	train_dataset_path = options['train_dataset_path']
	 
	style_layers = options['style_layers']
	content_layer = options['content_layer']
	
	net_name = options['net_name']
	
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
	
	# Training loop
	datagen = ImageDataGenerator()
	dummy_input = expand_batch_input(batch_size, np.array([0.0]))
	start_time = time.time()
	summ_time = 0

	for ep in range(epochs):
		print('Epoch: ', ep)
		iters = 0

		for x in datagen.flow_from_directory(train_dataset_path, 
										  class_mode = None, 
										  batch_size = batch_size, 
										  target_size = (height, width)):
			t1 = time.time()
			x = vgg16.preprocess_input(x)
			content_act = content_function([x])[0]
			history = model.fit([x, content_act, style_act[0], style_act[1], 
								 style_act[2], style_act[3]], 
								[dummy_input, dummy_input, dummy_input, dummy_input,
								 dummy_input, dummy_input, x], 
								epochs = 1, verbose = 0, batch_size = batch_size) #verbose = 0, we will print info manually
			t2 = time.time()
			summ_time += t2 - t1
			iters += 1

			if iters % verbose_iter == 0:
				# predict and save image on current iteration
				verbose_result = model.predict([content_test, content_test_activation, style_act[0], style_act[1], 
								 style_act[2], style_act[3]])
				verbose_image = deprocess_img(verbose_result[6][0], width, height)
				cv2.imwrite(os.path.join(test_content_imgs_save_path, '{}_{}_{}_test_img.jpg'.format(net_name, iters, ep)), verbose_image)

				# print loop info
				print()
				print('Current iteration: ', iters)
				loss = history.history['loss']
				try:
					improvement = (prev_loss - loss) / prev_loss * 100
					prev_loss = loss
					print('Improvement: {}%'.format(improvement))
				except:
					prev_loss = loss
				epoch_est_time = (summ_time / verbose_iter) * (steps_per_epoch - iters)  # estimted time until epoch`s end
				complete_est_time = start_time + (summ_time / verbose_iter) * steps_per_epoch * epochs # estimated time of training complition
				print('Expected time before the end of the epoch: ', 
					str(datetime.timedelta(seconds = epoch_est_time)))
				print_test_info(verbose_result) 
				print()
				print_training_loss(history)
				summ_time = 0

			if iters > steps_per_epoch:
				break
	print('Training process is over. Saving weights...')			
	# Saving weights after training
	# Create autonencoder model without frozen layers from VGG-16
	pred_model = get_pred_model(height, width)
	
	# Fill pred_model by trained weights
	training_model_layers = {layer.name: layer for layer in model.layers}
	for layer in pred_model.layers:
		if layer.name in training_model_layers.keys():
			layer.set_weights(training_model_layers[layer.name].get_weights())

	pred_model.save_weights(os.path.join(model_saving_path, 'fst_{}_weights.h5'.format(net_name)))
	print('Weights was saved!')
	
		 
def predict(options, write_result=True):
	img_file = options['img_path'] # It could also store PIL image, not only string with file path 
	weights_path = options['weights_path']
	result_dir = options['result_dir']
	
	img = preprocess_img(img_file, img_height = None, img_width = None, resize_img = False) 
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
	
	if write_result:
		cv2.imwrite(os.path.join(result_dir, 'result.png'), result_img)
	else:
		return result_img
		
	
	
