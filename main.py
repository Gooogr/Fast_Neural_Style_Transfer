import argparse
import json
from model_functions import train, predict

### TEMP FUNCTIONS ###

options = None

def train(options):
	print('Train!')
	
def predict():
	print('Predict!')
	
### TEMP FUNCTIONS END ###
	
def readDict(filename, sep):
	'''
	https://stackoverflow.com/questions/11026959/writing-a-dict-to-txt-file-and-reading-it-back
	'''
	 with open(filename, "r") as f:
        dict = {}
        for line in f:
            values = line.split(sep)
            dict[values[0]] = {int(x) for x in values[1:len(values)]}
        return(dict)
        
def create_main_parser():
	parser = argparse.ArgumentParser(description = 'Setting up script mode')
	parser.add_argument('mode', action = 'store', choices=['train', 'predict'], 
						help = 'What do you want? Train or predict?')
	return parser
	
def create_predict_parser():
	parser = argparse.ArgumentParser(description = 'Setting up file pathes')
	parser.add_argument('-w', action = 'store', dest = 'image_path',
						type = str, required = False, help = 'Path to the content image')
	parser.add_argument('-i', action = 'store', dest = 'image_path',
						type = str, required = True, help = 'Path to the content image')
	parser.add_argument('-d', action = 'store', dest = 'result_image_dir',
						type = str, required = True, help = 'Directory for saving the result image')
	
def create_train_parser():
	parser = argparse.ArgumentParser(description = 'Setting up file pathes')
	parser.add_argument('-c', action = 'store', dest = 'config_path', 
						type = str, required = False, help = 'Path to the configuration json file')
	
	return parser
	
if __name__ == '__main__':
	main_parser = create_mian_parser()
	main_args = main_parser.parse_args()
	if main_args.mode == 'train':
		parser = create_train_parser()
		args = parser.parse_args()
		with open(args.config_path) as f:
			options_dict = json.load(f)
			f.close()
		train(options_dict)
	else:
		parser = create_predict_parser()
		args = parser.parse_args()
		
		# Put args to functions
		predict()
		
