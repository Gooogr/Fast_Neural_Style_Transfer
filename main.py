import argparse
import json
from model_functions import train, predict

# Idea source: https://docs.python.org/2/library/argparse.html#sub-commands



# sub-command functions
def start_train(args):
	with open(args.config_path) as f:
		options = json.load(f)
		f.close()
	# ~ print(options_dict)
	train(options)
	
def start_predict(args):
	# ~ print(args)
	options = {'img_path':args.image_path,
				'weights_path':args.weights_path,
				'result_dir':args.result_image_dir}
	predict(options)
	
# create the top-level parser
parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(help = 'Choose mode: train or predict')
subparsers.required = True

#create sub-parser for train function
parser_train = subparsers.add_parser('train')
parser_train.add_argument('--conf', action = 'store', type = str, 
						  dest = 'config_path', default = 'config.json',
						  help = 'Path to the config.json file')
parser_train.set_defaults(func = start_train)

#create sub-parser for predict function
parser_predict = subparsers.add_parser('predict')
parser_predict.add_argument('-w', action = 'store', type = str,
							dest = 'weights_path', 
							help = 'Path to pre-trained weights',
							required = True)
parser_predict.add_argument('-i', action = 'store', type = str, 
							dest = 'image_path',
							help = 'Path to the content image',
							required = True)
parser_predict.add_argument('-r', action = 'store', type = str,
							dest = 'result_image_dir', default = '',
							help = 'Directory for saving the result image')
parser_predict.set_defaults(func = start_predict)
	
if __name__ == '__main__':
	args = parser.parse_args()
	args.func(args)	
