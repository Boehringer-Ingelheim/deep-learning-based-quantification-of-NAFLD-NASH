import argparse
import yaml
import json
import sys
import os
from cnn_utils import cnn_utils
	
def main(args):
	# Open the configuration YAML file
	# given as command line argument
	with open(args.config, "r") as file:		
		# Get arguments from YAML		
		config_yaml = yaml.load(file)

		pre_trained_model_path = config_yaml["model"]["pre_trained_model_path"]
		pre_trained_model_file_name = config_yaml["model"]["pre_trained_model_file_name"]

		model_path = config_yaml["model"]["model_path"]
		model_file_name = config_yaml["model"]["model_file_name"]				

		ground_truth_path = config_yaml["tiles"]["ground_truth_path"]

		max_epochs = int(float(config_yaml["settings"]["max_epochs"]))
		generate_confusion_matrix = config_yaml["settings"]["generate_confusion_matrix"]

		do_val_split =  config_yaml["settings"]["do_val_split"]
		val_fraction =  config_yaml["settings"]["val_fraction"]
		
		# Print arguments
		print("\n-----------------------------\n")
		print("Pre-trained transfer model path: \t\t%s\n" % (pre_trained_model_path))
		print("Pre-trained transfer model file name: \t%s\n" % (pre_trained_model_file_name))
		print("Model path: \t\t%s\n" % (model_path))		
		print("Model file name: \t%s\n" % (model_file_name))
		print("Ground truth path: \t%s\n" % (ground_truth_path))	
		print("Max epochs:\t\t%d\n" % (max_epochs))
		print("Gen conf_matrix: \t%s\n" % (generate_confusion_matrix))
		print("do_val_split: \t\t%s\n" % (do_val_split))
		print("val_fraction:  \t\t%s\n" % (val_fraction))
		print("-----------------------------")
		
		# Create a cnn_utils object (handles tiled data and the CNN)					
		cnn_utils_obj = cnn_utils(model_path = model_path, model_file_name = model_file_name, tile_path = ground_truth_path, results_path = model_path)

		# Split off validation data (if desired by do_val_split keyword in config yaml)
		if (do_val_split):
			ret = cnn_utils_obj.split_validation_data(val_fraction = val_fraction)
			if (ret == False):
				# There is data in val which needs to be manually moved to train
				sys.exit()

		# Prepare image data generators
		cnn_utils_obj.prepare_image_data_generators()

		# Set class weights
		cnn_utils_obj.set_class_weights()					

		# Initialize the CNN
		print("\n")
		print("Initializing CNN")
		pretrained_model_full_path = pre_trained_model_path + pre_trained_model_file_name

		# Load pretrained model for transfer learning
		if pre_trained_model_file_name:
			print("Loading pre-trained CNN for transfer learning %s" % (pretrained_model_full_path))
			cnn_utils_obj.initialize_model(pretrained_model_full_path=pretrained_model_full_path)
		else:
			print("Initialize ImageNet-pretrained CNN for transfer learning")
			cnn_utils_obj.initialize_model(pretrained_model_full_path="")
		
		# Train
		print("Training CNN")
		cnn_utils_obj.train_model(n_epochs = max_epochs)
		cnn_utils_obj.save_learning_curves()

		# Load best model
		print("Loading best model for evaluation")
		cnn_utils_obj.initialize_model(os.path.join(model_path, model_file_name))
			
		# Generate confusion matrix
		if generate_confusion_matrix is True:
			# Save non-normalized confusion matrix
			cnn_utils_obj.generate_and_save_confusion_matrix(normalize = False)
			
			# Save normalized confusion matrix
			acc = cnn_utils_obj.generate_and_save_confusion_matrix(normalize = True)
			performance_metrics = {"acc": acc}

			# Print performance metrics
			# To do: Add more
			print("acc (val) %.2f" % (acc*100))

			# Save performance metrics to disk
			with open(model_path + model_file_name[:-3] + "_metrics.json", "w", encoding="utf-8") as fp:
				json.dump(performance_metrics, fp)

	
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('-c', '--config_file',action="store", dest="config",help="Filename of config file (*.yaml)", default="./train_CNN.yaml", required = True)
	args = parser.parse_args()
	main(args)