import sys
import os
import time as time
import argparse
import yaml
import pandas as pd
from cnn_utils import cnn_utils
	
def main(args):
	# Open the configuration YAML file
	# given as command line argument
	with open(args.config, "r") as file:
		# Measure start time
		start = time.time()
	
		# Get arguments from YAML		
		config_yaml = yaml.load(file)		

		# Scores
		# score_list = ["Ballooning", "Inflammation", "Steatosis", "Fibrosis"]
		score_list = ["Ballooning", "Inflammation", "Steatosis", "Fibrosis"]

		# List of classes
		list_of_classes = {"Ballooning" : ["0", "1", "ignore"], "Inflammation" : ["0", "1", "2", "ignore"], "Steatosis" : ["05", "10", "15", "20", "25", "30", "35", "40", "45", "50", "55", "60", "65", "70", "75", "ignore"] , "Fibrosis" : ["0", "1", "2", "3", "4", "ignore"]}
		
		# Score name (for output tables, do not change)
		score_names = {"Ballooning" : "Ballooning_sub_score", "Inflammation" : "Inflammation_sub_score", "Steatosis" : "Steatosis_sub_score", "Fibrosis" : "Fibrosis_score"}
		
		# Models
		models = {"Ballooning" : config_yaml["models"]["ballooning_model"], 
		          "Inflammation" : config_yaml["models"]["inflammation_model"],
				  "Steatosis" : config_yaml["models"]["steatosis_model"],
				  "Fibrosis" :  config_yaml["models"]["fibrosis_model"]}
		
		# ANN model weights
		ANN_model_weights = {"Ballooning" : config_yaml["Scoring_ANN"]["balloning_ANN"],
		                    "Inflammation" : config_yaml["Scoring_ANN"]["inflammation_ANN"],
							"Steatosis" : config_yaml["Scoring_ANN"]["steatosis_ANN"],
							"Fibrosis" : config_yaml["Scoring_ANN"]["fibrosis_ANN"]}

		# ANN pre-processing files (min-max scaler for X)
		ANN_scaler = {"Ballooning" : config_yaml["Scoring_ANN"]["balloning_scaler"],
					  "Inflammation" : config_yaml["Scoring_ANN"]["inflammation_scaler"],
					  "Steatosis" : config_yaml["Scoring_ANN"]["steatosis_scaler"],
					  "Fibrosis" : config_yaml["Scoring_ANN"]["fibrosis_scaler"]}
		
		# Tiles
		tiles = {"Ballooning" : config_yaml["tiles"]["NAS_tile_path"], 
                 "Inflammation" : config_yaml["tiles"]["NAS_tile_path"],
                 "Steatosis" : config_yaml["tiles"]["NAS_tile_path"],
                 "Fibrosis" :  config_yaml["tiles"]["fibrosis_tile_path"]}				  
		
		# Results		
		results_path = config_yaml["results"]["results_path"]
		experiment_name = config_yaml["results"]["experiment_name"]				
		
		# Print arguments		
		print("\n-----------------------------\n")				
		print("scores: \t\t%s\n" % (score_list))
		
		for score, model_file_str in models.items():
			print("%s model: \t%s\n" % (score, os.path.basename(model_file_str)))
			
		for score, tile_path in tiles.items():
			print("%s tiles: \t%s\n" % (score, tile_path))
		
		for score, ANN_file_str in ANN_model_weights.items():
			print("%s ANN_files: \t%s\n" % (score, os.path.basename(ANN_file_str)))

		for score, ANN_scaler_str in ANN_scaler.items():
			print("%s ANN_scaler_files: \t%s\n" % (score, os.path.basename(ANN_scaler_str)))
			
		print("results_path: \t%s\n" % (results_path))
		print("experiment_name:  \t\t%s\n" % (experiment_name))
		print("-----------------------------")

		# Todo: Function to do initial check if all files are available
		# This can prevent crashes after hours of computation
		
		summary_result = pd.DataFrame()

		Y_pred = {}

		for score in score_list:
			print("\nCurrent score: %s" % (score))
		
			# Create a cnn_utils object (handles tiled data and the CNN)					
			cnn_utils_obj = cnn_utils(model_path = "", model_file_name = models[score], tile_path = tiles[score], results_path = results_path, list_of_classes = list_of_classes[score])

			# Initialize the CNN
			print("\n")
			print("Initializing CNN...")
			cnn_utils_obj.initialize_model(pretrained_model_full_path = models[score])
			print("Model loaded.\n")
			
			# Classify tiles
			classification_result = cnn_utils_obj.classify_tiles()
			
			# Renormalize classification result and add sample_id column
			classification_result = cnn_utils_obj.process_results(classification_result)
			
			# Save detailed results
			file_name_detailed_results = cnn_utils_obj.results_path + experiment_name + "_" + score_names[score] + ".csv"
			classification_result.to_csv(file_name_detailed_results, index = False, sep = ";", decimal=".", float_format='%.2f')
			print("Details saved to: %s" % (file_name_detailed_results))

			# Get scaled feature matrix X for scoring ANN
			X, X_scaled, slide_id_df = cnn_utils_obj.get_scoring_ANN_feature_matrix(classification_result, ANN_scaler[score])
			X["slide_id"] = slide_id_df.values
			X.to_csv(results_path + experiment_name + "+" + score + "_X.csv", index = False, sep = ";", decimal=",", float_format='%.4f')

			# Load scoring ANN with trained weights
			# and apply scoring ANN to get scores
			Y = cnn_utils_obj.get_ANN_scores(X_scaled, ANN_model_weights[score], score)

			if summary_result.empty:
				current_data_dict = {"slide_id" : slide_id_df["slide_id"], score + "_AI_score" : Y.flatten()}
				summary_result = pd.DataFrame(current_data_dict, columns = ["slide_id", score + "_AI_score"])
			else:
				current_data_dict = {"slide_id": slide_id_df["slide_id"], score + "_AI_score": Y.flatten()}
				current_result = pd.DataFrame(current_data_dict, columns=["slide_id", score + "_AI_score"])
				summary_result = pd.merge(summary_result, current_result, how = "left", left_on=["slide_id"], right_on=["slide_id"])

#		summary_result["NAS_Score_AI"] = summary_result["Ballooning_AI_score"] + summary_result["Inflammation_AI_score"] + summary_result["Steatosis_AI_score"]
			
		# Save summary			
		file_name_summary_results = results_path + experiment_name + "_Kleiner_score_summary.csv"
		summary_result.to_csv(file_name_summary_results, index = False, sep = ";", decimal=",", float_format='%.3f')
		print("Summary saved to: %s" % (file_name_summary_results))		
		
		# Print elapsed time
		end = time.time()
		print("Time elapsed: %.1f s" % (end - start))
		
	
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('-c', '--config_file',action="store", dest="config",help="Filename of config file (*.yaml)", required = True)	
	args = parser.parse_args()
	main(args)
