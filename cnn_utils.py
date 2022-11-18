import numpy as np
import os
import shutil
import warnings
import itertools
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import json
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Input
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization
from keras import backend as K
from keras import applications
from keras import optimizers
from keras.callbacks import EarlyStopping, TensorBoard, ReduceLROnPlateau, ModelCheckpoint
import skimage.measure
from sklearn.externals import joblib

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class cnn_utils:
		
	def __init__(self, model_path, model_file_name, tile_path, results_path, list_of_classes = []):
		""" Class to handle many aspects of tiled data for CNN
						
			Args:
				model_path: Path where the .h5 file with the CNN weights is located				
				model_file_name: File name where the .h5 file with the CNN weights is located (under model_path)
				tile_path: Base path where the tiles are located
				results_path: Path where results are written to
				list_of_classes: For classification: list of image classes. 
								 For training use default []: Will be automatically determined by subfolder names.
				
			Returns:	
				Bool: true on success, false otherwise
		""" 	
		# Base path (below this part should be subfolders for training and validation)
		self.model_path = model_path
		self.tile_path = tile_path
		
		# Model name
		self.model_file_name = model_file_name		

		# For training: Path names of training and validation images (relative to base path)
		self.train_path = "train/"
		self.val_path = "val/"		
				
		# Path where CNN model and confusion matrix will be saved (training only)
		self.model_path = model_path
		
		# Path where csv results will be saved (classification only)
		self.results_path = results_path
		
		# Image dimensions (299x299 for InceptionV3 based nets)
		self.img_width = 299
		self.img_height = 299
				
		# List of classes
		if (list_of_classes == []):
			self.list_of_classes = self.get_image_classes(self.tile_path + self.train_path)
		else:
			self.list_of_classes = list_of_classes
			
		# Use:
		# Tesla K40m 64
		# GTX1080 32
		self.batch_size = 32
		# Quadro
		#self.batch_size = 16
		# Todo: determine automatically based on detection of GPU and a fixed GPU -> batch size mapping
		
		# Device to run model on, e.g: "/gpu:0", or "/cpu:0"
		# See: https://stackoverflow.com/questions/40690598/can-keras-with-tensorflow-backend-be-forced-to-use-cpu-or-gpu-at-will		
		self.device_str = "/gpu:0"	
		
		# Class weight ratios to compensate class imbalance during training
		self.class_weight = {}
		
		# Image data generators
		self.train_generator = 0
		self.validation_generator = 0
		
		# The CNN model
		self.model = 0
		
		# The history
		self.history = 0

	def split_validation_data(self, val_fraction=0.1):
		""" Will take a random part of <fraction> from data in
			subfolders of <self.train_path> and move to a subfolder <self.val_path>

			Args:
				val_fraction: fraction of data to move from train to val

			Returns:
				Bool: true on success, false otherwise
		"""

		if (len(self.list_of_classes) == 0):
			print("self.list_of_classes is empty. Run init() first")

		# Make sure to move exiting validation data to train otherwise the split will not work
		# Delete old folders in val_path
		for image_class in self.list_of_classes:
			if os.path.exists(self.tile_path + self.val_path + image_class):
				if len(next(os.walk(self.tile_path + self.val_path + image_class))[2]) > 0:
					print("Please move data from val to train first and delete all subfolders below val.")
					return False
				else:
					shutil.rmtree(self.tile_path + self.val_path + image_class)

			if not os.path.exists(self.tile_path + self.val_path + image_class):
				os.makedirs(self.tile_path + self.val_path + image_class)

		# Move images
		#
		# Loop over all training classes
		for image_class in self.list_of_classes:

			# Loop over all images for the current image class
			image_name_list = next(os.walk(self.tile_path + self.train_path + image_class))[2]
			for image_name in image_name_list:

				# Move to val
				if np.random.rand() < val_fraction:
					file_name = self.tile_path + self.train_path + image_class + "/" + image_name
					file_name_new = self.tile_path + self.val_path + image_class + "/" + image_name

					shutil.move(file_name, file_name_new)
			# print(file_name, file_name_new)

		return True
										
	def get_image_classes(self, full_train_path):
		""" Determine labels of images classes from folders in <full_train_path>			
		Arguments:
			full_train_path (string): path containg training data in subfolders for each class
						
		Returns:
			List of folder names [string] found in full_train_path
			
		"""				
		image_classes_list = []
		
		if (os.path.isdir(full_train_path)):
			image_classes_list = next(os.walk(full_train_path))[1]
			image_classes_list = sorted(image_classes_list)
		return (image_classes_list)
		
	def prepare_image_data_generators(self):
		""" Sets class members 
			<self.train_generator> and <self.validation_generator> 
			with configured keras data generators for train and validation
		
		Args:
			None (TODO: Allow to modify augmentation settings)
				
		"""
		
		# Image augumentation configaduration for training
		train_datagen = ImageDataGenerator(
				rescale=1./255,
				rotation_range = 45,
				width_shift_range = 0.15,
				height_shift_range = 0.15,
				fill_mode = 'constant',
				cval = 0,
				horizontal_flip = True,
				vertical_flip = True)

		# Image augumentation configuration for validation
		# only rescaling
		validation_datagen = ImageDataGenerator(rescale=1./255)

		# this is a generator that will read pictures found in
		# subfolers of 'data/train', and indefinitely generate
		# batches of augmented image data
		train_generator = train_datagen.flow_from_directory(
				self.tile_path + self.train_path,  # this is the target directory
				target_size = (self.img_width, self.img_height),  # all images will be resized to img_width, img_height
				batch_size = self.batch_size,
				class_mode = 'categorical')

		# this is a similar generator, for validation data
		validation_generator = validation_datagen.flow_from_directory(
				self.tile_path + self.val_path,
				target_size = (self.img_width, self.img_height),
				batch_size = self.batch_size,
				class_mode = 'categorical')    
		
		self.train_generator = train_generator
		self.validation_generator = validation_generator
		
	def set_class_weights(self, verbose = True):    
		""" Determine class weight ratios in <train> to compensate class imbalance during training
			Function sets self.class_weight	
		
		Args:
			verbose (Bool): Print images in train, val in output
		"""    
		
		path_dict = {"train" : self.train_path, "val" : self.val_path}
		self.class_weight = {}

		for current_type in path_dict:
			if verbose:
				print(current_type)
			
			current_type_count = 0
			for image_class in self.list_of_classes:
				current_path = self.tile_path + path_dict[current_type] + image_class    
				#num_files_current_path = next(os.walk(current_path))[2]    				
				
				num_files_current_path = 0				
				for root, dirs, files in os.walk(current_path):
					for file in files:    
						if file.endswith('.png'):
							num_files_current_path += 1
				
				current_type_count = current_type_count + num_files_current_path
				
				if verbose:
					print("# class \'" + image_class + "\': " + str(num_files_current_path))
				
				if current_type == "train":
					self.class_weight[self.train_generator.class_indices[image_class]] = num_files_current_path
				
			if verbose:				
				print("----------------------------")
				print("Total " + current_type + ":", current_type_count, "\n")
			
		# Compute class weight to balance imbalanced training data
		total_count = 0
		for class_id in self.class_weight:
			total_count += self.class_weight[class_id]
			
		for class_id in self.class_weight:
			current_n = self.class_weight[class_id]
			self.class_weight[class_id] = total_count / current_n
			
		# The class weights multiplied by the number of samples should be equal for all classes
		if verbose:
			print("Class weights: ", self.class_weight)
			print("Class indices:", self.train_generator.class_indices)
			
	def initialize_model(self, pretrained_model_full_path = "", train_from_scratch = False):
		""" Sets <self.model> with an InceptionV3 based model, pretrained on ImageNet to be trained with num_classes
			
			Arguments:

				pretrained_model_full_path: (string)
										Path of a model to load
				train_from_scratch: (bool)
										Initialize without ImageNet weights if True (default: False)
											
		"""			
		# Clean up Keras
		K.clear_session()		
		
		input_shape = (self.img_width, self.img_height, 3)
	
		# Define base model
		if train_from_scratch:
			# Inception V3, without top layers
			print("Train from scatch *without* ImageNet weights")
			image_net_base_model = applications.InceptionV3(weights=None, include_top=False, input_shape=input_shape)
		else:
			# Inception V3, without top layers, trained on ImageNet
			image_net_base_model = applications.InceptionV3(weights="imagenet", include_top=False,
															input_shape=input_shape)

		# Define top model  
		input_tensor = Input(shape = input_shape)

		bn = BatchNormalization()(input_tensor)
		x = image_net_base_model(bn)
		x = GlobalAveragePooling2D()(x)
		x = Dropout(0.5)(x) # Default: 0.5
		output = Dense(len(self.list_of_classes), activation='softmax')(x)

		self.model = Model(input_tensor, output)						
					
		# Load weights of pre-trained model
		if pretrained_model_full_path:
			self.model.load_weights(pretrained_model_full_path)

		# Freeze early layers
		# https://www.tensorflow.org/tutorials/images/transfer_learning#un-freeze_the_top_layers_of_the_model
		# Let's take a look to see how many layers are in the base model
		# print("Number of layers in the base model: ", len(image_net_base_model.layers))

		# Fine-tune from this layer onwards
		# See also: https://github.com/keras-team/keras/issues/9460
		# fine_tune_at = 229

		# Freeze all the layers before the `fine_tune_at` layer
		# for layer in image_net_base_model.layers[:fine_tune_at]:
		# 	layer.trainable = False
		
		# Compile the model
		self.model.compile(loss = 'categorical_crossentropy', optimizer = optimizers.SGD(lr = 0.5e-4, momentum = 0.9), metrics = ['accuracy'])


	def train_model(self, n_epochs = 45):
		""" Trains <self.model>

			Will change <self.model> and <self.history>
			
			Arguments:
				n_epochs: Number of epochs to train
			
			TODO: callbacks as argument
		"""
		# Test if folder <self.model_path> is existent
		# The h5 file with the CNN weights and results are stored in this folder
		if (os.path.isdir(self.model_path) == False):			
			# Create sub-folder for model under self.model_path
			os.makedirs(self.model_path)

		print(os.path.join(self.model_path, "./logs"))
		tensorboard_callback = TensorBoard(log_dir=os.path.join(self.model_path, "./logs"))

		callbacks = [ModelCheckpoint(self.model_path + self.model_file_name, monitor='val_loss', verbose=1, save_best_only=True),
					 ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, cooldown=1, verbose=1, min_lr=1e-7),
					 tensorboard_callback]
										
		self.history = self.model.fit_generator(self.train_generator,
									  steps_per_epoch = self.train_generator.n // self.batch_size,
									  epochs = n_epochs,
									  validation_data = self.validation_generator,
									  validation_steps = self.validation_generator.n // self.batch_size,
									  verbose = 1,
									  class_weight = self.class_weight,
									  callbacks = callbacks)

	def save_learning_curves(self):
		""" Saves the learning curve in results_base_model_path			
		"""
		
		learning_curves = pd.DataFrame()
		learning_curves["acc"] = self.history.history["acc"]
		learning_curves["val_acc"] = self.history.history["val_acc"]
		learning_curves["loss"] = self.history.history["loss"]
		learning_curves["val_loss"] = self.history.history["val_loss"] 
		learning_curves.to_csv(self.results_path + self.model_file_name + "_learning_curve.csv", index=False)

	def generate_and_save_confusion_matrix(self, verbose = False, normalize = True):
		""" Generate a confusion matrix and save fig and npy do disk		
		
			Arguments:
				verbose: (Bool) print output or not
				normalize: (Bool) True: Normalize confusion matrix to prediction probabilities (otherwise numbers)

			Return:
				accuracy_score
		"""
		
		# Move data from ./val to ./val_unsorted.
		unsorted_validation_data_path = "val_unsorted/"
		sub_path = "test/"
		
		if verbose is True:
			print("\nCreating confusion matrix")
		
		if (os.path.isdir(self.tile_path + unsorted_validation_data_path + sub_path)):
			#print("Test folder exists. Please delete first folder + content and run function again!\n%s" % base_model_path + unsorted_validation_data_path + "test/")
			shutil.rmtree(self.tile_path + unsorted_validation_data_path + sub_path)
			
		# Create test folder (will contain identical data to val)
		os.makedirs(self.tile_path + unsorted_validation_data_path + "test")    
		
		# Copy files from val to unsorted_validation_data_path
		for image_class in self.list_of_classes:
			filenames = next(os.walk(self.tile_path + self.val_path + image_class))[2]
			
			for file in filenames:
				src = self.tile_path + self.val_path + image_class + "/" + file
				dst = self.tile_path + unsorted_validation_data_path + sub_path + file
				
				shutil.copyfile(src, dst)
		
		test_datagen = ImageDataGenerator(rescale=1./255)

		# Predict generator
		test_generator = test_datagen.flow_from_directory(
				self.tile_path + unsorted_validation_data_path,
				target_size=(self.img_width, self.img_height),
				batch_size = 1,
				class_mode = None,
				shuffle = False)

		# Make the prediction		
		y_predict_val = self.model.predict_generator(test_generator, test_generator.n, verbose=1)
			
		# Create data frame with a list of the validation data
		validation_list = pd.DataFrame(columns = self.list_of_classes)

		for image_class in self.list_of_classes:
			filenames = next(os.walk(self.tile_path + self.val_path + image_class))[2]
			
			for file in filenames:
				if file[-4:] == ".png":        
					validation_list = validation_list.append({"filename" : file[:-4]}, ignore_index=True)    
					
					p_i = {}
					for ic in self.list_of_classes:
						p_i[ic] = 0
						if ic == image_class:
							p_i[ic] = 1                                
												
						validation_list.at[validation_list["filename"] == file[:-4], ic] = p_i[ic]
		
		# Now create y_ground_truth_val with the same order as in y_predict_val
		y_ground_truth = np.zeros(y_predict_val.shape)
		np.set_printoptions(precision=2)		

		row = 0
		for name_str in test_generator.filenames:
			filename = name_str[len(sub_path):-4]
				
			i = 0
			for ic in self.list_of_classes:
				ground_truth_val = validation_list[validation_list["filename"] == filename][ic]				
				y_ground_truth[row,i] = np.float(np.float(ground_truth_val.values[0]))
				i = i + 1    
			
			# Binarize prediction
			i = 0
			for ic in self.list_of_classes:
				y_predict_val[row,i] = y_predict_val[row,i] == max(y_predict_val[row,:])    
				i = i + 1
				
			if verbose is True:
				if (np.argmax(y_predict_val[row, :]) != np.argmax(y_ground_truth[row, :])):
					print(filename)
					print("pred = ", y_predict_val[row, :])					
					print("truth = ", y_ground_truth[row, :])				
				
			row = row + 1
				
		# Create y_true and y_pred based on y_predict_val and y_ground_truth
		# in order to match requred shape for sklearn.metrics.confusion_matrix

		y_true = np.zeros(y_ground_truth.shape[0])
		y_pred = np.zeros(y_predict_val.shape[0])

		for row in range(0,y_ground_truth.shape[0]):
			
			i = 0
			for ic in self.list_of_classes:
				if (y_ground_truth[row, i] == 1):
					true_class = i 
				i = i + 1
				
			y_true[row] = true_class
				
			i = 0
			for ic in self.list_of_classes:
				if (y_predict_val[row, i] == 1):
					pred_class = i 
				i = i + 1
				
			y_pred[row] = pred_class    

		#confusion_matrix(y_true, y_pred, labels = image_classes_list)
		cm = metrics.confusion_matrix(y_true, y_pred)
		
		if normalize:
			cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
			
		fig = plt.figure(figsize=(7, 4), dpi=100)
		if len(self.list_of_classes) <= 6:
			matplotlib.rcParams.update({'font.size': 14})
		else:
			matplotlib.rcParams.update({'font.size': 10})

		plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
		tick_marks = np.arange(len(self.list_of_classes))
		plt.xticks(tick_marks, sorted(self.list_of_classes), rotation=0)
		plt.yticks(tick_marks, sorted(self.list_of_classes))

		fmt = '.2f' if normalize else 'd'
		thresh = cm.max() / 2.
		for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
			plt.text(j, i, format(cm[i, j], fmt),
				horizontalalignment="center",
				color="white" if cm[i, j] > thresh else "black")

		plt.tight_layout()
		matplotlib.rcParams.update({'font.size': 14})
		plt.ylabel('True label')
		plt.xlabel('Predicted label')
		plt.gcf().subplots_adjust(bottom=0.15)

		normalize_str = ""
		if (normalize):			
			normalize_str = "_normalized_by_row"		
			
		plt.savefig(self.results_path + self.model_file_name + normalize_str + '_confusion_matrix.png', dpi=300)
		np.save(self.results_path + self.model_file_name + normalize_str + '_confusion_matrix.npy', cm)

		# Compute metrics as a by-product
		acc = metrics.accuracy_score(y_pred, y_true)

		return acc
		
	def show_batch(self, batch_generator):
		""" Show one image batch of a generator object
			Can be useful to test augmentation settings
		Args:
			generator: DirectoryIterator from ImageDataGenerator object (e.g. result of flow from directory)

		Returns:
			Nothing
		"""    
		x_batch, y_batch = next(batch_generator)
		for i in range (0,self.batch_size):
			image = x_batch[i]
			plt.imshow(image)
			plt.show()
			
	def classify_tiles(self):
		""" Will classify tiles found in self.tiles_path
			(must contain one subfolder, e.g. "tiles")
			
		Arguments:
			tiles_path: (string) Location of the images (relative to self.tile_path). Must contain one subfolder containing images.
		
		Returns:
			Pandas Dataframe with columns: image filename and for each image class the prediction probability-like quantity
		"""
	
		test_datagen = ImageDataGenerator(rescale=1./255)
		
		# Predict generator
		test_generator = test_datagen.flow_from_directory(self.tile_path, target_size=(self.img_width, self.img_height), batch_size = 1, class_mode = None, shuffle = False)
		
		# Debug
		# print(test_generator.n, len(test_generator.filenames))
			
		# Make the prediction
		y_predict = self.model.predict_generator(test_generator, test_generator.n, verbose=1)
			
		# Add classification results to a dataframe
		df = pd.DataFrame(y_predict, columns = self.list_of_classes)
		df["filenames"] = test_generator.filenames
		
		return df
		
	def process_results(self, classification_result):
		""" Will process raw results from classify_tiles 
			by 
			(1) Adding rows with re-normalized probabilities to a sum of 1 _without_ the ignore class (if ignore class is largest probability, values are set to np.nan)
			(2) Adding a row with the weighted sum of all re-normalized probabilities _without_ the ignore class. This row will be called <weighted_class>
			(3) Add an uncertainty column (1-argmax(p_0, p_1, p_2, ...) without ignore)			
		
		Arguments:
			classification_result: (Pandas DataFrame) Result table as provided by classify tiles
			
		Return:
			Pandas DataFrame of prepared results with the new rows
			
		"""
		
		# Add new columns
		classification_result["x"] = np.nan
		classification_result["y"] = np.nan
		classification_result["slide_id"] = ""
		for image_class in self.list_of_classes:
			if (image_class != "ignore"):
				classification_result[image_class + "_norm"] = np.nan
				classification_result[image_class + "_norm_bin"] = np.nan
		classification_result["weighted_class"] = np.nan
		classification_result["uncertainty"] = np.nan
		
		# Extract slide_id x, y
		for index, row in classification_result.iterrows():
			# Remove ".png" file ending
			filename = row["filenames"][:-4]
			
			# Split filename by "_"
			# Typical filename "EXPID_X_Y" e.g. "17_231_201_12_24"
			# Sometimes there can be names such as:
			# (1) "18_211_Masson_205_100_11" , ("18_211_Masson_205" is slide_id)
			filename_short = filename.split("/")
			filename_parts = filename_short[1].split("_")
			num_items = len(filename_parts)
						
			# Last part: y
			y = int(filename_parts[num_items-1])
			# Second last part: x
			x = int(filename_parts[num_items-2])
			
			# Third last part: slide_id (the rest of the name)
			slide_id = ""
			for i in range(0, num_items-2):
				if i == 0:
					slide_id = filename_parts[i]
				else:
					slide_id = slide_id + "_" + filename_parts[i]

			classification_result.at[index, "x"] = x
			classification_result.at[index, "y"] = y
			classification_result.at[index, "slide_id"] = slide_id
			
		# Set data types of x and y to int
		classification_result["x"] = classification_result["x"].astype(int)
		classification_result["y"] = classification_result["y"].astype(int)
			
		# Renormalize tiles
		
		# Preparation
		# (1) Create the new columns for renormalized tiles with suffix "_norm"
		# (2) Create a column "weighted_class"
		# (3) Create a column "uncertainty"
		for image_class in self.list_of_classes:
			if (image_class != "ignore"):
				image_class_renormalized = image_class + "_norm"
				classification_result[image_class_renormalized] = np.nan				
				
		classification_result["weighted_class"] = np.nan
		classification_result["uncertainty"] = np.nan
					
		# (1) Renormalization
		# (1.1) Divide through sum of values without ignore
		# (1.2) In case ignore is the largest value for a tile set other values to np.nan
		# (2) Add weighted class
		# (3) Add uncertainty		 
		for index, row in classification_result.iterrows():
			# Get denominator 
			# sum of all non ignore classes
			denominator = 0
			for image_class in self.list_of_classes:
				if (image_class != "ignore"):
					denominator = denominator + row[image_class]
					
			# Find maximum probability class
			max_row = np.max(row[self.list_of_classes])    
			weighted_class = 0
			if (row["ignore"] < max_row):
				# Ignore class has not highest probability
				# (1) renormalize all class values (skip ignore)
				# (2) compute weighted class sum (assumes image class as ordered integers, 0, 1, 2, ...)
				# (3) compute binarized class values of renormalized values
				for image_class in self.list_of_classes:                
					if (image_class != "ignore"):
						# (1) renormalize all class values (skip ignore)
						classification_result.at[index, image_class + "_norm"] = row[image_class] / denominator

						# (2) compute weighted class sum (assumes image class as ordered integers, 0, 1, 2, ...)
						weighted_class = weighted_class + float(image_class)*row[image_class] / denominator

						# (3) compute binarized class values of renormalized values
						classification_result.at[index, image_class + "_norm_bin"] = np.round(row[image_class] / denominator)

				classification_result.at[index, "weighted_class"] = weighted_class
				classification_result.at[index, "uncertainty"] = 1 - max_row			
				
		return classification_result
		
	def get_dl_score(self, thresholds, value):
		""" Get deep learning score for a value based on a series of thresholds
		
		Arguments:
			thresholds (dict of floats): 
						Thresholds, keys are scores with the score for the interval at the higher interval end. 
						Assumes ordering such that thresholds[i] < thresholds[i+1]
			value (float): 
						Deep learning readout, e.g. log_1_mean_norm
			
		Return:
			score (int): 
						Score						
				
		"""

		# Return score
		ret_score = 0
		
		# Create list of possible scores (keys of thresholds)
		scores = list()    
		for i in thresholds.keys():
			scores.append(int(i))       
		
		# Find interval of value    
		for i in range(0, len(scores),1):        
			if (value > thresholds[i]):
				ret_score = i            
				
		return ret_score
		
	def generate_summary_results(self, classification_result, score_name = "", thresholds_json = ""):
			""" Will compute summary of final result per slide_id
				and map to pathologist score 
						
			Arguments:
				classification_result: (Pandas DataFrame) Result table as provided by classify tiles after processing by process_results	
				score_name: (str) Name of discrete score column (pathologist-like deep learning score)
				thresholds_json: filename of thresholds json for mapping to pathologist scores. Default = "" (will skip mapping step if "")
			
			Returns: 
				DataFrame with summary results
			"""
						
			column_names = ["slide_id", "n_tiles"]
			classification_result = classification_result.astype({"slide_id": str})
			summary_result = pd.DataFrame(columns = column_names)
						
			# Load existing thresholds from json			
			thresholds = {}
			if (len(thresholds_json) > 0):		
				if (os.path.isfile(self.model_path + thresholds_json)):
					with open(self.model_path + thresholds_json, "r") as read_file:
						thresholds = json.load(read_file)
						
						# Convert type of dict
						thresholds = {int(k):float(v) for k,v in thresholds.items()}			
				else:
					print("File %s not found." % (self.model_path + thresholds_json))
				

			# Loop over all unique slide_ids
			for slide_id in classification_result["slide_id"].unique():
				# Store subset data frame for current slide_id
				classification_result_current = classification_result[classification_result["slide_id"] == slide_id]
						
				# Dataframe row to hold current results
				current_result = pd.DataFrame([[slide_id]], columns = ["slide_id"])

				# Get number of tiles
				n_tiles = classification_result_current.shape[0]
				current_result["n_tiles"] = n_tiles

				# Get mean of uncertainty
				current_col = classification_result_current["uncertainty"]
				#current_col = current_col.str.replace(',', '.').astype({"uncertainty" : float})
				# https://stackoverflow.com/questions/29688168/mean-nanmean-and-warning-mean-of-empty-slice
				# I expect to see RuntimeWarnings in this block
				current_result["average_uncertainty"] = np.nan
				with warnings.catch_warnings():
					warnings.simplefilter("ignore", category=RuntimeWarning)
					current_result["average_uncertainty"] = np.nanmean(current_col)

				# Get mean of weighted class
				current_col = classification_result_current["weighted_class"]
				#current_col = current_col.str.replace(',', '.').astype({"weighted_class" : float})
				# https://stackoverflow.com/questions/29688168/mean-nanmean-and-warning-mean-of-empty-slice
				# I expect to see RuntimeWarnings in this block
				current_result[score_name + "_average_weighted_class"] = np.nan
				average_weighted_class = np.nan
				with warnings.catch_warnings():
					warnings.simplefilter("ignore", category=RuntimeWarning)
					average_weighted_class = np.nanmean(current_col)
					current_result[score_name + "_average_weighted_class"] = average_weighted_class

				# Special case ballooning, density of binarized ballooning detections (n_balloning_cells / n_all_tiles)
				if (score_name == "Ballooning_score"):
					current_col = classification_result_current["1_norm_bin"]

					# https://stackoverflow.com/questions/29688168/mean-nanmean-and-warning-mean-of-empty-slice
					# I expect to see RuntimeWarnings in this block
					current_result["Ballooning_density"] = np.nan
					ballooning_density = np.nan
					with warnings.catch_warnings():
						warnings.simplefilter("ignore", category=RuntimeWarning)
						ballooning_density = np.nanmean(current_col)
						current_result["Ballooning_density"] = ballooning_density

				# Add pathologist score
				if (len(thresholds) and (not np.isnan(average_weighted_class))):
					current_result[score_name] = self.get_dl_score(thresholds, average_weighted_class)
				else:
					current_result[score_name] = np.nan

				# Add to summary dataframe
				summary_result = summary_result.append(current_result, ignore_index = True)
						
			# Reorder columns			
			column_names.append("average_uncertainty")			
			column_names.append(score_name + "_average_weighted_class")				
			column_names.append(score_name)
			if (score_name == "Ballooning_score"):
				column_names.append("Ballooning_density")
			summary_result = summary_result[column_names]
			
			# Set data type of discrete pathologist score to float with 0 digits
			# Workaround to allow NaNs
			# https://stackoverflow.com/questions/41550746/error-using-astype-when-nan-exists-in-a-dataframe
			summary_result[score_name] = summary_result[score_name].round(0)
			summary_result[score_name] = summary_result[score_name].astype(float)
			summary_result = summary_result.astype({"slide_id": str})
			
			return summary_result
	
		
	def presort_tiles(self, eps, tiles_path, target_path, classification_result, class_of_interest = [], verbose = False):
		""" Will move classified images from tiles_path/tiles to subfolders with class labels in target_path
		
		Arguments:
			eps: (float 0<eps<1). A number descripting the maximal deviation from probability 1 for a class, e.g. 0.1
			tiles_path: (string), path below base_model_path (without sub_score_path) to move data from
			target_path: (string), path below base_model_path (with sub_score_path) to move data from
			classification_result: (Pandas Dataframe) with columns: image filename and for each image class the prediction probability
			class_of_interest: List of labels to move, leave empty to move all
			verbose: (bool) Print files to move
			
		Returns:
			Nothing							
		
		"""		
		
		# Autogenerate target subfolders
		sub_sub_path_list = ["/low_conf", "/mid_conf", "/high_conf"]
		
		for str_image_class in self.list_of_classes:  
			for sub_sub_path in sub_sub_path_list:
				str_dir = self.base_model_path + target_path + str_image_class + sub_sub_path			
				if (os.path.isdir(str_dir)):
					
					if (verbose is True):
						print("Removing %s (with content)" % (str_dir))
						
					shutil.rmtree(str_dir)

				if (verbose is True):
					print ("Removing %s" % (str_dir))
				os.makedirs(str_dir)    					
		
		# Iterate through dataframe to move tiles
		for index, row in classification_result.iterrows():
			# Is CNN certain with the current image?
			if (abs(max(row[:-1]) - 1) < eps):
				file_name = row["filenames"]
				file_name_base = file_name[len("tiles/"):]                
				
				for i in range(0,len(self.list_of_classes)):            
					if (max(row[:-1]) == row[i]):
						sub_path = self.list_of_classes[i] + "/"
						break;
							
				# Move only a certain class of interest
				if (str(i) in class_of_interest or class_of_interest == []):
					# Low confidence
					if ((abs(max(row[:-1]) - 1)) < eps and (abs(max(row[:-1]) - 1) >= eps/2)):
						sub_sub_path = "low_conf"
					# Medium confidence
					if ((abs(max(row[:-1]) - 1)) < eps/2 and (abs(max(row[:-1]) - 1) >= eps/10)):
						sub_sub_path = "mid_conf"
					# High confidence
					if (abs(max(row[:-1]) - 1) < eps/10):
						sub_sub_path = "high_conf"
					
					source_str = self.base_model_path + tiles_path + row["filenames"]
					target_str = self.base_model_path + target_path + sub_path + sub_sub_path   
					if (verbose is True):
						print("Copying %s to %s" % (row["filenames"], sub_path + sub_sub_path))        
					
					shutil.copy(source_str, target_str)
					
	def get_scoring_ANN_feature_matrix(self, classification_result, scaler_file):
		""" Will take a full results dataframe (processed by renormalize_and_process_CNN_results)
			and compute ANN features

			Input:
				classification_result (dataframe)
				Optional: scaler_file: Full path to min max scaler. If empty result will not be scaled

			Result:
				X, X_scaled (normal and scaled feature matrix)
				slide_id: Dataframe with list of slide_id in exact order as X, X_scaled
		"""

		epsilon = 0.0000001

		# Define input features to be used from classification_result
		features = []
		# All columns ending with "_norm"
		for column in classification_result.columns:
			if column.endswith("_norm"):
				features.append(column)

		# Define output features (in X and X_scaled)
		features_X = []
		features_X.append("average_weighted_class")
		for feature in features:
			features_X.append(feature+ "_Shannon_entropy")
			features_X.append(feature + "_mean")
		features_X.append("log_average_weighted_class")

		# Create empty dataframes
		X = pd.DataFrame(columns= features_X)
		X_scaled = pd.DataFrame(columns=features_X)

		# Working list to build dataframe in loop as a list of dicts
		X_list = []
		slide_id_list = []

		# Iterate over classification_result
		for slide_id in classification_result["slide_id"].unique():
			slide_id_list.append({"slide_id": slide_id})

			classification_result_subset = classification_result[classification_result["slide_id"] == slide_id]

			x_min = int(np.min(classification_result_subset["x"]))
			x_max = int(np.max(classification_result_subset["x"]))
			y_min = int(np.min(classification_result_subset["y"]))
			y_max = int(np.max(classification_result_subset["y"]))

			features_X_dict = {}
			for feature in features:
				# Create temporary 2d feature image to compute entropy with skimage
				feature_img = np.empty(shape=(int(x_max - x_min + 1), int(y_max - y_min + 1)))
				feature_img[:] = epsilon

				for index_full_results, row_full_results in classification_result_subset.iterrows():
					x = int(row_full_results["x"] - x_min)
					y = int(row_full_results["y"] - y_min)

					feature_img[x, y] = row_full_results[feature]

					if feature_img[x, y] == 0:
						feature_img[x, y] = epsilon

					if np.isnan(feature_img[x, y]):
						feature_img[x, y] = epsilon

				# Entropy of p1, p2, ...
				shannon_entropy = skimage.measure.shannon_entropy(feature_img)
				features_X_dict[feature + "_Shannon_entropy"] = shannon_entropy

				# Average feature of (p0, p1, ...
				feature_mean = np.mean(classification_result_subset[feature])
				features_X_dict[feature + "_mean"] = feature_mean

			# Average of weighted class
			average_weighted_class = np.mean(classification_result_subset["weighted_class"])
			features_X_dict["average_weighted_class"] = average_weighted_class

			# Log of average weighted class
			log_average_weighted_class = np.log(average_weighted_class)
			features_X_dict["log_average_weighted_class"] = log_average_weighted_class

			X_list.append(features_X_dict)

		X = pd.DataFrame(X_list)

		# Reorder columns
		X = X[features_X]

		# Load scaler
		if (scaler_file):
			scaler = joblib.load(scaler_file)

			# transform X with previously saved min-max scaler
			X_scaled = scaler.transform(X)
		else:
			X_scaled = []

		slide_id_df = pd.DataFrame(slide_id_list)

		return X, X_scaled, slide_id_df

	def create_ANN(self, input_dim, max_score_):
		""" Create the scoring ANN

		input:
			input_dim: number of dimensions for inpit
			max_score_: restrict output to 0...max_score

		output:
			model: ANN model (Keras)
		"""

		# Activation function with predefined range of score
		global max_score
		max_score = max_score_

		def mapping_to_target_range(x, target_min=0):
			global max_score
			target_max = max_score

			z = K.tanh(x) + 1
			scale = (target_max - target_min) / 2.
			return z * scale + target_min

		# define MLP network
		model = Sequential()
		model.add(Dense(100, input_dim = input_dim, activation="elu"))
		model.add(Dropout(0.8))
		model.add(Dense(100, activation="relu"))
		model.add(Dense(1, activation = mapping_to_target_range))

		# model
		return model

	def get_ANN_scores(self, X_scaled, ANN_model_weights_file_name, score_str):
		""" Will load the scoring ANN with weights stored under ANN_model_weights_file_name
			and apply this to X_scaled

			Input:
				X_scaled: Feature matrix
				ANN_model_weights_file_name

			Output:
				Y_pred: Scores
		"""
		max_score = {"Ballooning" : 2, "Inflammation" : 3, "Steatosis": 3, "Fibrosis" : 4}

		scoring_ANN = self.create_ANN(X_scaled.shape[1], max_score[score_str])
		scoring_ANN.load_weights(ANN_model_weights_file_name)

		Y_pred = scoring_ANN.predict(X_scaled)

		return Y_pred
					