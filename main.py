# for visualizing model predicitions
import matplotlib.pyplot as plot

# from tensorflow.keras.applications import EfficientNetB0
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import os

# Global constants:
size = 224;
shape = size, size

p = os.path.join
cdir = "/home/mitch/Documents/search/laptops/"
# datasets here:
ddir = p(cdir, "componentsds/aug/")
saves = p(cdir, "saved")
# make a folder for saved tensorflow models if it doesn't exist already
if not os.path.exists(saves):
  os.mkdir(saves)

class Prototype:
	"""Deep-learning model to identify laptop hardware components."""
	def __init__(self, model: str, visualize=40) -> None:
		self.model = model
		
		# number of samples to show in visualization
		self.samples = visualize
		self.s = int(self.samples ** .5)
		
		self.load_datasets()
		self.load_model()
		# self.visualize_output()

	def load_datasets(self) -> None:
		# 80:20 split 
		kw = dict(rescale=1/255., validation_split=.2)

		# TODO: keras.preprocessing is deprecated
		test_gen = tf.keras.preprocessing.image.ImageDataGenerator(**kw)
		train_gen = tf.keras.preprocessing.image.ImageDataGenerator(**kw)

		# the actual testing and validation datasets
		self.test  = test_gen.flow_from_directory(ddir, subset="validation", shuffle=True, target_size=shape)
		self.train = train_gen.flow_from_directory(ddir, subset= "training", shuffle=True, target_size=shape)

		self.classes = self.train.num_classes

	def load_model(self):
		m = p(saves, self.model)
		if os.path.exists(m):
			# load a pre-written model from local metadata
			self.model = tf.keras.models.load_model(p(saves, self.model))

		else:
			# create the model from the ground up
			self.model = tf.keras.Sequential([
				hub.KerasLayer(
					"https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4",
					output_shape=[1280],
					trainable=False
				),

				tf.keras.layers.Dropout(0.4),
				tf.keras.layers.Dense(self.classes, activation='softmax')
			])

			self.train()


	def visualize_output(self):
		# validations
		val_img, val_label = next(iter(self.test))

		# the actual labels from the dataset
		true_ids = np.argmax(val_label, axis=-1)
		_ = sorted(self.train.class_indices.items(), key = lambda pair: pair[1])
		true_labels = np.array([k.title() for k,v in _])

		# labels predicted by the model on the testing dataset
		pred_ids = np.argmax(self.model.predict(val_img, verbose=1), axis=-1)
		pred_labels = true_labels[pred_ids]

		# prepare matplotlib visualization
		plot.figure(figsize = (21, 20))

		for i in range(20):
			plot.subplot(6, 5, i+1)

			# the image fed into the model for prediction
			plot.imshow(val_img[i])

			# the label returned by the model as an answer.
			ans = pred_labels[i].title()
			col = "green" if pred_ids[i] == true_ids[i] else "red"

			plot.title(ans, color=col)
			plot.axis("off")

	def train(self):
		train_steps = np.ceil(self.train.samples / self.train.batch_size)
		test_steps  = np.ceil(self.test.samples  / self.test.batch_size)

		self.model.build([None, size, size, 3])
		self.model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

		# TODO: Check for overfitting
		history = self.model.fit(
			self.train, 
			epochs=10, 
			verbose=1, 
			steps_per_epoch=train_steps, 
			validata_data=self.test, 
			validation_steps=test_steps
		).history

		# save_format = 'tf' saves as a SavedModel class object instead of serializing it, or compressing into
		# an archive with the .keras extension. 
		self.model.save(p(saves, "testmodel"), save_format='tf')

if __name__ == '__main__':
	proto = Prototype(model="testmodel", visualize=32)
	proto.visualize_output()
