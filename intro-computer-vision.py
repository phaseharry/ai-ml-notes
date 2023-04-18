import tensorflow as tf

data = tf.keras.datasets.fashion_mnist
# loading data from fashion_mnist dataset

(training_images, training_labels), (test_images, test_labels) = data.load_data()
# images are in the form of 28 x 28 pixel arrays
# and labels are corresponding values from 0-9 indicating what type of clothing the image contains

training_images = training_images / 255.0
test_images = test_images / 255.0
# (/255.0) - python notation allows you to divide each element
# within the array by a particular value. Note the images contained in the dataset is grayscaled
# and only contains values between 0 and 255. By dividing by 255, we ensure each pixel/element will
# be within the range of 0 - 1. This process is called normalization. Normalization improves the performance
# when training a model with tensorflow & is usually required or your model will not learn and have massive
# errors.
# using current training setup,
# - unnormalized lead to a loss in the range of 3.53 - 0.53 and an accuracy of 0.69 - 0.82 within 5 epochs
# - normalized data lead to a loss in the range of 0.5 - 0.29 and an accuracy of 0.8 - 0.89 within 5 epochs


# before normalization, a row of the image can look like:
# [  0   0   0   0   0   0   0   0   0   0   0   0   6   0 102 204 176 134 144 123  23   0   0   0   0  12  10   0]
# after normalization
#  [0.         0.         0.         0.         0.         0.
#   0.         0.         0.         0.         0.         0.
#   0.         0.75686275 0.89411765 0.85490196 0.83529412 0.77647059
#   0.70588235 0.83137255 0.82352941 0.82745098 0.83529412 0.8745098
#   0.8627451  0.95294118 0.79215686 0.        ]

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
# Loss function is used to measure how good/bad the model's guess is during the current epoch. It takes the training data
# and predicts what it thinks is the correct answer and compares it to the actual answer and processes a loss ratio.
# Here the loss function used is sparse_categorical_crossentropy. Since we're categorizing the image into 1 of 10 categories, sparse_categorical_crossentropy
# is a good choice as it is good for measuring loss for categorical loss.

# The optimizer's role is to help the computer make another guess for the next epoch.
# metrics is an array of metrics we want logged during epochs. Here we stated, we want accuracy.
# Here we use the 'adam' optimizer function.
model.fit(training_images, training_labels, epochs=5)

model.evaluate(test_images, test_labels)
