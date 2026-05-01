from keras.models import load_model

model = load_model("brain_tumor_model.h5")
model.summary()
"""
This model is a custom Convolutional Neural Network (CNN). It is not using ResNet, VGG, or any pre-trained model.



It follows a sequential structure (Model: "sequential").

It has four convolutional layers (Conv2D) followed by max pooling (MaxPooling2D).

It flattens the output and passes it through two dense layers at the end.

The final dense layer (dense_1) has 4 neurons, which matches your four tumor classes. """