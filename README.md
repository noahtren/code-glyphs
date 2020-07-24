# Code Glyphs

This is an experiment that uses transformers and a convolutional neural network to learn visual representations of code (it can turn arbitrary Python code into a 128x128 RGB image.) This is possible by designing the system as an autoencoder with a visual latent space.

I'm using a few tricks to make this work, including perceptual loss and image augmentation. Much of the code to accomplish this is borrowed from an earlier project: [Cooperative Communication Networks](https://github.com/noahtren/Cooperative-Communication-Networks).

![Animation](vis/vis.gif)

This project was made possible by the [TensorFlow Research Cloud](https://www.tensorflow.org/tfrc) program. Thank you!!
