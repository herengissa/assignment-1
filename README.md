Imagenette image classifier

This repo is part of an ML assignment: train a transfer-learning classifier on Imagenette (10 ImageNet-like categories), using fastai and a ResNet34 backbone. Training and evaluation live in the exercise notebook; the model is exported as a .pkl learner.

Live demo
The same model is served in a small Streamlit app on Hugging Face Spaces. Open the link, upload a photo, and you get a class prediction, confidence, and top alternative labels (human-readable names for the Imagenette synsets).

Try it: https://herengissa-imagenette-classifier.hf.space/

Note: The model only knows the ten Imagenette classes—unrelated images may still get a forced “best guess”.
