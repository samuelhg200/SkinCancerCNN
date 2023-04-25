# SkinCancerCNN

Dataset and trained files can't be added to the github repo due to space limitations of the platform. Links will be provided below.

<h2>Setting up the environment:</h2>
1 - Make sure you have python 3.11.0 installed.<br>
2 - Create a virtual environment running python -m venv .venv<br>
3 - cd into ./venv/Scripts and type activate.bat into the terminal to activate the environment. <br>
4 - cd back to your working directory where the requirements.txt file should be located. <br>
5 - Run the following command to download all the required packages: pip install -r requirements.txt<br>
6 - Troubleshoot accordingly<br>
<br>

<h2>Downloading the dataset:</h2>
1 - Download at: <a href="https://www.kaggle.com/datasets/fanconic/skin-cancer-malignant-vs-benign">Download link</a><br>
2 - Unzip and save the folder into your working directory and rename to data2<br><br>

<h2>Downloading the trained models:</h2>
1 - Download the trained EfficientNetB3 model at: <a href="https://drive.google.com/file/d/14y3Q1AneC8bO3CRymK6JsA12ROeGQ1j0/view?usp=sharing">Download link</a><br>
2 - Download the trained ResNet50 model at: <a href="https://www.kaggle.com/datasets/fanconic/skin-cancer-malignant-vs-benign">Download link</a><br>
3 - Add both files to your working directory<br><br>

<h2>Evaluating and running the trained models:</h2>
1 - Go to line 313 in cnn.py and change the name of the model to the one you wish to load.<br>
2 - Comment out lines 377 to 389 in the cnn.py file. Only a function call to Evaluate_Model() should be kept.<br>
3 - Figures will be saved to the working directory and metrics will be printed to the terminal. <br>
<br>
<h2>Training a model:</h2>

1 - Check lines 377 to 390 at the bottom of the cnn.py file. Uncomment the 3 lines for the model you want to train. <br>
2 - Figures will be saved as well as the trained model to the working directory and metrics will be printed to the terminal. <br>

