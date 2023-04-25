# SkinCancerCNN

Setting up the environment: 

1 - Make sure you have python 3.11.0 installed.
2 - Create a virtual environment running python -m venv .venv
3 - cd into ./venv/Scripts and type activate.bat into the terminal to activate the environment. 
4 - cd back to your working directory where the requirements.txt file should be located. 
5 - Run the following command to download all the required packages: pip install -r requirements.txt
6 - Troubleshoot accordingly


Running the trained models:

1 - Go to line 313 in cnn.py and change the name of the model to the one you wish to load.
2 - Comment out lines 377 to 389 in the cnn.py file. Only a function call to Evaluate_Model() should be kept.
3 - Figures will be saved to the working directory and metrics will be printed to the terminal. 

Training a model:

1 - Check lines 377 to 390 at the bottom of the cnn.py file. Uncomment the 3 lines for the model you want to train. 
2 - Figures will be saved as well as the trained model to the working directory and metrics will be printed to the terminal. 

