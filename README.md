# 1-D-DeconvNet-for-ECG-signals

#Data description
- Folder data contains all ECG data in *.mat format
- Data with name of 
	"A###" -> PAC
	"C###" -> Noise
	"V###" -> PVC
	"N###" -> Normal

#File description
- Folder script contains all scripts you need

##The process of DeconvNet includes 2 main step:
###Train an 1-D ConvNet
- "model_train_v2.py" is the script to train the 1-D ConvNet. After training, weights of model will be stored in a newly created folder: model

###Visulization
- "1-layer_deconvolution.py" is the script to visulize the trained 1-D ConvNet
- Firstly, we need to load the weights of trained model
- Then, we use these weights to get the featuremap of ConvNet and then Deconv it.
- "function.py" includes some needed functions for plotting, preprocessing and unpooling


