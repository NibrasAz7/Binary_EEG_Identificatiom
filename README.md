# Binary_EEG_Identificatiom
EEG-biometrics system to identify two subjects based on EEG signals. This project is a basic project.  


## A.	Introduction 
EEG-biometric system is implemented using python version. 3.7.6. Binary classifier to identify between two subjects.

## B. Methodes
### Dataset
Data sets provided by the Dr. Cichocki's Lab \(Lab. for Advanced Brain Signal Processing\), BSI, RIKEN collaboration with Shanghai Jiao Tong University. The data set was available in 2018 on:
http://www.bsp.brain.riken.jp/~qibin/homepage/Datasets.html. 
But now I cannot find the data online. I have used a previous version that I have downloaded two years ago.
The data set was recorded g.USBamp device with Sampling rate of 256 Hz. Two subjects were used from the dataset SubA_6chan_2LR_s1 & SubH_6chan_2LR with 130 and 150 trials, respectively.All subjects' recordings include 6 EEG channels: C3, Cz, c4, Cp3, Cpz and cp4.
The subjects were asked to perform a motor Imagery (MI) task which is imagining moving left limb and right limb. For this Study, the MI task was discarded.  In other words, the EEG identification system built in this study is considered as a task independent.

### Preprocessing
Originally, the detest was prepossessed by the provider by a 2-30 Hz Band Pass filter with a 50 Hz notch filter.
By having a look at the Spectrum of the single as in Figure 3, it is noticeable that the performed filtering was good. Also, it is clear that the 50 Hz noise was eliminated.

### Feature Extraction
For Feature extraction, 48 features were used. The used features were extracted from the frequency-domain and statistical features. Regarding the frequency features, this was done by calculating the Power Spectral Density (PSD) according to Welch theorem of the EEG signal for each channel. Afterwards, the mean and standard deviation (STD) of each PSD were extracted to end up with 12 frequency features. Afterwards, statistical feathers were extracted. It contains the mean, standard deviation, variance, range, skewness, and kurtosis from the six channels.

### Data preparation
Before proceeding into the classification step, some preparation needs to be done. Firstly, shuffling the data. Afterwards, Normalizing the data using Min-Max Normalizing method, only in binary system. Finally, Splitting the data into training set (70\%) and Testing set (30\%).  50 trails from the training set were considered as a validation data.

### Classification
A deep Neural network was used with 8 dense layers, contains 3,603 trainable parameters.  Regarding the hyper-parameters, Adam was used as an optimizer while binary cross entropy was used as a loss function. The training process was performed for 120 epochs with batch size of 10. Table 3 descries the architecture of the Neural Network.

## C. Results
The systems was evaluated using the accuracy, F1-Score, precision score, and recall score obtained from the confusion matrix of the test data. Additionally, the training process was recorded. 

Accuracy:	    98.88%
F1-Score:	    98.87%
Precision:	  97.78%
Recall:     	100%


## D.	Discussion.
The binary system shows a very high accuracy, 98.80%. From the confusion matrix it is noticeable that that for the first subject, only one case of wrong authentication and zero case of successful attacks. 
