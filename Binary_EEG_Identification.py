import pandas as pd
import numpy as np
from numpy import mean
from numpy import std
import matplotlib.pyplot as plt
import scipy.io
import scipy.signal
from scipy.stats import skew, kurtosis
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.metrics import roc_curve,auc, precision_score,recall_score,f1_score
from keras import layers, models, regularizers
import seaborn as sns
import sys

print("Python Version:",sys.version)
#1 Initial Variables
Featuers = ["C3_mean_PSD",      "Cz_mean_PSD",      "C4_mean_PSD",
            "Cp3_mean_PSD",     "Cpz_mean_PSD",     "Cp4_mean_PSD",
                    
            "C3_STD_PSD",       "Cz_STD_PSD",       "C4_STD_PSD",
            "Cp3_STD_PSD",      "Cpz_STD_PSD",      "Cp4_STD_PSD",
            
            "C3_mean",          "Cz_mean",          "C4_mean",
            "Cp3_mean",         "Cpz_mean",         "Cp4_mean",
            
            "C3_STD",           "Cz_STD",           "C4_STD",
            "Cp3_STD",          "Cpz_STD",          "Cp4_STD",

            "C3_var",           "Cz_var",           "C4_var",
            "Cp3_var",          "Cp_var",           "Cp4_var",
            
            "C3_range",          "Cz_range",        "C4_range",
            "Cp3_range",         "Cpz_range",       "Cp4_range",
            
            "C3_skew",           "Cz_skew",         "C4_skew",
            "Cp3_skew",          "Cp_skew",         "Cp4_skew",
            
            "C3_kurtosis",       "Cz_kurtosise",    "C4_kurtosis",
            "Cp3_kurtosis",      "Cpz_kurtosis",    "Cp4_kurtosis",
            
            
            "Label"]                                # Featuers' Names
ch_names = ['C3','Cz','C4','Cp3','Cpz','Cp4']       # channel names   
sfreq = 256                                         # sampling frequency, in hertz  
Featuers_df_1 = pd.DataFrame(columns = Featuers)    # Featuers of Subject 1
Featuers_df_2 = pd.DataFrame(columns = Featuers)    # Featuers of Subject 2


#2 Featuer Extraction
for s in range(2): # Over 2 Subjects
    ## Load Data
    if s==0:
        mat = scipy.io.loadmat(r"D:\Master UNIVPM\Projects\04\SubA_6chan_2LR_s1.mat")
        print("Extracting Featuers From Subject A")
    elif s==1:
        mat = scipy.io.loadmat(r"D:\Master UNIVPM\Projects\04\SubH_6chan_2LR.mat")
        print("Extracting Featuers From Subject H")


    EEG_data = mat["EEGDATA"]
    
    for T in range(len(EEG_data[1,1,:])): # Over all Trailes 
    
        EEG_Trail = EEG_data[:,:,T] # Extract Traile
        Fet = np.zeros([49]) #Temporally Featuers+Lable array
        
        for i in range(6): # Over Six Channels
        
            channel = EEG_Trail[i,:] # Extract Channel
            channel_normalized = channel/max(channel) # Normalize data from -1 to +1
            f, Pxx =scipy.signal.welch(channel_normalized,sfreq) #Extract PSD according to Welch thiorem

            #peaks, _ = scipy.signal.find_peaks(Pxx)
            
            #Plot PSD
            #plt.plot(Pxx)
            #plt.plot(peaks, Pxx[peaks], "x")
            #plt.plot(np.zeros_like(Pxx), "--", color="gray")
            #plt.show()
            
            # PSD Featuers
            Fet[i]         = mean(Pxx)                          # Mean of PSD
            Fet[i+6]       = std(Pxx)                           # Standered Deviation of PSD
            
            #Statistics Featuers
            Fet[i+12]      = mean(channel)                       # Amplitude Mean
            Fet[i+18]      = std(channel)                        # Amplitude Standered Deviation
            Fet[i+24]      = np.var(channel)                     # Amplitude variance
            Fet[i+30]      = max(channel)-min(channel)           # Amplitude Range
            Fet[i+36]      = skew(channel)                       # Amplitude Skew
            Fet[i+42]      = kurtosis(channel)                   # Amplitude kurtosis             
            
            
            ## Add featuers to data frame
            if s==0:
                Featuers_df_1.loc[T]=Fet
            elif s==1:
                Featuers_df_2.loc[T]=Fet

            Fet[48]=s+1

frames = [Featuers_df_1, Featuers_df_2]
    
Featuers_df = pd.concat(frames) # Final Featuers dataframe

print("All featuers was extracted")
print("Begining Data Preperation")

#3 Data preperation
##3.1 Shuffling
Data = Featuers_df.sample(frac = 1) 

##3.2 Min-Max Normalizing
x = Data.values                                 # returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()   # Create Normalizer
x_scaled = min_max_scaler.fit_transform(x)      # Appl Normalization
Data = pd.DataFrame(x_scaled)                   # Back to data frame
Data.columns=Featuers

##3.3 Prepare Train and test Data
splitRatio = 0.3
train, test = train_test_split(Data ,test_size=splitRatio,
                               random_state = 123, shuffle = True)  # Spilt to training and testing data 

train_X = train[[x for x in train.columns if x not in ["Label"]]]   # Data for traing
train_Y = train['Label']                                            # Labels for traing

feature_cols = train_X.columns

test_X = test[[x for x in train.columns if x not in ["Label"]]]     # Data fo testing
test_Y = test["Label"]                                              # Labels for traing

x_val = train_X[:50]                                                # 50 Sample for Validation
partial_x_train = train_X[50:]
partial_x_train = partial_x_train.values

y_val = train_Y[:50]
partial_y_train = train_Y[50:]
partial_y_train = partial_y_train.values

print("Data is prepeared")
print("Start Building Classifer")

#4 Classification Model
##4.1 Building Model
###4.1.1 Architectuer
model = models.Sequential()
model.add(layers.Dense(48, activation = 'relu', input_shape=(48,)))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(20, activation = 'relu'))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(10, activation = 'relu'))
#model.add(layers.Dense(6, activation = 'relu'))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(5, activation = 'relu'))
model.add(layers.Dense(1,  activation= 'sigmoid'))

###4.1.2 Hyper Parameters Tuning
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

print("Classifier Bulit\n")
print("Start Training\n")
##4.2 Training Model
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=120,
                    batch_size=10,
                    validation_data=(x_val, y_val))
print("Fininsh Training")

#5 Model Evaluation

##5.1 Network Architecture
print(model.summary())
print("Start Evaluating Data")

##4.2 Training Process
history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

epochs = range(1,len(loss_values)+1)

plt.plot(epochs, loss_values, 'bo', label="training loss")
plt.plot(epochs, val_loss_values, 'b', label="validation loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epoches")
plt.ylabel("loss")
plt.legend()
plt.show()

acc_values = history_dict['accuracy']
val_acc_values = history_dict['val_accuracy']

plt.plot(epochs, acc_values, 'bo', label="training acc")
plt.plot(epochs, val_acc_values, 'b', label="validation acc")
plt.title("Training and Validation acc")
plt.xlabel("Epoches")
plt.ylabel("acc")
plt.legend()
plt.show()

##5.3 Prediction
results = model.evaluate(test_X, test_Y)
print('Results: ', results)

predictions = model.predict(test_X)
for i in range(0,len(predictions)):
    predictions[i] = 1 if (predictions[i] > 0.5) else 0
Predictions = pd.DataFrame(predictions)
Predictions[0].value_counts().plot.pie(labels = ["1","2"])
pd.value_counts(Predictions.values.flatten())

##5.5 Metrics
print("Accuracy:",accuracy_score(test_Y, predictions))
print("f1 score:", f1_score(test_Y, predictions))
print("precision score:", precision_score(test_Y, predictions))
print("recall score:", recall_score(test_Y, predictions))
print("confusion matrix:\n",confusion_matrix(test_Y, predictions))
print("classification report:\n", classification_report(test_Y, predictions))

##5.6 Plots

###5.6.1 plot Confusion Matrix as heat map
plt.figure(figsize=(13,10))
plt.subplot(221)
sns.heatmap(confusion_matrix(test_Y, predictions),annot=True,fmt = "d",linecolor="k",linewidths=3)
plt.title("CONFUSION MATRIX",fontsize=20)

###5.6.2 plot ROC curve
test_Y_01 =test_Y.astype('category')
test_Y_01 = test_Y_01.cat.codes

fpr,tpr,thresholds = roc_curve(test_Y, predictions)
plt.subplot(222)
plt.plot(fpr,tpr,label = ("Area_under the curve :",auc(fpr,tpr)),color = "r")
plt.plot([1,0],[1,0],linestyle = "dashed",color ="k")
plt.legend(loc = "best")
plt.title("ROC - CURVE & AREA UNDER CURVE",fontsize=15)
