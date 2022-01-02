# Import dependencies
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import time 
import pickle
from glob import glob

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.densenet import DenseNet121
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras import backend as K
from keras.models import load_model
from keras.applications.densenet import DenseNet121
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping, TensorBoard
from keras.preprocessing import image

"""
Part 1: Dataset Preprocessing.
Authored by Vijeta Nayak
"""
# Read the dataset
dataset=pd.read_csv("archive/Data_Entry_2017.csv")
print(dataset.head())

# Data Preprocessing Dealing with labels
dataset["Atelectasis"] =0
dataset["Consolidation"] =0
dataset["Infiltration"] =0
dataset["Pneumothorax"]=0
dataset["Edema"]=0
dataset["Emphysema"]=0
dataset["Fibrosis"]=0
dataset["Effusion"]=0
dataset["Pneumonia"]=0
dataset["Pleural_thickening"]=0
dataset["Cardiomegaly"]=0
dataset["Nodule Mass"]=0
dataset["Hernia"]=0
dataset["No Finding"]=0

#Replacing the labels with columns
dataset.loc[dataset["Finding Labels"].str.contains("Atelectasis"),"Atelectasis"]=1
dataset.loc[dataset["Finding Labels"].str.contains("Consolidation"),"Consolidation"]=1
dataset.loc[dataset["Finding Labels"].str.contains("Infiltration"),"Infiltration"]=1
dataset.loc[dataset["Finding Labels"].str.contains("Pneumothorax"),"Pneumothorax"]=1
dataset.loc[dataset["Finding Labels"].str.contains("Edema"),"Edema"]=1
dataset.loc[dataset["Finding Labels"].str.contains("Emphysema"),"Emphysema"]=1
dataset.loc[dataset["Finding Labels"].str.contains("Fibrosis"),"Fibrosis"]=1
dataset.loc[dataset["Finding Labels"].str.contains("Effusion"),"Effusion"]=1
dataset.loc[dataset["Finding Labels"].str.contains("Pneumonia"),"Pneumonia"]=1
dataset.loc[dataset["Finding Labels"].str.contains("Pleural_thickening"),"Pleural_thickening"]=1
dataset.loc[dataset["Finding Labels"].str.contains("Cardiomegaly"),"Cardiomegaly"]=1
dataset.loc[dataset["Finding Labels"].str.contains("Nodule Mass"),"Nodule Mass"]=1
dataset.loc[dataset["Finding Labels"].str.contains("Hernia"),"Hernia"]=1
dataset.loc[dataset["Finding Labels"].str.contains("No Finding"),"No Finding"]=1

# Adding path column in data frame
all_image_paths = {os.path.basename(x): x for x in 
                   glob(os.path.join('archive','images*', 'images', '*.png'))}
print('Scans found:', len(all_image_paths), ', Total Headers', dataset.shape[0])
dataset['path'] = dataset['Image Index'].map(all_image_paths.get)
dataset.drop(['Unnamed: 11'], inplace=True, axis=1)
dataset.head

#Creating a label array with possible output values.
labels = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Nodule Mass', 'No Finding', 'Pleural_thickening', 'Pneumonia', 'Pneumothorax']

# Divide the dataset into train, test and validate dataframes. 
train, test = train_test_split(dataset, test_size=0.2)
train, valid = train_test_split(dataset, test_size=0.25)

"""
Part 2: Image Generators 
Authored by: Anirudh Poroorkara
"""
# Method to create a train generator. 
def get_train_generator(df, x_col, y_cols, shuffle=True, batch_size=8, seed=1, target_w = 900, target_h = 900):
    # Create a image generator from Keras
    image_generator = ImageDataGenerator(
                                        samplewise_center=True,
                                        samplewise_std_normalization= True)
    
    # Create the train image generator to flow from the given dataframe
    train_generator = image_generator.flow_from_dataframe(
                                        dataframe=df,
                                        x_col=x_col,
                                        y_col=y_cols,
                                        class_mode="raw",
                                        batch_size=batch_size,
                                        shuffle=shuffle,
                                        seed=seed,
                                        target_size=(target_w,target_h))
    return train_generator

# Method to create a test and validate generator 
def get_test_and_valid_generator(valid_df, test_df, train_df, x_col, y_cols, sample_size=100, batch_size=8, seed=1, target_w = 900, target_h = 900):
    
    # Create a sample train generator for computing mean and standard deviation. 
    raw_train_generator = ImageDataGenerator().flow_from_dataframe( dataframe=train_df, 
                                                                    x_col= x_col, 
                                                                    y_col=labels, 
                                                                    class_mode="raw", 
                                                                    batch_size=sample_size, 
                                                                    shuffle=True, 
                                                                    target_size=(target_w, target_h))
    
    # Create a random sample batch
    batch = raw_train_generator.next()
    data_sample = batch[0]

    # Create an image generator
    image_generator = ImageDataGenerator(featurewise_center=True, 
                                         featurewise_std_normalization= True)
    
    # fit the mean and standard deviation on the obtained data sample
    image_generator.fit(data_sample)

    #Create a test generator
    test_generator = image_generator.flow_from_dataframe( dataframe=test_df,
                                                          x_col=x_col,
                                                          y_col=y_cols,
                                                          class_mode="raw",
                                                          batch_size=batch_size,
                                                          shuffle=False,
                                                          seed=seed,
                                                          target_size=(target_w,target_h))
    
    #Create a valid generator
    valid_generator = image_generator.flow_from_dataframe(dataframe=valid_df,
                                                          x_col=x_col,
                                                          y_col=y_cols,
                                                          class_mode="raw",
                                                          batch_size=batch_size,
                                                          shuffle=False,
                                                          seed=seed,
                                                          target_size=(target_w,target_h))
    
    # Return the created generators
    return valid_generator, test_generator

# Create the train, test and validate generator
train_generator = get_train_generator(train,"path", labels)
valid_generator, test_generator= get_test_and_valid_generator(valid, test, train, "path", labels)

"""
# Testing the generator

sns.set_style("white")
generated_image, label = train_generator.__getitem__(0)
plt.imshow(generated_image[0], cmap='gray')
"""

"""
Part 3: Model development 
Authored by: Anirudh Poroorkara
"""

# Finding the frequency of each class label
plt.xticks(rotation=90)
plt.bar(x=labels, height=np.mean(train_generator.labels, axis=0))
plt.title("Frequency of Each Class")
plt.show()

# Method to compute the positive and negative frequencies in each class.
def compute_class_freqs(labels):
    N = labels.shape[0]
    
    positive_frequencies = np.sum(labels, axis=0)/N
    negative_frequencies = (N - np.sum(labels, axis=0))/N 

    return positive_frequencies, negative_frequencies

# Calculating the frequencies and plotting the graph
freq_pos, freq_neg = compute_class_freqs(train_generator.labels)

data = pd.DataFrame({"Class": labels, "Label": "Positive", "Value": freq_pos})
data = data.append([{"Class": labels[l], "Label": "Negative", "Value": v} for l,v in enumerate(freq_neg)], ignore_index=True)
plt.xticks(rotation=90)
f = sns.barplot(x="Class", y="Value", hue="Label" ,data=data)

# Balancing the dataset for class imbalance
pos_weights = freq_neg
neg_weights = freq_pos
pos_contribution = freq_pos * pos_weights 
neg_contribution = freq_neg * neg_weights

"""
# Testing the results
data = pd.DataFrame({"Class": labels, "Label": "Positive", "Value": pos_contribution})
data = data.append([{"Class": labels[l], "Label": "Negative", "Value": v} 
                        for l,v in enumerate(neg_contribution)], ignore_index=True)
plt.xticks(rotation=90)
sns.barplot(x="Class", y="Value", hue="Label" ,data=data)
"""

def generate_model():
  
    # create the base pre-trained model
    base_model = DenseNet121(weights='densenet.hdf5', include_top=False)
    x = base_model.output
    
    # add a global spatial average pooling layer
    x = GlobalAveragePooling2D()(x)
    
    # add logistic layer
    predictions = Dense(len(labels), activation="sigmoid")(x)
    
    #create model
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer='adam', loss=keras.losses.categorical_crossentropy)

    return model 

model = generate_model()

"""
# Checking total number of layers
layers_ = model.layers
print('total number of layers =',len(layers_))
"""

"""
Part 4: Training 
Authored by Bhavya Bhimani
"""
#Training the model
history = model.fit(train_generator, 
                    validation_data=valid_generator,
                    steps_per_epoch=100, 
                    validation_steps=25, 
                    epochs = 5)

"""
Part 5: Evaluation
Authored by Prashanti Pamulapati
"""

# Evaluation
plt.plot(history.history['loss'])
plt.ylabel("loss")
plt.xlabel("epoch")
plt.title("Training Loss Curve")
plt.show()

# Prediction
predicted_vals = model.predict_generator(test_generator, steps = len(test_generator))

# Generating ROC curve
def get_roc_curve(labels, predicted_vals, generator):
    auc_roc_vals = []
    for i in range(len(labels)):
        try:
            gt = generator.labels[:, i]
            pred = predicted_vals[:, i]
            auc_roc = roc_auc_score(gt, pred)
            auc_roc_vals.append(auc_roc)
            fpr_rf, tpr_rf, _ = roc_curve(gt, pred)
            plt.figure(1, figsize=(10, 10))
            plt.plot([0, 1], [0, 1], 'k--')
            plt.plot(fpr_rf, tpr_rf,
                     label=labels[i] + " (" + str(round(auc_roc, 3)) + ")")
            plt.xlabel('False positive rate')
            plt.ylabel('True positive rate')
            plt.title('ROC curve')
            plt.legend(loc='best')
        except:
            print(
                f"Error in generating ROC curve for {labels[i]}. "
                f"Dataset lacks enough examples."
            )
    plt.show()
    return auc_roc_vals
  
  auc_rocs = get_roc_curve(labels, predicted_vals, test_generator)
