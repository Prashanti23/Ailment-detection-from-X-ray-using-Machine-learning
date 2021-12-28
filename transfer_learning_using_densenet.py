
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from keras.preprocessing.image import ImageDataGenerator
from keras.applications.densenet import DenseNet121
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras import backend as K

from sklearn.model_selection import train_test_split

from keras.models import load_model

import keras
from keras import backend as K
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import os
import time
import pickle

import keras
from keras.applications.densenet import DenseNet121
from keras.models import Model
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping, TensorBoard
from keras.preprocessing import image

import os

df = pd.read_csv("archive/BBox_List_2017.csv")
df.tail()

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


from glob import glob
all_image_paths = {os.path.basename(x): x for x in 
                   glob(os.path.join('archive','images*', 'images', '*.png'))}
print('Scans found:', len(all_image_paths), ', Total Headers', dataset.shape[0])
dataset['path'] = dataset['Image Index'].map(all_image_paths.get)
dataset.drop(['Unnamed: 11'], inplace=True, axis=1)
dataset.head


train, test = train_test_split(dataset, test_size=0.2)
train, valid = train_test_split(dataset, test_size=0.25)

print(len(valid))


columns = train.keys()
columns = list(columns)
print(columns)


train.head


def get_train_generator(df, x_col, y_cols, shuffle=True, batch_size=8, seed=1, target_w = 900, target_h = 900):
    image_generator = ImageDataGenerator(
                                        samplewise_center=True,
                                        samplewise_std_normalization= True)
    generator = image_generator.flow_from_dataframe(
                                        dataframe=df,
                                        x_col=x_col,
                                        y_col=y_cols,
                                        class_mode="raw",
                                        batch_size=batch_size,
                                        shuffle=shuffle,
                                        seed=seed,
                                        target_size=(target_w,target_h))
    
    return generator

def get_test_and_valid_generator(valid_df, test_df, train_df, x_col, y_cols, sample_size=100, batch_size=8, seed=1, target_w = 900, target_h = 900):
    # get generator to sample dataset
    raw_train_generator = ImageDataGenerator().flow_from_dataframe(
        dataframe=train_df, 
        x_col= x_col, 
        y_col=labels, 
        class_mode="raw", 
        batch_size=sample_size, 
        shuffle=True, 
        target_size=(target_w, target_h))
    
    # get data sample
    batch = raw_train_generator.next()
    data_sample = batch[0]

    # use sample to fit mean and std for test set generator
    image_generator = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization= True)
    
    # fit generator to sample from training data
    image_generator.fit(data_sample)

    # get test generator
    valid_generator = image_generator.flow_from_dataframe(
            dataframe=valid_df,
            x_col=x_col,
            y_col=y_cols,
            class_mode="raw",
            batch_size=batch_size,
            shuffle=False,
            seed=seed,
            target_size=(target_w,target_h))

    test_generator = image_generator.flow_from_dataframe(
            dataframe=test_df,
            x_col=x_col,
            y_col=y_cols,
            class_mode="raw",
            batch_size=batch_size,
            shuffle=False,
            seed=seed,
            target_size=(target_w,target_h))
    return valid_generator, test_generator

labels = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Nodule Mass', 'No Finding', 'Pleural_thickening', 'Pneumonia', 'Pneumothorax']

train_generator = get_train_generator(train,"path", labels)
valid_generator, test_generator= get_test_and_valid_generator(valid, test, train, "path", labels)

sns.set_style("white")
generated_image, label = train_generator.__getitem__(0)
plt.imshow(generated_image[0], cmap='gray')
plt.colorbar()
plt.title('Raw Chest X Ray Image')
print(f"The dimensions of the image are {generated_image.shape[1]} pixels width and {generated_image.shape[2]} pixels height")
print(f"The maximum pixel value is {generated_image.max():.4f} and the minimum is {generated_image.min():.4f}")
print(f"The mean value of the pixels is {generated_image.mean():.4f} and the standard deviation is {generated_image.std():.4f}")

plt.xticks(rotation=90)
plt.bar(x=labels, height=np.mean(train_generator.labels, axis=0))
plt.title("Frequency of Each Class")
plt.show()


def compute_class_freqs(labels):
    ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###
    
    # total number of patients (rows)
    N = labels.shape[0]
    
    positive_frequencies = np.sum(labels, axis=0)/N
    negative_frequencies = (N - np.sum(labels, axis=0))/N  # broadcasting of N to a line vector of dim num_classes

    ### END CODE HERE ###
    return positive_frequencies, negative_frequencies

freq_pos, freq_neg = compute_class_freqs(train_generator.labels)



data = pd.DataFrame({"Class": labels, "Label": "Positive", "Value": freq_pos})
data = data.append([{"Class": labels[l], "Label": "Negative", "Value": v} for l,v in enumerate(freq_neg)], ignore_index=True)
plt.xticks(rotation=90)
f = sns.barplot(x="Class", y="Value", hue="Label" ,data=data)


pos_weights = freq_neg
neg_weights = freq_pos
pos_contribution = freq_pos * pos_weights 
neg_contribution = freq_neg * neg_weights


data = pd.DataFrame({"Class": labels, "Label": "Positive", "Value": pos_contribution})
data = data.append([{"Class": labels[l], "Label": "Negative", "Value": v} 
                        for l,v in enumerate(neg_contribution)], ignore_index=True)
plt.xticks(rotation=90)
sns.barplot(x="Class", y="Value", hue="Label" ,data=data)

x, y = train_generator.__getitem__(100)
plt.imshow(x[0]);

def get_weighted_loss(pos_weights, neg_weights, epsilon=1e-7):
    """
    Return weighted loss function given negative weights and positive weights.

    Args:
      pos_weights (np.array): array of positive weights for each class, size (num_classes)
      neg_weights (np.array): array of negative weights for each class, size (num_classes)
    
    Returns:
      weighted_loss (function): weighted loss function
    """
    def weighted_loss(y_true, y_pred):
        """
        Return weighted loss value. 

        Args:
            y_true (Tensor): Tensor of true labels, size is (num_examples, num_classes)
            y_pred (Tensor): Tensor of predicted labels, size is (num_examples, num_classes)
        Returns:
            loss (Float): overall scalar loss summed across all classes
        """
        # initialize loss to zero
        loss = 0.0
        
        for i in range(len(pos_weights)):
            # for each class, we add average weighted loss for that class 
            pos_loss = -1 * K.mean(pos_weights[i] * y_true[:, i] * K.log(y_pred[:, i] + epsilon))
            neg_loss = -1 * K.mean(neg_weights[i] * (1 - y_true[:, i]) * K.log(1 - y_pred[:, i] + epsilon))
            loss += pos_loss + neg_loss            
        return loss
    
        ### END CODE HERE ###
    return weighted_loss



def load_C3M3_model():
   
    class_pos = train.loc[:, labels].sum(axis=0)
    class_neg = len(train) - class_pos
    class_total = class_pos + class_neg

    pos_weights = class_pos / class_total
    neg_weights = class_neg / class_total
    print("Got loss weights")
    # create the base pre-trained model
    base_model = DenseNet121(weights='densenet.hdf5', include_top=False)
    print("Loaded DenseNet")
    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # and a logistic layer
    predictions = Dense(len(labels), activation="sigmoid")(x)
    print("Added layers")

    model = Model(inputs=base_model.input, outputs=predictions)
    
    model.compile(optimizer='adam', loss=keras.losses.categorical_crossentropy)
    print("Compiled Model")

    return model
model = load_C3M3_model()

layers_ = model.layers
print('total number of layers =',len(layers_))

conv2D_layers = [layer for layer in model.layers 
                if str(type(layer)).find('Conv2D') > -1]
print('Model input -------------->', model.input)
print('Feature extractor output ->', model.get_layer('conv5_block16_concat').output)
print('Model output ------------->', model.output)

history = model.fit(train_generator, 
                              validation_data=valid_generator,
                              steps_per_epoch=10, 
                              validation_steps=2, 
                              epochs = 1)

plt.plot(history.history['loss'])
plt.ylabel("loss")
plt.xlabel("epoch")
plt.title("Training Loss Curve")
plt.show()
