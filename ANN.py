import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
import uuid


#Global attribute variables, to be used in classifier.
learning = None
hidden_layers = None
activation_function = None


#Reading dataset
def ReadCSV():
    oldDataset = pd.read_csv('mammographic_masses.data.txt', header=None, na_values=['?'])
    oldDataset.columns = ["BI-RADS", "Age", "Shape", "Margin", "Density", "Severity"]
    oldDataset = oldDataset.dropna()
    oldDataset = oldDataset.drop("BI-RADS", 1)

    newDataset = pd.DataFrame() #creates a new dataframe that's empty
    newDataset = newDataset.append(oldDataset, ignore_index = True) # ignoring index is optional
    return newDataset

#dataset normalization
def Normalization(dataset):
    scaler = StandardScaler()
    scaler = scaler.fit_transform(dataset.drop('Severity',axis=1))
    dataset = pd.DataFrame(scaler,columns=dataset.columns[:-1]) 
    return dataset
 

def Classifier(parameters):

    learning = parameters[0]
    layers = parameters[1]
    nodes = parameters[2]
    hidden_layers =  [nodes for i in range(layers)]
    activation_function = tf.nn.sigmoid if parameters[3] == 0 else tf.nn.relu



    age_var = tf.feature_column.numeric_column('Age')
    shape_var = tf.feature_column.numeric_column('Shape')
    margin_var = tf.feature_column.numeric_column('Margin')
    density_var =tf.feature_column.numeric_column('Density')
    features = [age_var,shape_var,margin_var,density_var]

    return tf.estimator.DNNClassifier(hidden_units=hidden_layers, 
                                      n_classes=2,
                                      feature_columns=features, 
                                      activation_fn=activation_function,
                                      model_dir='/tmp/'+uuid.uuid4().hex,
                                      optimizer=tf.train.AdamOptimizer(learning_rate=learning),
                                      config=tf.contrib.learn.RunConfig(save_checkpoints_steps=250,
                                                                        save_checkpoints_secs=None,
                                                                        save_summary_steps=500))

                                    
def Model(classifier, x_train, y_train):

    training = tf.estimator.inputs.pandas_input_fn(x=x_train,y=y_train,batch_size=len(y_train), shuffle=True, num_epochs=3)
    model = classifier.train(input_fn=training, steps=500)

    return model


def Scorer(model, x_test, y_test):


    testing = tf.estimator.inputs.pandas_input_fn(x=x_test,batch_size=len(x_test),shuffle=False)
    predictions = list(model.predict(input_fn=testing))
    
    y_pred  = []
    for pred in predictions:
        y_pred.append(pred['class_ids'][0])

    return accuracy_score(y_test, y_pred), confusion_matrix(y_test, y_pred)
    

def K_Fold_CrossValidation(dataset, K, randomise = True):

    #Performing a shuffle, just before starting partitioning the dataset with k fold cross validation.
    if randomise:
        dataset = shuffle(dataset)

    for k in range(K):
        training = pd.DataFrame(columns=dataset.columns)
        validation = pd.DataFrame(columns=dataset.columns)
        for i in dataset.index:
            if i % K != k:
                training = training.append(dataset.iloc[[i]].copy(), ignore_index = True)
            else:
                validation = validation.append(dataset.iloc[[i]].copy(), ignore_index = True)

        yield training, validation

def CrossValidationScorer(rows, classifier):

    scores = []
    X = rows
    for training, validation in K_Fold_CrossValidation(X, K=10):
        x_train = training.iloc[:,:-1]
        y_train = training.iloc[:,-1].apply(pd.to_numeric)
        x_test = validation.iloc[:,:-1]
        y_test = validation.iloc[:,-1].apply(pd.to_numeric)

        accuracy, matrix = Scorer(Model(classifier, x_train, y_train), x_test, y_test)
        scores.append(accuracy)
    score = np.mean(scores)

    return score




def Initialization(mode, dataset, parameters):

    x = Normalization(dataset) #Setting features
    y = dataset['Severity'] #Setting class labels

    classifier = Classifier(parameters)
    accuracy = 0
    if mode == False:
        #Performing a train test split of the dataset
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

        score, matrix = Scorer(Model(classifier, x_train, y_train), x_test, y_test)
        accuracy = score
    else:
        dataset = x
        dataset['Severity'] = y
        score = CrossValidationScorer(dataset, classifier)
        accuracy = score

    return accuracy





