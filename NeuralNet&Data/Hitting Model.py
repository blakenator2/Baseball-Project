from NeuralNetwork import Model, Layer, ReLU, MeanSquaredError, Accuracy_Regression, Adam, Linear, Sigmoid, Adagrad, MeanAbsoluteError
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
import warnings
import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning, module="numpy")
#NOTE: IBB WERE NOT TRACKED UNTIL 1955

HittingStats = pd.read_excel(r"C:\Users\blake\OneDrive\Desktop\BattingFix.xlsx")

stat_cols = ["G","AB","R","H","2B","3B","HR","RBI","SB","BB","SO","IBB","HBP"]

for col in stat_cols:
    HittingStats[f"next_{col}"] = HittingStats.groupby("playerID")[col].shift(-1) #This will make a new column with next years stats next to it

df = HittingStats.dropna(subset=[f"next_{col}" for col in stat_cols]) #Rows with null for next year (players' final year) get dropped

X = df[["age","position"] + stat_cols]      # Current season stats + age/pos
Y = df[[f"next_{col}" for col in stat_cols]]  # Next season stats

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=67) 
train_idx, test_idx = next(gss.split(X, Y, groups=df["playerID"])) #This randomly splits the data by group and returns an index
                                                                   #Next is used to grab the first and onlhy split, more splits require a for loop
X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]   
Y_train, Y_test = Y.iloc[train_idx], Y.iloc[test_idx]

X_test = (X_test.to_numpy().reshape(X_test.shape[0], -1).astype(np.float32)) / 100 
X_train = (X_train.to_numpy().reshape(X_train.shape[0], -1).astype(np.float32)) / 100
Y_test = (Y_test.to_numpy().reshape(Y_test.shape[0], -1).astype(np.float32)) / 100
Y_train = (Y_train.to_numpy().reshape(Y_train.shape[0], -1).astype(np.float32)) / 100 # These take the data frames, make them into numpy arrays and normalize them

def hiddenlayers(num):
    for i in range(num):
        HittingModel.add(Layer(128, 128))
        HittingModel.add(ReLU())

HittingModel = Model()

HittingModel.add(Layer(15, 128))
HittingModel.add(ReLU())
hiddenlayers(50)
HittingModel.add(Layer(128, 13))
HittingModel.add(Linear())

HittingModel.set(loss=MeanSquaredError(), optimizer=Adam(decay=1e-2), accuracy=Accuracy_Regression())

HittingModel.finalize()

HittingModel.train(X_train,Y_train, epochs=20, print_every=100, batch_size= 128, validation_data=(X_test, Y_test))

HittingModel.save(r"C:\Users\blake\OneDrive\Desktop\model.model")