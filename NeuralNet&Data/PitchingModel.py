from NeuralNetwork import Model, Layer, ReLU, MeanSquaredError, Accuracy_Regression, Adam, Linear, Sigmoid, Adagrad, MeanAbsoluteError
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
import warnings
import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning, module="numpy")
#NOTE: IBB WERE NOT TRACKED UNTIL 1955

PitchingStats = pd.read_excel(r"C:\Users\blake\OneDrive\Desktop\PitchingFix.xlsx")

stat_cols = ["W","L","G","GS","CG","SHO","SV","IPouts","H","ER","HR","BB","SO","BAOpp","ERA","IBB","WP","HBP","BK","BFP","GF","R"]

for col in stat_cols:
    PitchingStats[f"next_{col}"] = PitchingStats.groupby("playerID")[col].shift(-1) #This will make a new column with next years stats next to it

df = PitchingStats.dropna(subset=[f"next_{col}" for col in stat_cols]) #Rows with null for next year (players' final year) get dropped

X = df[["age","position"] + stat_cols]      # Current season stats + age/pos
Y = df[[f"next_{col}" for col in stat_cols]]  # Next season stats

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=67) 
train_idx, test_idx = next(gss.split(X, Y, groups=df["playerID"])) #This randomly splits the data by group and returns an index
                                                                   #Next is used to grab the first and onlhy split, more splits require a for loop
X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]   
Y_train, Y_test = Y.iloc[train_idx], Y.iloc[test_idx]

X_test = X_test.to_numpy()
X_train = X_train.to_numpy()
Y_test = Y_test.to_numpy()
Y_train = Y_train.to_numpy()

def hiddenlayers(num):
    for i in range(num):
        PitchingModel.add(Layer(128, 128))
        PitchingModel.add(ReLU())

PitchingModel = Model()

PitchingModel.add(Layer(24, 128))
PitchingModel.add(ReLU())
hiddenlayers(50)
PitchingModel.add(Layer(128, 22))
PitchingModel.add(Linear())

PitchingModel.set(loss=MeanSquaredError(), optimizer=Adam(decay=1e-1), accuracy=Accuracy_Regression())

PitchingModel.finalize()

PitchingModel.train(X_train,Y_train, epochs=10, print_every=100, batch_size= 128, validation_data=(X_test, Y_test))

PitchingModel.save(r"C:\Users\blake\OneDrive\Desktop\PitchingModel.model")