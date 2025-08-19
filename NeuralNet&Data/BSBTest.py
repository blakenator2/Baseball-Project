from NeuralNetwork import Model
import pandas as pd
import numpy as np

df = pd.read_excel(r"C:\Users\blake\OneDrive\Desktop\PitchingFix.xlsx")
stat_cols_H = ["age","position","G","AB","R","H","2B","3B","HR","RBI","SB","BB","SO","IBB","HBP"]
stat_cols_P = ["age","position","W","L","G","GS","CG","SHO","SV","IPouts","H","ER","HR","BB","SO","BAOpp","ERA","IBB","WP","HBP","BK","BFP","GF","R"]
    
guess = df[(df['playerID'] == 'skenepa01') & (df['yearID'] == 2024)][stat_cols_P]

model = Model.load(r"C:\Users\blake\OneDrive\Desktop\PitchingModel.model")
confidences = model.predict(guess.values)
predictions = model.output_layer_activation.predictions(confidences)
predictions = (predictions.reshape(predictions.shape[0], -1).astype(np.float32)) * 100
print(predictions)