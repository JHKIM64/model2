import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score

def normalized_plot(df, timestep, lat, lon, target, mean, std) :
    df = pd.DataFrame(df)
    df.columns = ["Obsv", "Pred"]
    plt.figure(figsize=(20, 15))
    plt.plot(df.Obsv, 'r', label='Obsv')
    plt.plot(df.Pred, 'b', label='Pred')
    plt.title(str(timestep) + ' hours '+target+' Prediction at ' + str(lat) + ',' + str(lon))
    plt.xlabel('timestep')
    plt.ylabel('normalized')
    PredR2Value = r2_score(df.Obsv, df.Pred)
    font = {'color': 'black', 'size': 14}
    plt.legend()
    plt.text(0, -1, "Prediction R-Square=" + str(round(PredR2Value, 4)), fontdict=font)
    plt.savefig(str(timestep) + 'h_'+target+' Pred_' + str(lat) + '_' + str(lon) + '.png')
    plt.show()

def actual_plot(df, timestep, lat, lon, target, mean, std, unit) :
    df = pd.DataFrame(df)
    df = df*std + mean
    df.columns = ["Obsv", "Pred"]
    plt.figure(figsize=(20, 15))
    plt.plot(df.Obsv, 'r', label='Obsv')
    plt.plot(df.Pred, 'b', label='Pred')
    plt.title(str(timestep) + ' hours '+target+' Prediction at ' + str(lat) + ',' + str(lon))
    plt.xlabel('timestep')
    plt.ylabel(unit)
    PredR2Value = r2_score(df.Obsv, df.Pred)
    font = {'color': 'black', 'size': 14}
    plt.legend()
    plt.text(0, 0, "Prediction R-Square=" + str(round(PredR2Value, 4)), fontdict=font)
    plt.savefig(str(timestep) + 'h_'+target+' Pred_' + str(lat) + '_' + str(lon) + '.png')
    plt.show()