import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

def cp_predict(target, obsv, pred) :
    plt.plot(obsv,pred,'ro')
    plt.title(target +'Training Set')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.show()
    #Compute R-Square value for training set
    TestR2Value = r2_score(obsv,pred)
    print("Training Set R-Square=", TestR2Value)