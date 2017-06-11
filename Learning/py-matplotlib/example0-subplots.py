import pandas as pd
import matplotlib.pyplot as mplot
from sklearn import preprocessing as pre

df = pd.read_csv("d:\\workspace\\MachineLearning\\Examples\\train.csv")

y = df["y"]
X = df.drop("y", 1)

""" encode a feature """
def encodeF(x):
    enc = pre.LabelEncoder()
    enc.fit(x)
    return pd.Series(enc.transform(x))

""" display one subchart """
def getChart(fig, h, v, c, key, y):
    fig.add_subplot(h, v, c).scatter(encodeF(X[key]), y)

fig = mplot.figure()
h = 3
v = 3
getChart(fig, h, v, 1, "X0", y)
getChart(fig, h, v, 2, "X1", y)
getChart(fig, h, v, 3, "X2", y)
getChart(fig, h, v, 4, "X3", y)
getChart(fig, h, v, 5, "X4", y)
getChart(fig, h, v, 6, "X5", y)
getChart(fig, h, v, 7, "X6", y)
getChart(fig, h, v, 8, "X8", y)
getChart(fig, h, v, 9, "X10", y)