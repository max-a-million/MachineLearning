import pandas as pd
import matplotlib.pyplot as mplot

df = pd.read_csv("d:\\workspace\\MachineLearning\\Examples\\train.csv")

y = df["y"]
X = df.drop("y", 1)

mplot.figure()
mplot.hist(y, bins=20)
mplot.xlabel('Target value')
mplot.ylabel('Occurences')
mplot.title('Distribution of the target value')
print('min: {} max: {} mean: {} std: {}'.format(min(y), max(y), y.mean(), y.std()))
