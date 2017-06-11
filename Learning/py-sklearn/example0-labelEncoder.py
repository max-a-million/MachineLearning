import pandas as pd
from sklearn import preprocessing as pre

df = pd.read_csv("d:\\workspace\\MachineLearning\\Examples\\train.csv")

y = df["y"]
X = df.drop("y", 1)
x = X["X0"]

# As we can see, X0 is a categorical feature.
print(x)

# Let's encode this feature with LabelEncoder.
# As a result we'll see numerical representation this future.
enc = pre.LabelEncoder()
enc.fit(x)
x = enc.transform(x)
print(x)