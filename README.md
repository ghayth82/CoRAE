# CoRAE: Concreate Relaxation Autoencoder for Differentiable Gene Selection and Pan-Cancer Classification
CoRAE is a novel global feature selection method based on concrete relaxation discrete random variable selection, which can efficiently identify a subset of most significant features that have an effective contribution in data reconstruction and classification. The proposed method is a variation of standard autoencoder where a concrete feature selection layer is added in the encoder and a standard neural network is used as a decoder.

We evaluated the proposed method using coding and non-coding gene expression profiles of 33 different cancers from TCGA. It significantly outperforms state-of-the-art methods in identifying top coding and non-coding genes.

## Installation 
To install, use `$ pip install corae`

## Example
Below code will run on a sample gene expression dataset and return top 50 genes
```python 
from corae import CoRAEFeatureSelector
from keras.layers import Dense, Dropout, LeakyReLU
import numpy as np
import pnadas as pd

df = pd.read_csv("gene-expression.csv")
X = df.iloc[:,1:-1]
y = df.iloc[:,-1]
X_norm = MinMaxScaler().fit_transform(X)
x_train, x_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.25, random_state=31)

def decoder(x):
    x = Dense(150)(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.1)(x)
    x = Dense(150)(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.1)(x)
    x = Dense(NodeInFinalLayer)(x)
    return x

model = CoRAEFeatureSelector(K = 50, output_function = decoder, num_epochs = 100, tryout_limit=2)
model.fit(x_train, x_train, x_test, x_test)
model.get_support(indices = True)
coef = pd.Series(model.get_support(), index = X.columns)
coef = coef[(coef != 0)].index.tolist()
df_l = pd.DataFrame(data=coef, columns=['features'])
df_l.to_csv(PATH+'CoRAE-'+str(k)+'.csv', index=False)
print(str(len(coef)), 'features has been selected by CoRAE and saved successfully')```

