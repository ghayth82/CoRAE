# CoRAE: Concreate Relaxation Autoencoder for Differentiable Gene Selection and Pan-Cancer Classification
CoRAE is a novel global feature selection method based on concrete relaxation discrete random variable selection, which can efficiently identify a subset of most significant features that have an effective contribution in data reconstruction and classification. The proposed method is a variation of standard autoencoder where a concrete feature selection layer is added in the encoder and a standard neural network is used as a decoder.

We evaluated the proposed method using coding and non-coding gene expression profiles of 33 different cancers from TCGA. It significantly outperforms state-of-the-art methods in identifying top coding and non-coding genes.

## Installation 
Run the code [here](https://colab.research.google.com/drive/1xEkc_f2weNquAzeNDaxR_5OTwtKozdfN) in google colab.
OR
To install, use `$ pip install corae`

## Example
Below code will run on a sample gene expression dataset and return top 50 genes
```python 
from CoRAE import CoRAEFeatureSelector
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split 
from keras.layers import Dense, Dropout, LeakyReLU

def fetureByCoRAE(k, NodeInFinalLayer):
    def CoRAE_decoder(x):
        x = Dense(150)(x)
        x = LeakyReLU(0.2)(x)
        x = Dropout(0.1)(x)
        x = Dense(150)(x)
        x = LeakyReLU(0.2)(x)
        x = Dropout(0.1)(x)
        x = Dense(NodeInFinalLayer)(x)
        return x
    model = CoRAEFeatureSelector(K = k, number_try=1, number_epoch = 10, decoder_function = CoRAE_decoder)
    model.fit(x_train, x_train, x_test, x_test)
    model.get_feature_support(indxs = True)
    coef = pd.Series(model.get_feature_support(), index = X.columns)
    coef = coef[(coef != 0)].index.tolist()
    df_l = pd.DataFrame(data=coef, columns=['Gene-Ids'])
#     df_l.to_csv(PATH+'CoRAE-'+str(k)+'.csv', index=False)
    print(str(len(coef)), 'features has been selected by CAE and saved successfully')
    print(df_l)

def main():
    nFeature = 1022  # number of original features
    k=10             # number of feature to be selected
    fetureByCoRAE(k, nFeature)
if __name__== "__main__":
    main()
```

