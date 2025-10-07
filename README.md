## EXNO-3-DS

## AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

## ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

## FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

## Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

## CODING AND OUTPUT:
```
import pandas as pd
df=pd.read_csv("/content/Encoding Data.csv")
df
```

<img width="424" height="486" alt="image" src="https://github.com/user-attachments/assets/8f1c632f-fe4c-46a7-bf03-cdd1b8533f11" />

```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```
<img width="607" height="315" alt="image" src="https://github.com/user-attachments/assets/02f5bfe5-7090-45cc-8c4f-c947386cffb6" />

```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```

<img width="405" height="462" alt="image" src="https://github.com/user-attachments/assets/fef99c9b-01fc-485d-8b78-049f63f8ae87" />

```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
<img width="396" height="504" alt="image" src="https://github.com/user-attachments/assets/5738c655-55c2-415c-b173-30ce0af0b35e" />

```
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse_output=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
df2=pd.concat([df2,enc],axis=1)
df2
```
<img width="492" height="542" alt="image" src="https://github.com/user-attachments/assets/735dcc57-560c-4465-a5e3-8082b6fbfbae" />

```
pd.get_dummies(df2,columns=["nom_0"])
```

<img width="720" height="439" alt="image" src="https://github.com/user-attachments/assets/397990a1-3d0b-4c4a-8b7d-241bfee19fed" />

```
pip install --upgrade category_encoders
from category_encoders import BinaryEncoder
df=pd.read_csv("/content/data.csv")
df
```

<img width="573" height="494" alt="image" src="https://github.com/user-attachments/assets/ba2fc106-a47e-488d-8002-4380870b8128" />

```
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
df
```

<img width="521" height="488" alt="image" src="https://github.com/user-attachments/assets/dfe60102-35cc-4626-9c63-f1ee3ce83a4e" />

```
dfb=pd.concat([df,nd],axis=1)
dfb
```

<img width="757" height="459" alt="image" src="https://github.com/user-attachments/assets/62916e7c-b958-49b6-8704-85cb67e5ee66" />

```
from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC['Target'])
CC=pd.concat([CC,new],axis=1)
CC
```

<img width="639" height="557" alt="image" src="https://github.com/user-attachments/assets/83ccadd4-822f-451b-8ee7-cfcc6fbdd8c4" />

```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/Data_to_Transform.csv")
df
```

<img width="916" height="598" alt="image" src="https://github.com/user-attachments/assets/5dbe9fda-ac70-47ec-808b-150ad5683879" />

```
df.skew()
```

<img width="313" height="220" alt="image" src="https://github.com/user-attachments/assets/66238de6-6587-44af-99ad-f488feacfc7e" />

```
np.log(df["Highly Positive Skew"])
```

<img width="260" height="514" alt="image" src="https://github.com/user-attachments/assets/fe165f88-b940-4af3-849d-534ab7c1eb55" />

```
np.reciprocal(df["Moderate Positive Skew"])
```

<img width="403" height="555" alt="image" src="https://github.com/user-attachments/assets/441098d7-b786-4838-85c2-dbf4ab180557" />

```
np.sqrt(df["Highly Positive Skew"])
```

<img width="374" height="557" alt="image" src="https://github.com/user-attachments/assets/7b838c57-a59a-4fe6-b6a4-807e1f7fe54a" />

```
np.square(df["Highly Positive Skew"])
```

<img width="383" height="558" alt="image" src="https://github.com/user-attachments/assets/d7014629-a8a2-4f96-979a-80c4780bffde" />

```
from scipy import stats
df["Highly Positive Skew_boxcox"],parameters=stats.boxcox(df["Highly Positive Skew"])
df
```

<img width="1110" height="560" alt="image" src="https://github.com/user-attachments/assets/2a773fe6-b06e-4efc-9b2e-a7d2bb3af8a9" />

```
df.skew()
```

<img width="382" height="302" alt="image" src="https://github.com/user-attachments/assets/b679870f-c071-4f28-8e3f-3db4fb885675" />


```
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
```

<img width="844" height="360" alt="image" src="https://github.com/user-attachments/assets/4800836d-fb47-4c03-93c2-6dada5e5d054" />

```
import pandas as pd
from sklearn.preprocessing import QuantileTransformer
df = pd.read_csv("Data_to_Transform.csv")
qt = QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"] = qt.fit_transform(df[["Moderate Negative Skew"]])
df
```

<img width="1113" height="471" alt="image" src="https://github.com/user-attachments/assets/7978dab7-3915-490b-a562-2c9765fcc788" />

```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```

<img width="682" height="625" alt="image" src="https://github.com/user-attachments/assets/460c26b9-119b-4ae2-aba2-94e9cd8e0331" />

```
import numpy as np
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```

<img width="677" height="582" alt="image" src="https://github.com/user-attachments/assets/d8ebc252-b40a-4fd5-91fb-df7179453837" />

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```

<img width="714" height="613" alt="image" src="https://github.com/user-attachments/assets/6e210df0-a45d-47c1-afc0-e0375ce32d25" />

```
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
```

<img width="722" height="576" alt="image" src="https://github.com/user-attachments/assets/8bacc038-1c94-4f21-b2e0-a429f9ef8b14" />

```
dt=pd.read_csv("/content/titanic_dataset (1).csv")
dt
```

<img width="1312" height="520" alt="image" src="https://github.com/user-attachments/assets/53d37534-cf64-4dcc-99e4-f9948ace650b" />

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
dt["Age_1"]=qt.fit_transform(dt[["Age"]])
sm.qqplot(dt['Age'],line='45')
plt.show()
```

<img width="654" height="620" alt="image" src="https://github.com/user-attachments/assets/d7b617f4-6f6f-4181-8348-75dd2da2d5f2" />

```
 sm.qqplot(df["Highly Negative Skew_1"],line='45')
 plt.show()
```

<img width="709" height="559" alt="image" src="https://github.com/user-attachments/assets/9dce5e4c-9f34-41e7-bda6-cb69f99fbedb" />



## RESULT:
Thus, To read the given data and perform Feature Encoding and Transformation process and save the data to a file is successfully completed.     


       
