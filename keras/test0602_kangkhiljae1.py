# 모듈 임포트
import numpy as np
import pandas as pd

#.1 데이터
samsung = pd.read_csv('./data/csv/samsung.csv', index_col = 0, header = 0,
                         sep = ',', encoding = 'cp949')
# print(samsung.shape)                 #(700,1)
# print(samsung)
         

hite = pd.read_csv('./data/csv/hite.csv', index_col = 0, header = 0,
                      sep = ',', encoding = 'cp949')
# print(hite.shape)               #(720,5)
# print(hite)


samsung = samsung.dropna()
# print(samsung.shape)             #(509,1)


hite.iloc[0,1:]=hite.iloc[0,1:].fillna(0)
hite=hite.dropna()

# print(hite.shape)                #(509,5)


def remove_comma(x):
    return x.replace(',', '')

samsung['시가'] = samsung['시가'].apply(remove_comma)
# print(samsung)
# print(samsung.dtypes)               

hite['시가'] = hite['시가'].astype(str)
hite['고가'] = hite['고가'].astype(str)
hite['저가'] = hite['저가'].astype(str)
hite['종가'] = hite['종가'].astype(str)
hite['거래량'] = hite['거래량'].astype(str)

hite['시가'] = hite['시가'].apply(remove_comma)
hite['고가'] = hite['고가'].apply(remove_comma)
hite['저가'] = hite['저가'].apply(remove_comma)
hite['종가'] = hite['종가'].apply(remove_comma)
hite['거래량'] = hite['거래량'].apply(remove_comma)

# print(hite)
# print(hite.dtypes)                   # object

samsung['시가'] = samsung['시가'].astype('int64')

# print(samsung.dtypes)                # int64

hite['시가'] = hite['시가'].astype('int64')
hite['고가'] = hite['고가'].astype('int64')
hite['저가'] = hite['저가'].astype('int64')
hite['종가'] = hite['종가'].astype('int64')
hite['거래량'] = hite['거래량'].astype('int64')

# print(hite.dtypes)                   # int64

hite = hite.replace({'고가': 0}, {'고가': 39500})
hite = hite.replace({'저가': 0}, {'저가': 38500})
hite = hite.replace({'종가': 0}, {'종가': 38750})
hite = hite.replace({'거래량': 0}, {'거래량': 580653})

# print(hite.head())


samsung = samsung.sort_values(['일자'], ascending = [True])
hite = hite.sort_values(['일자'], ascending = [True])

# print(samsung)
# print(hite)

print(samsung.tail())
print(hite.tail())

samsung = samsung.values
hite = hite.values
# print(type(samsung))                 # <class 'numpy.ndarray'>
# print(type(hite))                    # <class 'numpy.ndarray'>
# print(samsung.shape)                 # (509, 1)
# print(hite.shape)                    # (508, 5)

np.save('./data/samsung.npy', arr = samsung)
np.save('./data/hite.npy', arr = hite)


np_samsung = np.load('./data/samsung.npy')
np_hite = np.load('./data/hite.npy')
# print(np_samsung)
# print(np_hite)
# print(np_samsung.shape)              # (509, 1)
# print(np_hite.shape)                 # (509, 5)




 









