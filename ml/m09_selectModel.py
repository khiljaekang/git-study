import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators
import warnings                                

warnings.filterwarnings('ignore')           # warning이라는 에러에 대해서 넘어가겠다.

iris = pd.read_csv('D:/Study/data/csv/iris.csv', header = 0)

x = iris.iloc[:, 0:4]
y = iris.iloc[:, 4]

print(x)
print(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)

allAlegorithms = all_estimators(type_filter = 'classifier')

for (name, algorithm) in allAlegorithms:
    model = algorithm()

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print(name, "의 정답률 = ", accuracy_score(y_test, y_pred))


import sklearn
print(sklearn.__version__)

#사이킷런. 0.22.1 버전에서 0.20.1 버전으로 바꿔줘야 에러안남.

