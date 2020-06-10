from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
cancer = load_breast_cancer()
x_train, x_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, train_size = 0.8, random_state = 42
)

# model = DecisionTreeClassifier(max_depth =4)      # max_depth 몇 이상 올라가면 구분 잘 못함
# model = RandomForestClassifier()
model = GradientBoostingClassifier()

#max_feature : 기본값 써라!
#n_estimators : 클수록 좋다, 단점 메모리 짱 차지, 기본값 100
#n_jobs= -1 : 병렬처리 




model.fit(x_train, y_train)

acc = model.score(x_test, y_test)
print('acc: ', acc)

print(model.feature_importances_)

import matplotlib.pyplot as plt
import numpy as np
def plot_feature_importances_cancer(model):
    n_features = cancer.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_, #x 데이터가 카테고리 값인 경우에는 bar 명령과 barh 명령으로 바 차트(bar chart) 시각화를 할 수 있다.
             align='center')
    plt.yticks(np.arange(n_features), cancer.feature_names)
    plt.xlabel("Feature Importances")
    plt.ylabel("Feature")
    plt.ylim(-1,n_features)

plot_feature_importances_cancer(model)
plt.show()