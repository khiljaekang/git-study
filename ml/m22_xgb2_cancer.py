# 과적합 방지
# 1. 훈련데이터량을 늘린다.
# 2. 피쳐 수를 줄인다.
# 3. regularization     = Dropout과 결과가 비슷 또는 똑같다

from xgboost import XGBRegressor, plot_importance      # plot_importance
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


## 데이터 가져오기
x, y = load_breast_cancer(return_X_y = True)
print(x.shape)      # (569, 30)
print(y.shape)      # (569, )

## train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2,
    shuffle = True, random_state = 77)

# Tree의 Ensemble == Forest
# Forest의 Upgrade == Boosting
# XGBooster의 장점
# 1. 딥러닝 모델에 비해 속도가 빠르다
# 2. 결측치 제거 기능을 자체적으로 제공함
# 3. 하지만 상황에 따라, 판단에 따라 사람이 처리해야 할 필요가 있음
n_estimators = 1200             # 앙상블 모델에서 트리의 갯수
learning_rate = 0.01            # 학습률, default == 0.01, 핵심 파라미터 중 하나, 머신러닝 딥러닝 양쪽에서 모두 사용함
colsample_bytree = 0.9          # 성능 좋은 모델들은 통상적으로 0.6 ~ 0.9 사이, tree의 컬럼 샘플 비율을 얼마나 할지 설정
colsample_bylevel = 0.9         # 

max_depth = 13                   # 개별 tree의 깊이, default == 6
n_jobs = -1

model = XGBRegressor(max_depth = max_depth,
                     learning_rate = learning_rate,
                     n_estimators = n_estimators,
                     n_jobs = n_jobs,
                     colsample_bytree = colsample_bytree)
                    #  colsample_bylevel = colsample_bylevel)

model.fit(x_train, y_train)
score = model.score(x_test, y_test)
print('Score : ', score)

print(model.feature_importances_)
# print(model.best_estimator_)
# print(model.best_params_)

plot_importance(model)
# plt.show()                    

# Score :  0.8233277344493765
# [3.3707290e-03 1.4795881e-02 1.8483869e-03 1.8665431e-03 6.6568162e-03
#  5.8931052e-03 9.8257465e-03 3.8782895e-02 2.3128691e-03 1.3373140e-02
#  5.3968034e-03 7.2353979e-04 1.4515867e-03 1.5249859e-03 5.8154750e-04
#  1.4353062e-03 4.1180133e-04 4.3107979e-02 9.7030657e-05 1.9410155e-04
#  7.3284702e-03 2.0865677e-02 5.7463330e-01 1.9264047e-01 6.2937024e-03
#  1.2278090e-03 3.7288207e-03 3.2886308e-02 5.9214868e-03 8.2307588e-04]
