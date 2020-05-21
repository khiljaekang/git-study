# https://medium.com/@saurabh.rathor092/simple-rnn-vs-gru-vs-lstm-difference-lies-in-more-flexible-control-5f33e07b1e57

'''
RNN VS LSTM vs GRU

결론은 데이터의 종류나 크기에 따라 SIMPLE RNN 을 사용할 것인지, LSTM을 사용할 것인지,
GRU를 사용할 것인지를 결정하는 것은 나의 몫이고, 하이퍼 파라미터 튜닝이 가장 중요한 것 같다.

같은 데이터로 SIMPLE RNN, LSTM, GRU 의 총 파라미터의 개수를 확인해본 결과 LSTM > GRU > SIMPLE RNN 이라는 결과가 나왔다.
조건이 동일 한 상태에서, 도출 된 결과값은 LSTM과 GRU는 원하던 결과값에 비교적 근사치로 나왔고, SIMPLE RNN은 조금 떨어졌다.

SIMPLE RNN의 파라미터의 개수가 적은 이유는, 기본 아키텍처의 구성을 보면 인풋과 아웃풋에 간단한 곱셈이 있고,
TANH을 통과하는 기능이 있고, 게이트는 존재하지 않는다.

반면에 LSTM의 아키텍처는 게이트가 4개 GRU는 3개로 연산에 개입한다.

가중치에 변화가 layer들을 지나갈 때 마다 4번씩 영향을 미치기 떄문에 곱하기 LSTM은 곱하기 4,

GRU는 3개이니 곱하기 3을 해준다.

이러한 차이는 총 파라미터개수의 영향을 끼친다.










'''