from keras.preprocessing.text import Tokenizer
import numpy as np

docs = ["너무 재밌어요", "참 최고에요", "참 잘 만든 영화네요",
        '추천하고 싶은 영화입니다', ' 한 번 더 보고 싶네요', '글쌔요',
        '별로에요', '생각보다 지루해요', '연기가 어색해요',
        '재미없어요', '너무 재미없다', ' 참 재밌네요']

#긍정 1, 부정 0

labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1])

#토큰화

token = Tokenizer()
token.fit_on_texts(docs)
print(token.word_index)

x = token.texts_to_sequences(docs)
print(x)


from keras.preprocessing.sequence import pad_sequences
pad_x = pad_sequences(x, padding='pre') #pre는 0이 앞에서부터 들어간다 post는 뒤에서부터

print(pad_x)                             #(12, 5)

word_size =len(token.word_index) + 1
print("전체 토큰 사이즈 :", word_size)   #25 

from keras.models import Sequential
from keras.layers import Dense, Embedding, Flatten, LSTM

model = Sequential()
model.add(Embedding(word_size, 25, input_length=5)) # 전체싸이즈, 노드의 개수 , 인풋의 길이
# model.add(Embedding(word_size, 250, input_length=5)) # 전체싸이즈, 노드의 개수 , 인풋의 길이 #(None, 5, 10)
# model.add(Embedding(25, 10)) # 전체싸이즈, 노드의 개수 , 인풋의 길이
model.add(LSTM(3))
# model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

model.summary()

model.compile(optimizer='adam', loss ='binary_crossentropy', metrics=['acc'])

model.fit(pad_x, labels, epochs=30)

acc = model.evaluate(pad_x, labels)[1]
print("acc : ", acc)







