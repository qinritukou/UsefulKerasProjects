from keras.applications import ResNet50
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

resnet = ResNet50(include_top=False, pooling='avg')

for layer in resnet.layers:
    layer.trainable = False

model = Sequential()
model.add(resnet)
model.add(Dense(1))

print(model.summary())

# model.compile(loss='mean_square_error', optimizer=Adam())
# model.fit(batch_size=32, x=train_X, y=train_Y, epochs=30)