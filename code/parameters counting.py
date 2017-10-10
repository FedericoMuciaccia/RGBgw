
# Copyright (C) 2017  Federico Muciaccia (federicomuciaccia@gmail.com)
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


import keras

from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D, Activation, Dropout, Flatten, Dense, ZeroPadding2D

from keras.layers.normalization import BatchNormalization

height = 256
width = 128
channels = 3

number_of_classes = 2

model = Sequential()

model.add(Dropout(0, input_shape=[height, width, channels])) # TODO piccolo hack per mancanza di un appropriato layer di input

for i in range(6):
    model.add(ZeroPadding2D()) # TODO
    model.add(Convolution2D(9, 3))
    #model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.8))

model.add(Flatten())
model.add(Dense(number_of_classes))
model.add(Activation('softmax'))

# plot a model summary to check the number of parameters
model.summary()
#print('parameters: ', model.count_params())




