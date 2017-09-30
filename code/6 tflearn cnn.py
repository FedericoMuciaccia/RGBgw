
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


# NOTA: un ragazzo che studiava medicina preso a caso in biblioteca ci ha messo <60 immagini per capire come classificare

import netCDF4 # TODO BUG: anche se inutilizato, va importato per primo
#import h5py

import xarray

import tflearn

import sklearn.model_selection
import sklearn.utils

## data loading
#train_images = xarray.open_dataarray('/storage/users/Muciaccia/train_images.netCDF4')
#train_classes = xarray.open_dataarray('/storage/users/Muciaccia/train_classes.netCDF4')
#test_images = xarray.open_dataarray('/storage/users/Muciaccia/test_images.netCDF4')
#test_classes = xarray.open_dataarray('/storage/users/Muciaccia/test_classes.netCDF4')
#
#number_of_train_samples, height, width, channels = train_images.shape
#number_of_train_samples, number_of_classes = train_classes.shape

# TODO hardcoded
height, width, channels = 256, 128, 3
number_of_classes = 2

# no data preprocessing is required
# data augmentation # TODO BUG of tflearn (core dumped)
#image_augmentation = tflearn.ImageAugmentation()
#image_augmentation.add_random_flip_leftright()
#image_augmentation.add_random_flip_updown()

# build the convolutional network
network = tflearn.layers.core.input_data(shape=[None, height, width, channels], name='input') # TODO data_augmentation=image_augmentation
#network = tflearn.layers.core.dropout(network, 0.8)
for i in range(6): # 6 convolutional block is the maximum dept with the given image size
    network = tflearn.layers.conv.conv_2d(network,
                                          nb_filter=9,
                                          filter_size=3,
                                          strides=1,
                                          padding='valid',
                                          activation='linear',
                                          bias=True, # TODO controllare per batch normalization
                                          weights_init='uniform_scaling',
                                          bias_init='zeros',
                                          regularizer=None,
                                          weight_decay=0) # regularizer='L2', weight_decay=0.001, scope=None
#    network = tflearn.layers.normalization.batch_normalization(network,
#                                                               beta=0.0,
#                                                               gamma=1.0, # TODO il layer dopo è un ReLU
#                                                               decay=0.9, # TODO default: decay=0.9
#                                                               stddev=0.002, # standard deviation for weights initialization
#                                                               trainable=True)
    network = tflearn.activation(network, activation='relu')
    #network = tflearn.layers.normalization.local_response_normalization(network) # TODO depth_radius=5, bias=1.0, alpha=0.0001, beta=0.75 interchannel or intrachannel?
    network = tflearn.layers.conv.max_pool_2d(network, kernel_size=2) # strides=None, padding='same'
    network = tflearn.layers.core.dropout(network, 0.8)
network = tflearn.layers.core.flatten(network)
#network = tflearn.layers.core.fully_connected(network, n_units=10, activation='relu') # TODO regularizer and weight decay
network = tflearn.layers.core.fully_connected(network, n_units=number_of_classes, bias=True, activation='softmax', weights_init='truncated_normal', bias_init='zeros', regularizer=None, weight_decay=0)
#network = tflearn.layers.core.fully_connected(network, n_units=number_of_classes, bias=True, weights_init='truncated_normal', bias_init='zeros', activation='softmax') # weight_decay=0.001, scope=None
network = tflearn.layers.estimator.regression(network, optimizer='adam', learning_rate=0.001, batch_size=128, loss='categorical_crossentropy', shuffle_batches=True, name='target') # metric='default', to_one_hot=False, n_classes=None, validation_monitors=None

model = tflearn.DNN(network, tensorboard_verbose=0) # 3

# Keras BatchNormalization
## momentum for the moving average
#momentum=0.99
## if True, add offset of beta to normalized tenso
#center=True
## if True, multiply by gamma. If False, gamma is not used. When the next layer is linear (also e.g. nn.relu), this can be disabled since the scaling will be done by the next layer # TODO
#scale=True
## initializers for the weights
#beta_initializer='zeros'
#gamma_initializer='ones'
#moving_mean_initializer='zeros'
#moving_variance_initializer='ones'
## optional regularizers for the weights
#beta_regularizer=None
#gamma_regularizer=None
## optional constraints for the weights
#beta_constraint=None
#gamma_constraint=None

# TODO provare batch_size 64, controllare feed_dict nell'altra rete, controllare summary della rete, fare la prova in bianconero, controllare i valori strani di validation loss and accuracy

# df/dt = spindown
# 0-order pipeline
# Band Sample Data (BSD)
# manuale Sergio Frasca su BSD
# sfdb nel tempo levando il primo e l'iltimo quarto perché interallacciate
# frame grezzi (nel tempo) (fft e poi estrarre la banda e pulire tutto)
# libreria pubblica di LIGO lal-lalaps (Paola Laeci) (cercare eccessi di potenza per le wavelet)
# pss_frameS.c (per leggere i frame)
# f = 0 - 2048
# t_FFT circa come t_coerenza (per massimizzare il SNR)
# teamspeak + reveal.js per le slides sincronizzate
# vidyo (video+slides+group_chat)
# talk massimo 20 minuti
# numero DCC messo nella presentazione (ad una conferenza)

# pwd nella home
# /home/VIRGO/muciaccia
# software su virgo3
# dati su virgo4 (molto spazio)
# lcg-tools (grid) + directory speciale in virgo4 con scrittura normale
# path logico VS path fisico
# cartella magica: /storage/gpfs_virgo4/CW_Roma1/
# ui01 ui02 ui03 (poco potenti)
# referenti CNAF: Matteo Tenti, Lucia Morganti

# dati Virgo 02 (C01 ?)
# dati BSD nel tempo
# whitening
# limite peakmap
# macchina con GPU
# metodo per rigetto spettri e differenza relativa
# atoregressivo (articolo Pia) (codice pss_sfdb.c) short_psar_rev_freq (media autoregressiva fatta dal basso verso l'alto (ps spectrum ar autoregressive rev reversal freq frequency))
# whitening virgo
# lunghezza scritto 100+ pagine
# problemi col training
# in-memory
# .h5 file extension
# soglia di selezione fatta in base alla rilevazione sui segnali iniettati su tutti gli spettri indiscriminatamente (ottimizzare la ripulitura dei dati)

# NOTE e TODO:
# tra i miei 10 minuti di training e la settimana di traning scritta nell'articolo c'è una differenza esattamente di un fattore mille
# poi il numero di kernel andrà ottimizzato guardando le immagini generate massimizzando il gradiente, in modo da essere sicuri di non star levando spazio a features rilevanti (come ad esempio tutte le varie combinazioni di colore)
# vedere se si riesce ad andare sotto la soglia della peakmap, nel qual caso si può alaborare una strategia per analizzare anche i segnali continui
# nel futuro fa estrarre alla rete anche i parametri del segnale (o farlo fare ad una rete dedicata, a valle di una pulizia ottimale del segnale)
# nel futuro far fare la selezione degli spettri direttamente ad un sistema automatico
# l'eventuale dropout iniziale di fatto gioca il ruolo di data augmentation
# in futuro farlo direttamente con gli streaming di dati che escono dall'interferometro
# ogni tanto il training non ingrana per niente e bisogna spegnere e ricominciare da capo
# valutare max_pooling VS average_pooling
# studiare local_response_normalization

class EarlyStoppingCallback(tflearn.callbacks.Callback):
    def on_epoch_end(self, training_state):
        if training_state.acc_value > 0.9999: # when the training accuracy is 1 the model cannot learn further
            print('train accuracy is 1')
            raise StopIteration
        # TODO early stopping automatico per evitare l'overfitting
        #loss_ratio = training_state.loss_value / training_state.val_loss
        #accuracy_ratio = training_state.acc_value / training_state.val_acc
        #print(loss_ratio, accuracy_ratio)

# signal amplitude: 32, 16, 8, 4, 2, 1
# epoche: 20+5+5+5+10+
# learning rate: 0.001, 0.0001 # andare a passi di fattore 2
# TODO serve trovare un modo per aggirare il plateau iniziale (magari con un'inizializzazione furba)
# TODO mettere leggeri dropout, per avvicinare train e test
# TODO valutare se partire direttamente da 16
# TODO fare checkpoints e prendere quello con validation accuracy più alta (per fare early stopping)
signal_amplitudes = [32, 16, 8, 4, 2, 1] # descending order
biggest_amplitude = max(signal_amplitudes)
smallest_amplitude = min(signal_amplitudes)

validation = True

# TODO poi rimettere 30+10+10 epoche
number_of_epochs = 50
# TODO fare un altro shuffle preliminare per garanzia sul validation set

if not validation:
    # load first weights
    #model.load('/storage/users/Muciaccia/models/pretraining_amplitude_{}.tflearn'.format(32))
    # curriculum learning
    for amplitude in signal_amplitudes:
        print('signal amplitude:', amplitude)
#        if amplitude is not biggest_amplitude:
#            # load pretrained weights (to start closer to the minimum)
#            model.load('/storage/users/Muciaccia/models/pretraining_amplitude_{}.tflearn'.format(amplitude))
        
        # TODO mettere un if sull'attributo signal_intensity (ereditato dai dataset) per implementare il curriculum learning
        
        # load data
        dataset = xarray.open_dataset('/storage/users/Muciaccia/data/train/signal_amplitude_{}.netCDF4'.format(amplitude))
        train_images, test_images, train_classes, test_classes = sklearn.model_selection.train_test_split(dataset.images, dataset.classes, test_size=0.5, shuffle=True)
        
        # training
        try:
            model.fit({'input':train_images}, {'target':train_classes}, n_epoch=number_of_epochs, validation_set=({'input':test_images}, {'target':test_classes}), snapshot_step=100, show_metric=True, callbacks=EarlyStoppingCallback()) # run_id='tflearn_conv_net_trial'
        except StopIteration:
            print('training finished!')

        # save the model
        model.save('/storage/users/Muciaccia/models/pretraining_amplitude_{}.tflearn'.format(amplitude))
        # TODO save (append) the training history
    exit()

if validation:
    # load latest weights
    model.load('/storage/users/Muciaccia/models/pretraining_amplitude_{}.tflearn'.format(smallest_amplitude))





# tempo pretraining con segnale a 10: 4 minuti, <30 epoche con 2500 immagini di train
# tempo pretraining con segnale a 5: 1 minuto, <10 epoche
# tempo pretraining con segnale a 1: 2 minuti, <15 epoche

# TODO:
# predizioni, grafici loss e accuracy/error, confusion_matrix
# istogrammi di classificazione (con signoide finale)
# veto sui dati iniziali
# iniezione distribuita in chunck per l'out-of-memory
# iniezione di segnali veri con fft (e loro istogramma per capirne il livello)
# script curriculum learning automatizzato (vedere model.load)
# calcolo complessivo SNR con l'energia
# generare corretta forma d'onda
# log10 nei grafici
# spettro autoregressivo nel grafico di whitening
# mediana, 50%, 90%, 100% nel grafico dello spettro completo
# valutare vari livelli di accuracy col classificatore finale
# cominciare a scrivere!
# leggere articolo
# documentarsi su curve ROC e su local_response_normalization
# script per fare 100 training e poter fare un grafico





#model.evaluate(test_images[0:1024],test_classes[0:1024], batch_size=64)
#
#tf.confusion_matrix(labels=test_classes[0:1024,1], predictions=model.predict(test_images[0:1024])[0:1024,1]).eval()
#
## TODO BUG: non sembra possibile avere i pesi di tflearn caricati nel grafo di tensorflow
## TODO BUG: tflearn non supporta il model.predict out-of-memory, costringendo a caricare tutto il dataset in memoria
## TODO BUG: ci vorrebbe il parametro batch_size pure per model.predict()
#model.predict
#model.net # sembra avere i pesi non inizializzati persino dopo model.load
#
#model.net.eval(feed_dict={'input/X:0':test_images[0:64]})
#
#model.predict(test_images[0:64])
#
#sess = tf.InteractiveSession()
#sess.run(tf.global_variables_initializer())
#
#network.eval(feed_dict={'input_1/X:0':test_images[0:64]})




# validation

import numpy
import sklearn.metrics

# TODO hack simil out-of-memory
# TODO sulla CPU del mio laptop la predizione di un intero dataset di 5120 immagini prende un minuto circa usando tutti i core
def predict_in_chunks(images):
    predictions = []
    chunk_size = 256
    number_of_chunks = len(images)/chunk_size
    chunks = numpy.split(images, number_of_chunks)
    for chunk in chunks:
        predictions.append(model.predict(chunk[:,:,0:128,:])) # TODO HACK
        # the last activation function of the network is a softmax
        # so the predictions are the probability for the various classes
        # they sum up to 1
    predictions = numpy.concatenate(predictions)
    return predictions

def compute_metrics(dataset):
    predictions = predict_in_chunks(dataset.images)
    predicted_signal_probabilities = predictions[:,1]
    predicted_classes = numpy.round(predicted_signal_probabilities) # TODO vedere dall'istogramma dove deve essere messa la soglia ottimale
    true_classes = dataset.classes[:,1]
    confusion_matrix = sklearn.metrics.confusion_matrix(true_classes, predicted_classes)
    [[true_negatives,false_positives],[false_negatives,true_positives]] = [[predicted_0_true_0,predicted_1_true_0],[predicted_0_true_1,predicted_1_true_1]] = confusion_matrix
    # NOTA: fino a segnale 2 i falsi negativi sono pochissimi ed invece i falsi positivi sono parecchi
    # serve fare un trainig a 2
    # serve fare validation più fitta, coi mezzi punti
    purity = true_positives/(true_positives + false_positives) # precision
    efficiency = true_positives/(true_positives + false_negatives) # recall
    accuracy = (true_positives + true_negatives)/(true_positives + false_positives + true_negatives + false_negatives)
    # normalizations
    all_real_signals = true_positives + false_negatives
    all_real_noises = true_negatives + false_positives
    all_predicted_as_signals = true_positives + false_positives
    all_predicted_as_noise = true_negatives + false_negatives
    all_validation_samples = true_positives + false_positives + true_negatives + false_negatives
    # TODO fare qui istogramma della separazione tra le due classi
    # con predicted_signal_probabilities e true_classes
    return [dataset.signal_intensity, all_validation_samples, 100*true_negatives/all_real_noises, 100*false_positives/all_real_noises, 100*false_negatives/all_real_signals, 100*true_positives/all_real_signals, 100*purity, 100*efficiency, 100*accuracy]
    # TODO mettere solo due cifre decimali

import glob

validation_paths = glob.glob('/storage/users/Muciaccia/data/validation/**/*.netCDF4', recursive=True) # TODO 10 risulta venire dopo 1 e non dopo 9

metrics = []
for path in validation_paths:
    validation_dataset = xarray.open_dataset(path)
    metrics.append(compute_metrics(validation_dataset))

# TODO salvare i risultati su file
# TODO fare vari plot dei risultati

# TODO considerare il fatto che le immagini sono troncate da 148 a 128

# TODO magari fare anche la validation col rumore bianco senza buchi (e con la vera ampiezza relativa del regnale)

import pandas

# TODO 'false alarms' è brutto e poco chiaro. mettere qualcosa tipo 'misclassified noise'
metrics = pandas.DataFrame(metrics, columns=['signal_intensity', 'all_validation_samples', 'rejected noise (%)', 'false alarms (%)', 'missed signals (%)', 'selected signals (%)', 'purity (%)', 'efficiency (%)', 'accuracy (%)'])
metrics.sort_values(by='signal_intensity', inplace=True)

print(metrics)

# TODO serve del reinforcement learning per evitare tutti questi falsi positivi?
# TODO far fare il primo addestramento con rumore bianco? (oppure mischiare i dataset di rumore bianco e rumore vero coi buchi)

#  signal_intensity  true_negatives  false_positives  false_negatives  \
#                 1            1159              104              112   
#                 2            1180              104                3   
#                 3            1153              105                1   
#                 4            1151              100                0   
#                 5            1153              100                0   
#                 6            1215              108                0   
#                 7            1168              107                0   
#                 8            1180              112                0   
#                 9            1199              104                0   
#                10            1216               90                0   
#
#  true_positives    purity  efficiency  
#            1185  0.919317    0.913647  
#            1273  0.924473    0.997649  
#            1301  0.925320    0.999232  
#            1309  0.929028    1.000000  
#            1307  0.928927    1.000000  
#            1237  0.919703    1.000000  
#            1285  0.923132    1.000000  
#            1268  0.918841    1.000000  
#            1257  0.923586    1.000000  
#            1254  0.933036    1.000000  


#   signal_intensity  true_negatives  false_positives  false_negatives  \
#                 1            5077               20             1048   
#                 2            5116               19                9   
#                 4            5151               12                0   
#
#   true_positives    purity  efficiency  accuracy  
#            4095  0.995140    0.796228  0.895703  
#            5096  0.996285    0.998237  0.997266  
#            5077  0.997642    1.000000  0.998828  




