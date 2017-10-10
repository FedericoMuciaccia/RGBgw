
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


import glob
import xarray
import numpy
import scipy.io
import scipy.signal

def to_whitened(img):
    # from 0-1 interval to the whole R^+
    return numpy.exp(img*10 - 7) # TODO generalizzare (qui l'immagine originale (in scala logaritmica naturale) era tra -7 e +3)

def to_dense_peakmap(grayscale_image):
    # to search for local maxima (along frequency dimension), we search for derivatives with alternating sign
    signs_of_vertical_differences = numpy.sign(numpy.diff(grayscale_image, axis=0))
    
    # TODO sostituire la convoluzione con un secondo diff fatto sui segni, in modo che la somma di +1 e -1 faccia zero (controllare che la peakmap in output sia identica)
    kernel = [[-1],[1]]
    convolutions = scipy.signal.convolve2d(signs_of_vertical_differences, kernel, mode='valid')
    #convolutions = numpy.array([numpy.convolve(signs_of_vertical_differences[:,i], kernel, mode='valid') for i in range(grayscale_image.shape[1])])
    # TODO parallelizare
    # TODO attenzione ai bordi
    
    threshold = 2.5 # they keep only >= 2.5
    under_threshold = grayscale_image < threshold
    
    maxima = numpy.equal(convolutions, 2)
    # TODO sarebbe forse più corretto prendere i massimi locali lungo la direzione dello spindown, invece che in quella orizzontale della finestra
    peakmap = numpy.pad(maxima, pad_width=[[1,1],[0,0]], mode='constant')
    peakmap[under_threshold] = 0
    return peakmap

#def to_sparse(dense_peakmap):
#    frequencies, times = numpy.nonzero(dense_peakmap)
#    return dict(time=times, frequency=frequencies)

def create_sparse_peakmap(file_path):
    dataset = xarray.open_dataset(file_path)
    
    whitened_spectrogram = to_whitened(dataset.images) # without log: values in R^+
    # TODO farlo a monte, prima di comprimere nell'intervalo 0-1
    
    H = whitened_spectrogram.sel(channels='red')
    L = whitened_spectrogram.sel(channels='green')
    V = whitened_spectrogram.sel(channels='blue')

    file_name = 'amplitude_{}.mat'.format(dataset.signal_intensity)
    # TODO capire perché i bool vengono salvati come uint8
    # TODO salvare in formato compresso (h5?)
    scipy.io.savemat('/storage/users/Muciaccia/data/validation/dense_peakmaps/'+file_name, # TODO salvare in h5
                     mdict={'H':[to_dense_peakmap(image) for image in H], # TODO vettorializzare per bene con le convoluzioni di TensorFlow
                            'L':[to_dense_peakmap(image) for image in L],
                            'V':[to_dense_peakmap(image) for image in V],
                            'classes':dataset.classes[:,1].astype(numpy.uint8),
                            'signal_amplitude':dataset.signal_intensity})

file_list = glob.glob('/storage/users/Muciaccia/data/validation/*.netCDF4', recursive=True)

for file_path in file_list:
    create_sparse_peakmap(file_path)

# TODO LENTISSIMO PERCHÉ FA TUTTI I FOR DIRETTAMENTE IN PYTHON E IN SINGLE CORE
# TODO ha senso salvare le peakmap come sparse per salvare un poco di spazio su disco MA ritrovarsi a non poter vettorializzare il calcolo (perché le righe non sono tutte di lunghezza uguale). se si salvassero le peakmap direttamente dense, ma col nuovo .mat (>7.3) che è un h5 e dunque supporta la coversione? si arriverebbe orientativamente allo stesso spazio occupato, ma con la possibilità di vettorializzare e parallelizzare tutto il calcolo?
# TODO quando le peakmap sono state inventate, più di 15 anni fa, il collo di bottiglia era lo spazio su disco, perché gli HDD erano piccoli e costosi

# loro peakmap: circa 1.5 MB per 1 mese per 1 Hz
# dunque io dovrei occupare 1.5 MB * 1/5 mese * 40 Hz * 3 detector * 8 dataset = 288 MB

