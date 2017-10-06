
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


import numpy

from matplotlib import pyplot

import tensorflow as tf

session = tf.InteractiveSession()

second = 1
minute = 60*second
hour = 60*minute
day = 24*hour

signal_time_start = 2*day
signal_duration = 2*day
signal_time_stop = signal_time_start + signal_duration

signal_starting_frequency = 96 #32 #10 #90 # Hz
# spindown = df/dt
random_multiplier = 9*numpy.random.rand()+1 # uniform from 1 to 10
signal_spindown = random_multiplier*-1e-9 # uniform from -10^-9 (small) to -10^-8 (big)

#delta_frequency = -0.005 # Hz
#delta_frequency/signal_duration# binning_spindown = delta_f/t_durata_segnale

time_sampling_rate = 256 # Hz # subsampled from 4096 Hz data
time_resolution = 1/time_sampling_rate # s # time-domain time binning

image_time_start = 0.0
image_time_interval = 2**19 # 2^19 = 524288 seconds (little more than 6 days)
image_time_stop = image_time_start + image_time_interval

t = numpy.arange(start=image_time_start, stop=image_time_stop, step=time_resolution, dtype=numpy.float64)
# NOTE: float64 is needed to guarantee enough time resolution

#t = tf.linspace(start=image_time_start, stop=image_time_stop, num=image_time_interval*time_sampling_rate + 1) # last value included

# TODO trovare il valore corretto usando a ritroso la distribuzione chi quadro con due gradi di libertà
noise_amplitude = 1.5e-5 # deve dare 1e-6 # TODO check normalizzazione
#white_noise = noise_amplitude*numpy.random.randn(t.size).astype(numpy.float32) # gaussian noise around 0
white_noise = tf.random_normal(shape=t.shape, mean=0, stddev=noise_amplitude).eval() # float32
# tensorflow is way faster than numpy at generating random numbers
#real_part, imaginary_part = noise_amplitude*numpy.random.randn(2,len(t)).astype(numpy.float32)
#white_noise = real_part + 1j*imaginary_part
#real_part = tf.random_normal(shape=t.shape, mean=0, stddev=noise_amplitude) # dtype=float32
#imaginary_part = tf.random_normal(shape=t.shape, mean=0, stddev=noise_amplitude) # dtype=float32
#white_noise = tf.complex(real_part, imaginary_part) # dtype=complex64

# TODO BUG: tf.random_normal mu and sigma
# TODO BUG: range(start=0, stop, step=2) VS ordered dictionary
# TODO VEDERE tf.shape di array 1D e di scalari e obblico di mettere sempre le parentesi quadre negli argomenti shape delle funzioni e dimensioni con label/nome tipo xarray e pandas

signal_scale_factor = 0.004 #0.1 #0.005 #0.1
signal_amplitude = signal_scale_factor*noise_amplitude
# a 0.005 il segnale è ancora ben visibile
# a 0.004 limite di accettabile visibilità: non si vede benissimo dove finisce
# a 0.003 è dove entra la soglia della peakmap e dove si comincia a intravedere qualcosa (ma non si riesce affatto ad essere sicuri che sia effettivamente un segnale)
# a 0.002 completamente invisibile (e sotto soglia nella peakmap)
# NOTA: tutte le precedenti considerazioni valgono per immagini bianconero. TODO poi andrà fatta anche la prova col rumore bianco ed i 3 detector in contemporanea (immagine RGB senza buchi)

make_plot = True

def signal_waveform(t):
    # signal = exp(i phi(t))
    # phi(t) = integrate_0^t{omega(tau) d_tau}
    # omega = 2*pi*frequency
    # df/dt = s # linear (first order) spindown
    # f(t) = f_0 + s*t
    # ==> phi(t) = integrate_0^t{2*pi*(f_0+s*tau) d_tau}
    # ==> phi(t) = 2*pi*(f_0*t + (1/2)*s*t^2 + C) # TODO capire perché è necessario mettere 'modulo 2 pi'
    #signal = signal_amplitude*numpy.sin(2*numpy.pi*signal_starting_frequency*t).astype(numpy.float32) # pure sinusoid # TODO CAPIRE perché la versione con la sola sinusoide pura è micidialmente lenta
    signal = signal_amplitude*numpy.sin(2*numpy.pi*(signal_starting_frequency + (1/2)*signal_spindown*t)*t).astype(numpy.float32) # sinusoid with spindown
    # TODO usando invece l'esponenziale complesso serve mettere 'modulo 2 pi'
    #signal = signal_amplitude*numpy.sin(2*numpy.pi*(signal_starting_frequency*t))
    return signal

signal = signal_waveform(t) # TODO abbastana lento

# SNR in un solo chunk di dati ? # ratio=0.1 ancora ben visibile # rapporto critico = (amp segnale - media del rumore)/deviazione standard = 1 # poi scaling con CR*sqrt(N_FFT) # N_FFT = numero di chunk temporali # rapporto critico nel piano della Hough (coi conteggi in quel piano) # calcolare quanta è l'energia trasportata dal segnale (integrale dello spettro di potenza?) (VS valore di picco della sinusoide) # energia totale (integrale) ceduta dal segnale nel rivelatore # vs SNS_su_singola_FFT # l'ampiezza della FFT diminuisce col tempo, perché ci sono meno cicli nel tempo dato che la frequenza diminuisce # procedura completamente coerente, con un'unica FFT (coerente e incoerente, con o senza sqrt(durata segnale OR t_osservazione) (vedere articolo Explorer) # signal power density
#signal = signal_amplitude*tf.exp(tf.complex(1.0,2*numpy.pi*(signal_starting_frequency - signal_spindown*t)*t))

# signal temporal truncation
# TODO dovrebbero essere evitate le discontinuità, che non vengono decomposte bene nella base di Fourier, quindi fare il taglio quando la sinusoide passa per 0
# TODO probabilmente è molto più efficiente far calcolare la sinusoide solo sui tempi troncati
signal[numpy.logical_or(t < signal_time_start, t > signal_time_stop)] = 0
# TODO farlo con le slice su xarray

#pyplot.figure(figsize=[15,10])
#pyplot.plot(white_noise)
#pyplot.plot(signal)
#pyplot.show()

signal_start_index = signal_time_start*time_sampling_rate
time_slice = range(signal_start_index-256, signal_start_index+256)

if make_plot is True:
    pyplot.figure(figsize=[15,10])
    pyplot.title('injected signal (displayed 50x) and gaussian white noise\ncritical ratio = {}'.format(signal_scale_factor))
    pyplot.plot(white_noise[time_slice]) # TODO senza logy?
    pyplot.plot(50*signal[time_slice])
    pyplot.xticks([0,256,512],[signal_time_start-1,signal_time_start,signal_time_start+1]) # TODO hardcoded
    #pyplot.show()
    pyplot.xlabel('time [s]')
    pyplot.savefig('/storage/users/Muciaccia/media/white_noise_and_injected_signal.svg', dpi=300)
    pyplot.close()
    # NOTE: the y scale is linear: the noise is gaussian around 0

#pyplot.figure(figsize=[15,10])
#pyplot.plot(white_noise+signal)
#pyplot.show()

FFT_lenght = 8192 # s # frequency-domain time binning
Nyquist_frequency = time_sampling_rate/2 # 128 Hz
number_of_time_values_in_one_FFT = FFT_lenght*time_sampling_rate
unilateral_frequencies = numpy.linspace(0, Nyquist_frequency, int(number_of_time_values_in_one_FFT/2 + 1)) # TODO float32 or float64 ?
frequency_resolution = 1/FFT_lenght

# TODO per risparmiare memoria, limitare il calcolo alla sola sottobanda interessante (tra 80 e 120 Hz)
# TODO poi gererare pure immagini RGB

#pyplot.figure(figsize=[15,10])
#pyplot.hist2d(t, white_noise, bins=[100,number_of_chunks])
#pyplot.show()

#pyplot.figure(figsize=[15,10])
#pyplot.hist2d(t, white_noise+signal, bins=[100,number_of_chunks])
#pyplot.show()

def flat_top_cosine_edge_window(window_lenght = number_of_time_values_in_one_FFT):
    # 'flat top cosine edge' window function (by Sergio Frasca)
    # structure: [ascending_cosine, flat, flat, descending_cosine]

    half_lenght = int(window_lenght/2)
    quarter_lenght = int(window_lenght/4)
    
    index = numpy.arange(window_lenght)
    
    # TODO valutare la Hamming window e/o quella cos^2 (in modo che interallacciata sommi sempre a 1)
            
    # sinusoidal part at the edges
    factor = 0.5 - 0.5*numpy.cos(2*numpy.pi*index/half_lenght)
    # flat part in the middle
    factor[quarter_lenght:window_lenght-quarter_lenght] = 1
    
    # TODO attenzione all'ultimo valore:
    # factor[8191] non è 0
    # (perché dovrebbe esserlo invece factor[8192], ma che è fuori range)
    
    # calcolo delle normalizzazione necessaria per tenere in conto della potenza persa nella finestra
    # area sotto la curva diviso area del rettangolo totale
    # se facciamo una operazione di scala (tanto il rapporto è invariante) si capisce meglio
    # rettangolo: [x da 0 a 2*pi, y da 0 a 1]
    # area sotto il seno equivalente all'integrale del seno da 0 a pi
    # integrate_0^pi sin(x) dx = -cos(x)|^pi_0 = 2
    # area sotto il flat top: pi*1
    # dunque area totale sotto la finestra = 2+pi
    # area del rettangolo complessivo = 2*pi*1
    # potenza persa (rapporto) = (2+pi)/2*pi = 1/pi + 1/2 = 0.818310
    # fattore di riscalamento = 1/potenza_persa = 1.222031
    # TODO questo cacolo è corretto? nel loro codice sembra esserci un integrale numerico sui quadrati
    # caso coi quadrati:
    # integrate sin^2 from 0 to pi = x/2 - (1/4)*sin(2*x) |^pi_0 = 
    # = pi/2
    # dunque (pi/2 + pi)/2*pi = 3/4
    
    return factor.astype(numpy.float32)

# TODO duplicato
window = flat_top_cosine_edge_window()

if make_plot is True:
    pyplot.figure(figsize=[15,10])
    pyplot.plot(window)
    pyplot.xticks(number_of_time_values_in_one_FFT*numpy.arange(5)/4, (FFT_lenght*numpy.arange(5)/4).astype(int))
    pyplot.xlabel('time [s]')
    pyplot.title('flat top cosine edge window')
    #pyplot.show()
    pyplot.savefig('/storage/users/Muciaccia/media/flat_top_cosine_edge_window.svg', dpi=300)
    pyplot.close()

def make_chunks(time_data, windowed = False):
    number_of_chunks = int(len(t)/number_of_time_values_in_one_FFT)
    chunks = numpy.split(time_data, number_of_chunks)
    time_shift = int(number_of_time_values_in_one_FFT/2)
    # TODO ottimizzare il codice e magari farlo con una funzione rolling
    middle_chunks = numpy.split(time_data[time_shift:-time_shift], number_of_chunks-1)
    middle_chunks.append(numpy.zeros_like(chunks[0])) # dummy empty chunk to be removed later # TODO far in modo che tutto sia sempre in potenze di 2
    # join and reorder odd and even chunks
    interlaced_chunks = numpy.transpose([chunks, middle_chunks], axes=[1,0,2]).reshape([2*number_of_chunks, -1])
    # TODO buttare l'ultimo chunk farlocco
    if windowed is False:
        return interlaced_chunks
    if windowed is True:
        window = flat_top_cosine_edge_window()
        windowed_interlaced_chunks = interlaced_chunks*window
        return windowed_interlaced_chunks

time_data = white_noise+signal

#whos # per vedere tutte le variabili definite, residenti in memoria

#index = numpy.arange(8192)
#factor = index.copy()
#factor[4096:8192] = 8192-index[4096:8192]
#noise = numpy.random.randn(8192)
#window = factor+noise
#
#fft = numpy.fft.rfft(window)
#dBV = 20*numpy.log(numpy.sqrt(numpy.square(numpy.abs(fft)))/8192)
#
#a = numpy.log(numpy.abs(fft))
#
#pyplot.figure(figsize=[15,10])
#pyplot.plot(window)
#pyplot.show()
#
#pyplot.figure(figsize=[15,10])
#pyplot.plot(dBV)
#pyplot.show()

# NOTA:
# con e senza finestra, spindown 1e-9, segnale 0.1
# picco 10^4, base 10^0
# con segnale molto basso (0.004) l'immagine risulta meglio visibile con la finestra (!)

def make_whitened_spectrogram(time_data): # the fast Fourier transform needs power on two to be fast
    windowed_interlaced_chunks = make_chunks(time_data, windowed = True)
    unilateral_fft_data = numpy.fft.rfft(windowed_interlaced_chunks).astype(numpy.complex64) # the one-dimensional real fft is done on the innermost dimension of the array. the results are the unilateral frequencies, from 0 to the Nyquist frequency
#unilateral_fft_data = numpy.fft.rfftn(windowed_interlaced_chunks, axes=[-1])
    #unilateral_fft_data = list(map(numpy.fft.rfft, windowed_interlaced_chunks))
    #unilateral_fft_data.pop()
    #unilateral_fft_data = numpy.array(unilateral_fft_data).astype(numpy.complex64)
    # TODO vedere normalizzazione per la potenza persa
    spectra = numpy.square(numpy.abs(unilateral_fft_data)) # TODO sqrt(2), normd, normw, etc etc
    # TODO normd (normalizzare sul numero di dati)
    spectrogram = numpy.transpose(spectra[0:-1]) # remove the last dummy empty chunk
    whitened_spectrogram = spectrogram/numpy.median(spectrogram)
    # TODO in realtà facendo il whitening col periodogramma si eliminano tutti quei picchi che occupano diversi bin di frequenza, dunque si ripulirebbe l'effetto di allargamento sporco dovuto alla finestra della FFT
    # TODO fare plot zoomato attorno alla riga, in modo da vedere i ghost laterali prima e dopo lo sbiancamento
    return whitened_spectrogram # TODO valutare se inserire un dato fittizio per arrivare ad una potenza di 2

#####################################

# TODO BUG: tensorflow non ha una versione di rfft che giri su CPU, quindi il codice non è portabile su tutti i dispositivi
# TODO VEDERE tf.batch_fft2d
# TODO VEDERE tf.contrib.signal ha delle routine per calcolare gli spettrogrammi
# TODO BUG: numpy ha il grossissimo bug che computa molte cose (tipo la fft) in float64 e poi si deve convertire in float32, apendo però nel mentre occupato temporaneamente il doppio della RAM
# TODO cercare un modo per forzare la computazione di numpy ad essere sempre fatta in float32

whitened_spectrogram = make_whitened_spectrogram(time_data)

if make_plot is True:
    pyplot.figure(figsize=[15,10])
    spectrum_with_signal_example = whitened_spectrogram[:,60]
    #spectrum_median = numpy.median(spectrum_with_signal_example) # deve fare circa 1e-6
    #pyplot.hlines(y=spectrum_median, xmin=0, xmax=unilateral_frequencies.size, color='black', label='spectrum median = {}'.format(spectrum_median), zorder=1) # TODO invece che la mediana plottare lo spettro autoregressivo
    pyplot.semilogy(spectrum_with_signal_example, label='whitened spectrum', zorder=0) # grafico obbligatoriamente col logaritmo in y
    pyplot.title('frequency spectrum')
    pyplot.xticks((unilateral_frequencies.size*numpy.arange(5)/4).astype(int), (Nyquist_frequency*numpy.arange(5)/4).astype(int))
    #pyplot.vlines(x=8192, ymin=1e-9, ymax=1e1, color='orange', label='1 Hz')
    pyplot.xlabel('frequency [Hz]')
    #pyplot.legend(loc='lower right', frameon=False)
    #pyplot.show()
    pyplot.savefig('/storage/users/Muciaccia/media/white_noise_complete_whitened_spectrum_with_injected_signal.jpg')#, dpi=300)
    pyplot.close()
    # TODO WISHLIST: spectrogram[all,1]

########################################
#mail Ricci per settore informatico
#Ubuntu per mia GPU
########################################
#
#la function per l'autoregressivo si chiama int short_psar_rev_freq
#
#il periodogramma invece e’ standard…dividi i dati i sottoperiodi ne fai lo spettro e poi la media.
#
#matched filtering for delta-like signals
#identification of the events
#removal of the events
#
#media non parametrica fatta con spline? (valore del polinomio fatto in modo da ottimizzare il fit senza overfitting ed underfitting, magari con valore anche variabile nel tempo?)
#regolarizzare il rumore non-stazionario la cui forma non è nota a priori
#poi fare il whitening rispetto a questa curva (considerando anche la varianza variabile nel tempo del segnale iniziale?)
#fare un operazione direttamente sui dati grezzi nel dominio del tempo?
#invece che fare un semplice downsampling dei dati iniziali, si può fare uno smooting gaussiano o un qualche tipo di regolarizzazione per ridurre le fluttuazioni che non serve osservare?
########################################

# TODO con la fft unilatera le frequenze non sono più divisibili in base 2 (e pure i tempi adesso sono 127)

# power_spectrum = FFT autocorrelazione
# power_spectrum != modulo quadro dell'FFT (serve un fattore di normalizzazione)
# normd circa 10^-5 o 10^-6
# dati reali con livello a 10^-23
# simulare rumore bianco in frequenza direttamente con una gaussiana di larghezza 1/sigma # TODO corretto? grafici lin VS logy? si intendono le due gaussiane reale e immaginaria dell'esito della FFT?
# finestra flat_coseno per minimizzare l'allargamento di segnali che variano un poco in frequenza (per smussare i bordi). e dunque c'è poi la necessità di buttare i bordi mediante le FFT interallacciate (per ripassare poi nel dominio del tempo). (minimizzare i ghost laterali della delta di Dirac allargata della sinusoide e/o massimizzare l'altezza del picco). poi usare normw per tenere in conto della potenza persa ai bordi della finestra rispetto alla funzione gradino (fattore comuque vicino ad 1). tutti fattori da rimoltiplicare per controbilanciare la perdita di potenza spettrale

#pyplot.figure(figsize=[15,10])
#pyplot.hist(numpy.log10(spectrogram[image_range].flatten()), bins=300) # log10
#pyplot.show()
## TODO valutarre la sovrapponibilità coi dati veri (dopo aver correttamnete normalizzato tutto)

frequency_values_in_one_Hz = 8192 # TODO hardcoded
middle_index = frequency_values_in_one_Hz*signal_starting_frequency
frequency_range = slice(middle_index-128,middle_index+128)

image = whitened_spectrogram[frequency_range]

if make_plot is True:
    pyplot.figure(figsize=[15,10])
    pyplot.hist(numpy.log10(image.flatten()), bins=300)
    pyplot.vlines(x=numpy.log10(2.5), ymin=0, ymax=900, color='orange', label='peakmap threshold = 2.5')
    linear_ticks = numpy.linspace(-5, 5, num=11)
    log_labels = ['10^{}'.format(int(i)) for i in linear_ticks]
    pyplot.xticks(linear_ticks, log_labels)
    pyplot.xlabel('whitened ratio')
    pyplot.ylabel('counting')
    pyplot.title('histogram of pixel values in 1 image')
    pyplot.legend(frameon=False)
    pyplot.savefig('/storage/users/Muciaccia/media/histogram_of_whitened_white_noise_with_injected_signal.svg', dpi=300)
    #pyplot.show()
    pyplot.close()

# TODO sovraimporre nel grafico la distribuzione teorica (chi quadro a due gradi di libertà)

# NOTE: FFT (Fast Fourier Transform) refers to a way the discrete Fourier Transform (DFT) can be calculated efficiently, by using symmetries in the calculated terms. The symmetry is highest when `n` is a power of 2, and the transform is therefore most efficient for these sizes.

pyplot.figure(figsize=[10,10*256/148])
pyplot.imshow(numpy.log10(image), origin="lower", interpolation="none", cmap='gray')
pyplot.title('whitened spectrogram (log10 values)')
# TODO mettere linea a time_index = 60 per far capire dove si posiziona il plot successivo
#pyplot.colorbar()
#pyplot.show()
pyplot.savefig('/storage/users/Muciaccia/media/white_noise_image_with_signal_scale_factor_{}.jpg'.format(signal_scale_factor), dpi=300)
pyplot.close()

if make_plot is True:
    pyplot.figure(figsize=[15,10])
    vertical_slice = image[:,60]
    pyplot.semilogy(vertical_slice, label='raw spectrum')
    pyplot.title('vertical slice of the image')
    pyplot.xlabel('frequency index')
    pyplot.ylabel('pixel value')
    #pyplot.show()
    pyplot.savefig('/storage/users/Muciaccia/media/vertical_slice.svg', dpi=300)
    pyplot.close()
# TODO fare deconvoluzione per accentuare e stringere il picco?

# TODO la derivata prima dovrebbe essere più rumorosa dei dati originali, quindi ci dovrebbe essere un modo per migliorare la peakmap facendo una derivata multiscala, con kernel sempre maggiori
def peakmap(image):
    copied_image = image.copy() # here without log10
    # TODO controllare i bordi
    maxima = numpy.array([numpy.convolve(numpy.sign(numpy.diff(copied_image, axis=0))[:,i], [-1,1]) for i in range(image.shape[1])]).T == 2
    # TODO sarebbe forse più corretto prendere i massimi locali lungo la direzione dello spindown, invece che in quella orizzontale della finestra
    under_threshold = copied_image < 2.5 # TODO provare poi a mettere la soglia a 2 per fare il confronto con la Hough per i segnali normalmente sottosoglia
    maxima[under_threshold] = 0
    return maxima.astype(int)

pyplot.figure(figsize=[10,10*256/148])
pyplot.imshow(peakmap(image), origin="lower", interpolation="none", cmap='gray_r')
pyplot.title('peakmap')
#pyplot.show()
pyplot.savefig('/storage/users/Muciaccia/media/white_noise_peakmap_with_signal_scale_factor_{}.jpg'.format(signal_scale_factor), dpi=300)
pyplot.close()
# NOTA: per segnali giganteschi appare correttamente la regione di svuotamento attorno al segnale



# NOTA: trattare sempre grandezze relative e pubblicare grafici e risultati con i quali tutti i diversi esperimenti possono confrontarsi indipendentemente

# domande Pia:
# forma d'onda con funzione analitica precisa
# sembrano più sinusoidi modulate
# ghosts dovuti alla finestra?
# Hamming window
# spindown a salire ed inversione di fase
# spindown lineare vs peakmap con onda continua ondeggiante
# livello ampiezza rumore bianco per sovrapponibilità coi dati veri
# fattore di normalizzazione per gli spettri (radice di 2 etc)
# unilatero e bilatero
# fft con sliding gaussiano invece che interallacciate
# FFT(real) = real ?
# campionamento originale 4096 Hz
# tipici valori di spindown
# SNR 1 vs 1 inverosimile. più ragionevole 1 vs 0.1
# varie unità di misura nei grafici

# fft reale bilatera con ghost (dividendo l'energia/potenza)
# fft complessa senza frequenze negative
# campionare a frequenza doppia
# i dati veri sono reali e non complessi
# iniettare solo in una piccola banda (nei complessi) per velocizzare il calcolo
# iniettare nel tempo per avere tutti gli artefatti
# ipoteticamente, dato che 128 pixel temporali corrispondono a circa 6 giorni, si potrebbe replicare l'analisi con segnali continui di 3 mesi, con 2048 pixel temporali (e ovviamente un batch-size piccolissimo)

# niente punti del cielo e linee divergenti per le varie correzioni Doppler e linee discontinue per i pattern d'antenna
# [punto_del_cielo, frequenza_base, vari_ordini_di_spindown]
# gd->y sono i dati (struttura)
# documento che descrive la forma della finestra e il suo perché
# normw = 2*sqrt(2/3) calcolato numerico o integrando simbolicamente
# pipeline: data_stream, trigger, denoiser, nonparametric_fit, parameter_extractor
# chiedere simulazione reale per i pattern delle linee divergenti che si vedrebbero (Doppler + pattern d'antenna)
# generare il segnale nel tempo con tensorflow
# data_generator/queue per i vari file in-memory con tensorflow
# data stream direttamete da LIGO, non dal CNAF
# running window
# generare segnali a parte e poi fare funzione add_signal([noise_image, signal]) che tenga conto dei buchi dell'immagine
# usare tensorflow e SparseTensor per generare segnali in blocco
# vedere vectorialization su tensorflow con tf.map_fn (BUG su numpy)
# serialization on tensorflow: define a queue -> define single preprocessing -> set a batch to preprocess
# fare istogrammi puliti con fft del segnale
# denoiser a cui si danno in pasto add_signal(rumore,segnale) e solo segnale (eventualemte senza i buchi) come target

# 
# randn su GPU
# out-of-core con massimo 8 GB di RAM (memoria GPU)








#factor = numpy.random.randn(8192)

#x = numpy.linspace(0, 2*numpy.pi, 8192)

#noise = 0.01*numpy.random.randn(8192)
#y1 = numpy.sin(20*x)
#y2 = 2*numpy.sin(40*x)
#y = y1 + y2 # + noise

#y = factor + noise

#pyplot.figure(figsize=[15,10])
##pyplot.plot(x, noise)
#pyplot.plot(x, y1)
#pyplot.plot(x, y2)
#pyplot.plot(x, y)
#pyplot.show()

#pyplot.figure(figsize=[15,10])
#pyplot.plot(y)
#pyplot.show()

#window_fft = numpy.real(numpy.fft.fft(y))
## TODO con fft (bilatera) fare attenzione a rimettere le due metà ordinate nello spettro e a centrare tutto correttamente
## TODO con rfft (unilatera) attenzione alle potenze di 2

#fft = numpy.fft.fft(y)
#dBV = 20*numpy.log(numpy.sqrt(numpy.square(numpy.abs(fft)))/8192)

#pyplot.figure(figsize=[15,10])
#pyplot.plot(dBV)
#pyplot.show()

#pyplot.figure(figsize=[15,10])
#pyplot.semilogy(x, window_fft)
#pyplot.show()

#pyplot.figure(figsize=[15,10])
#pyplot.plot(window_fft[0:50])
#pyplot.show()


# numpy.fft.rfft(...)
# When the DFT is computed for purely real input, the output is Hermitian-symmetric, i.e. the negative frequency terms are just the complex conjugates of the corresponding positive-frequency terms, and the negative-frequency terms are therefore redundant.  This function does not compute the negative frequency terms, and the length of the transformed axis of the output is therefore ``n//2 + 1``.
# When ``A = rfft(a)`` and fs is the sampling frequency, ``A[0]`` contains the zero-frequency term 0*fs, which is real due to Hermitian symmetry.
# If `n` is even, ``A[-1]`` contains the term representing both positive and negative Nyquist frequency (+fs/2 and -fs/2), and must also be purely real.
# If `n` is odd, there is no term at fs/2; ``A[-1]`` contains the largest positive frequency (fs/2*(n-1)/n), and is complex in the general case.
# If `n` is even, the length of the transformed axis is ``(n/2)+1``.
# If `n` is odd, the length is ``(n+1)/2``.
# Notice how the final element of the `fft` output is the complex conjugate of the second element, for real input. For `rfft`, this symmetry is exploited to compute only the non-negative frequency terms.

exit()

# dimostrazione dell'invisibilità sottosoglia (signal_amplitude = 1)

def to_whitened(img):                                                  
    return numpy.exp(img*10 - 7)

img = images[random_signal_index]

R = img[:,:,0]
G = img[:,:,1]
B = img[:,:,2]

pyplot.figure(figsize=[10,10*256/148])
pyplot.imshow(img, origin="lower", interpolation="none")
pyplot.title('RGB image')
pyplot.show()

pyplot.figure(figsize=[10,10*256/148])
pyplot.imshow(peakmap(to_whitened(R)), origin="lower", interpolation="none", cmap='gray_r')
pyplot.title('peakmap R')
pyplot.show()

pyplot.figure(figsize=[10,10*256/148])
pyplot.imshow(peakmap(to_whitened(G)), origin="lower", interpolation="none", cmap='gray_r')
pyplot.title('peakmap G')
pyplot.show()

pyplot.figure(figsize=[10,10*256/148])
pyplot.imshow(peakmap(to_whitened(B)), origin="lower", interpolation="none", cmap='gray_r')
pyplot.title('peakmap G')
pyplot.show()

# from dense to sparse

rows, columns = peakmap.nonzero() # int64
# TODO trasformare in frequenze e tempi, partendo dagli indici e moltiplicando per la precisione in frequenze e tempi
# TODO salvare in .mat
# TODO mettere t, f, weights, df, dt, f_min=80, f_max=120, t_start, t_end, velocity, ecc...
# TODO dare a Iuri 10 dataset da 4 grandi peakmap ognumo
# TODO assemblare tutte le paekmap
# TODO mettere in .h5 o .npy ?
# TODO duplicare i dati di VSR4 per avere più dati e poi aggiungere i dati veri di Virgo ad agosto. oppure spostare VSR4 a destra in modo da rosicchiare dati nella parte finale dove i due detector veri sono più stabili
# TODO valutare se fare 2 settimane ma con spindown più bassi, come li usano loro

# TODO VEDERE SE SPETTRO SBIANCATO È USABILE PURE SOTTO 80 Hz e sopra 120 Hz








