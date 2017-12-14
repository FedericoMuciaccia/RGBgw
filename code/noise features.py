
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
import xarray
import matplotlib
matplotlib.use('SVG') # per poter girare lo script pure in remoto sul server, dove non c'è il server X
from matplotlib import pyplot

dataset = xarray.open_mfdataset('/storage/users/Muciaccia/netCDF4/O2/C01/128Hz/LIGO Hanford*.netCDF4')
# only the L channel is enough: we want grayscale images

# TODO provare a fare plot RGB o fare le peakmap combinate per confermare se ci sono altri segnali

# O2 noise features
# (hardware injections and big resonances)


# hardware injection "pulsar3"
# frequency = 108.8572 Hz

# whole O2 run view
time_slice = slice('2016-11-30','2017-08-25')
frequency_slice = slice(108.845,108.870)
pulsar3_whole_run = dataset.whitened_spectrogram.sel(frequency=frequency_slice, time=time_slice, detector='LIGO Hanford')

fig = pyplot.figure(figsize=[20,10])
numpy.log(pulsar3_whole_run).plot(vmin=-10, vmax=5, cmap='gray', extend='neither', cbar_kwargs=dict(shrink=0.5))
fig.autofmt_xdate() # rotate the labels of the time ticks
pyplot.savefig('/storage/users/Muciaccia/media/pulsar3_hardware_injection_whole_run.jpg', dpi=300, bbox_inches='tight')
pyplot.close()

# 5 days view
#time_slice = slice('2017-07-26','2017-07-31') # L
time_slice = slice('2017-02-09','2017-02-14') # H
#frequency_slice = slice(108.835,108.865)
frequency_slice = slice(108.840,108.870)
pulsar3_in_6_days = dataset.whitened_spectrogram.sel(frequency=frequency_slice, time=time_slice, detector='LIGO Hanford')

fig = pyplot.figure(figsize=[10,20])
numpy.log(pulsar3_in_6_days).plot(vmin=-10, vmax=5, cmap='gray', extend='neither', cbar_kwargs=dict(shrink=0.5))
fig.autofmt_xdate() # rotate the labels of the time ticks
pyplot.savefig('/storage/users/Muciaccia/media/pulsar3_hardware_injection_6_days.jpg', dpi=300)
pyplot.close()


# 100 Hz resonance line

# 5 days view
time_slice = slice('2017-07-26','2017-07-31')
#time_slice = slice('2017-02-01','2017-02-06') # TODO strane strutture
frequency_slice = slice(99.985,100.015)
resonance_line = dataset.whitened_spectrogram.sel(frequency=frequency_slice, time=time_slice, detector='LIGO Hanford')

fig = pyplot.figure(figsize=[10,20])
numpy.log(resonance_line).plot(vmin=-10, vmax=5, cmap='gray', extend='neither', cbar_kwargs=dict(shrink=0.5))
fig.autofmt_xdate() # rotate the labels of the time ticks
pyplot.savefig('/storage/users/Muciaccia/media/resonance_line_at_100Hz_6_days.jpg', dpi=300)
pyplot.close()



# NOTE:
# la Hough senza correzione Doppler non vede nulla (perché la modulazione annuale dà una sinusoide invece di una linea)
# non capisco perché la modulazione giornaliera in O2 si vede solo nella fase ascendente (può essere un problema di finestra in Fourier?)
# hardware injection della pulsar3 a 108.85
# la visibilità dipende dalla scala a cui si guarda
# capire se la pulsar3 si vede nella finestra di 6 giorni
# vedere se il whitening deprime il segnale allargato dal Doppler giornaliero
# il follow-up è gerarchico e iterativo, facendo la Hough e correggendo e poi facendo Fourier più lunghe per aver maggiore precisione in frequenza e ripetere tutto per avvicinarsi ed amplificare sempre di più
# vedere come si vede la pulsar3 senza il whitening su singolo detector
# vedere come si vede la pulsar3 in RGB (senza correzione Doppler)
# vedere la linea gigante a 100 Hz
# il rumore gioca un ruolo fondamentale nella visibilità delle hardware injections
# citare articolo dove c'è la figura della peakmap con la sinusoide gigantesca
# spiegare che coi transienti il SNR è radicalmente diverso che coi segnli continui
# il trigger gira SENZA la correzione Doppler (anche per ovvie ragioni di tempo di calcolo)
# Pia dice che il rate di falsi allarmi ottenuto dall'ultimo training è comunque grande
# si preferirebbe avere ottima efficienza anche a discapito della purezza, perché gestire molti falsi positivi è solo un problema di risorse di calcolo nel follow-up per uccidere tutti i non-segnali
# in pratica la mia analisi non Doppler-corretta deve competere con la loro Doppler-corretta
# righe con doppie corna e correzione Doppler sia di relatività speciale che generale?
# ci vorrebbe una finestra fft ottimizzata per il doppler giornaliero (e le veloci variazioni che questo comporta)
# rendere più pulito ed omogeneo lo spettrogramma deve essere la priorità per poi poter fare un buon riconscimento visuale


