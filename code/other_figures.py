
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

# TODO script già incluso nella relazione

import numpy
import matplotlib
matplotlib.use('SVG')
from matplotlib import pyplot

matplotlib.rcParams.update({'font.size': 20}) # il default è 10 # TODO attenzione che fa l'override di tutti i settaggi precedenti

#pyplot.rc('text', usetex=True)

# ReLU
x = numpy.linspace(start=-2, stop=2, num=1000)
relu = numpy.maximum(x, 0)
#tanh = numpy.tanh(x)
#softplus = numpy.log(1+numpy.exp(x))

pyplot.figure(figsize=[10, 5])
pyplot.plot(x,relu)
#pyplot.plot(x,sigmoid, label=r'$sigmoid(x) = 1/(1+e^{-x})$')
#pyplot.plot(x,tanh, label=r'$tanh(x) = $')
#pyplot.plot(x,softplus, label=r'$softplus(x) = \log(1 + e^x)$')

pyplot.xlabel('x')
pyplot.ylabel('ReLU(x) = max(x,0)')
#pyplot.ylabel(r'$ReLU(x) = \max(x,0)$')
pyplot.xticks([-2,-1,0,1,2])
#pyplot.yticks([-2,-1,0,1,2])
pyplot.yticks([0,1,2])
#pyplot.legend(frameon=False)
pyplot.savefig('/storage/users/Muciaccia/media/ReLU.jpg', dpi=300, bbox_inches='tight')
#pyplot.show()

x = numpy.linspace(start=-10, stop=10, num=1000)
sigmoid = 1/(1+numpy.exp(-x))

pyplot.figure(figsize=[10, 5])
pyplot.plot(x,sigmoid)
pyplot.xlabel('x')
pyplot.ylabel('sigmoid(x) = 1/(1+exp(-x))')
pyplot.xticks([-10,-5,0,5,10])
pyplot.yticks([0,1])
pyplot.savefig('/storage/users/Muciaccia/media/sigmoid.jpg', dpi=300, bbox_inches='tight')
pyplot.close()



