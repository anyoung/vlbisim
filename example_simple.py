#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  example_simple.py
#  Jan 21, 2015 13:10:41 EST
#  Copyright 2015
#         
#  Andre Young <andre.young@cfa.harvard.edu>
#  Harvard-Smithsonian Center for Astrophysics
#  60 Garden Street, Cambridge
#  MA 02138
#  
#  Changelog:
#  	AY: Created 2015-01-21
"""
A simple example to illustrate the use of signals and blocks.

Let the signal of interest be a 1GHz sinusoidal signal. It is
received by two antennas (not included in model), with a slight 
delay in the propagation path towards one antenna, and a gain
offset in the response of the other antenna. The antenna signals
are combined, and then delayed. Gaussian noise signals are also
independently added at the antennas.

"""

# some useful libraries to import
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

# import SimSWARM modules
import SimSWARM.Signal as sg
import SimSWARM.Blocks as bl

# turn interactive plot on
plt.interactive(True)

def main():
	
	# The signals herein are mostly encapsulated in AnalogSignal 
	# instances which need to be sampled in order to visualize.
	# The sampling characteristics for this purpose are defined below.
	rate = 16.0e9 # 16 samples per SoI period
	num_of_samples = 64 # total of 64 samples (4 SoI periods)
	time_start = 0 # sample from t=0
	tvec = np.arange(time_start,1.0*num_of_samples/rate,1.0/rate)
	# scale tvec to nanoseconds
	tvec = tvec*1.0e9
	
	# First a bunch of analog signals. Each analog signal needs a generator
	# which can be used to generate signal samples according to its type
	# and the parameters used to define the generator.
	#
	# signal-of-interest
	soi_frequency = 1.0e9
	soi_amplitude = 1.0
	soi_phase = 0.0
	soi_generator = sg.SinusoidGenerator(soi_amplitude,soi_frequency,soi_phase)
	#
	# construct two inpependent noise sources
	n1_mean = 0.0
	n1_variance = 0.1
	n1_generator = sg.GaussianNoiseGenerator(n1_mean,n1_variance)
	n2_mean = 0.5
	n2_variance = 0.05
	n2_generator = sg.GaussianNoiseGenerator(n2_mean,n2_variance)
	#
	# create analog signals with these generators
	soi_signal = sg.AnalogSignal(soi_generator)
	n1_signal = sg.AnalogSignal(n1_generator)
	n2_signal = sg.AnalogSignal(n2_generator)
	
	# The input signals are now defined, so now some blocks can be added
	# to describe the signal paths.
	#
	# define block to combine the SoI and noise at antenna 1
	block_ant1_1 = bl.AnalogCombiner()
	ant1_inputs = list((soi_signal,n1_signal))
	block_ant1_1.attach_source(ant1_inputs)
	#
	# define block for relative gain offset of antenna 1
	gain_ant1 = 1.2
	block_ant1_2 = bl.AnalogGain(gain_ant1)
	# and attach output of combiner block to its input
	block_ant1_2.attach_source(block_ant1_1)
	#
	# define block for relative delay to antenna 2
	delay_ant2 = 1.0/(4.0*soi_frequency) # quarter-wavelength at SoI frequency
	block_ant2_1 = bl.AnalogDelay(delay_ant2)
	block_ant2_1.attach_source(soi_signal)
	#
	# combine noise after delay
	block_ant2_2 = bl.AnalogCombiner()
	ant2_inputs = list((block_ant2_1,n2_signal))
	block_ant2_2.attach_source(ant2_inputs)
	#
	# define a block to combine the outputs of the two antennas
	block_array_1 = bl.AnalogCombiner()
	array_inputs = list((block_ant1_2,block_ant2_2))
	block_array_1.attach_source(array_inputs)
	
	# Everything is now defined, and a call to the array combiner block
	# should now output the signal.
	array_output = block_array_1.output()
	
	# Plot some of the signals in the simulation.
	#
	# original signal and noise
	plt.figure()
	plt.plot(tvec,soi_signal.sample(rate,num_of_samples,time_start),label='SoI')
	plt.plot(tvec,n1_signal.sample(rate,num_of_samples,time_start),label='Noise 1')
	plt.plot(tvec,n2_signal.sample(rate,num_of_samples,time_start),label='Noise 2')
	plt.legend()
	plt.xlabel('Time [ns]')
	plt.show()
	#
	# plot antenna 1 signals
	ant1_midput = block_ant1_1.output()
	ant1_output = block_ant1_2.output()
	plt.figure()
	plt.plot(tvec,soi_signal.sample(rate,num_of_samples,time_start),'--',label='SoI')
	plt.plot(tvec,n1_signal.sample(rate,num_of_samples,time_start),'--',label='Noise 1')
	plt.plot(tvec,ant1_midput.sample(rate,num_of_samples,time_start),':',label='Signal + Noise')
	plt.plot(tvec,ant1_output.sample(rate,num_of_samples,time_start),label='Ant1 output')
	plt.legend()
	plt.xlabel('Time [ns]')
	plt.show()
	#
	# plot antenna 2 signals
	ant2_midput = block_ant2_1.output()
	ant2_output = block_ant2_2.output()
	plt.figure()
	plt.plot(tvec,soi_signal.sample(rate,num_of_samples,time_start),'--',label='SoI')
	plt.plot(tvec,ant2_midput.sample(rate,num_of_samples,time_start),':',label='SoI delayed')
	plt.plot(tvec,n2_signal.sample(rate,num_of_samples,time_start),':',label='Noise 2')
	plt.plot(tvec,ant2_output.sample(rate,num_of_samples,time_start),label='Ant2 output')
	plt.legend()
	plt.xlabel('Time [ns]')
	plt.show()
	#
	# plot antenna outpus and array signal
	plt.figure()
	plt.plot(tvec,ant1_output.sample(rate,num_of_samples,time_start),'--',label='Ant1 output')
	plt.plot(tvec,ant2_output.sample(rate,num_of_samples,time_start),'--',label='Ant2 output')
	plt.plot(tvec,array_output.sample(rate,num_of_samples,time_start),label='Array output')
	plt.legend()
	plt.xlabel('Time [ns]')
	plt.show()
	
	# wait for input before closing
	print "Press ENTER to exit."
	raw_input()
	
	return 0

if __name__ == '__main__':
	main()

