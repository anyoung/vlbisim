#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  example_chains.py
#  Jan 21, 2015 15:26:02 EST
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
Simple example to illustrate the use of signal chains.

A 1GHz sinusoidal is used as input to two antennas. Each antenna
applies a phase gradient and flat gain to the received signals. 
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
	
	# First create a signal generator and the input AnalogSignal.
	soi_frequency = 1.0e9
	soi_amplitude = 1.0
	soi_phase = 0.0
	soi_generator = sg.SinusoidGenerator(soi_amplitude,soi_frequency,soi_phase)
	soi_signal = sg.AnalogSignal(soi_generator)
	
	# Build processing chain for antenna 1
	ant1_phase_gradient = 0.5e-9 # 0.5ns delay
	ant1_flat_gain = 2.0
	chain1 = build_chain(ant1_phase_gradient,ant1_flat_gain)
	
	# Build processing chain for antenna 2
	ant2_phase_gradient = -0.25e-9 # 0.25ns advance
	ant2_flat_gain = 0.3
	chain2 = build_chain(ant2_phase_gradient,ant2_flat_gain)
	
	# Attach inputs, the same source for each chain
	chain1.attach_source(soi_signal)
	chain2.attach_source(soi_signal)
	
	# Generate output
	out1 = chain1.output()
	out2 = chain2.output()
	
	# Plot the results
	plt.figure()
	plt.plot(tvec,soi_signal.sample(rate,num_of_samples,time_start),label='SoI')
	plt.plot(tvec,out1.sample(rate,num_of_samples,time_start),'--',label='out1')
	plt.plot(tvec,out2.sample(rate,num_of_samples,time_start),'--',label='out2')
	plt.legend()
	plt.xlabel('Time [ns]')
	plt.show()
	
	# wait for input before closing
	print "Press ENTER to exit."
	raw_input()
	
	
	return 0

def build_chain(phase_gradient,flat_gain):
	"""
	Build and return a signal processing block chain.
	
	The chain comprises a phase gradient block and a flat gain block,
	in that order.
	
	Arguments:
	phase_gradient -- Phase gradient block parameter.
	flat_gain -- Flat gain block parameter
	"""
	
	chain = bl.Chain()
	chain.add_block(bl.AnalogFrequencyPhaseSlope(phase_gradient))
	chain.add_block(bl.AnalogGain(flat_gain))
	
	return chain

if __name__ == '__main__':
	main()

