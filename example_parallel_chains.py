#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  example_parallel_chains.py
#  Jan 27, 2015 10:45:00 EST
#  Copyright 2015
#         
#  Andre Young <andre.young@cfa.harvard.edu>
#  Harvard-Smithsonian Center for Astrophysics
#  60 Garden Street, Cambridge
#  MA 02138
#  
#  Changelog:
#  	AY: Created 2015-01-27

"""
Simple example to illustrate the use of parallel signal chains.

A multitone signal is used as input to two parallel signal processing
paths. Each path consists of two chains. The first chain consists of an
analog delay and flat gain, the parameters of which are different for each
parallel path. The second chain consists of a frequency phase and magnitude
slope, which is use the same parameters for both parallel paths. 

The input contains a 1GHz and 4GHz sinusoid. The delay is a quarter period (1GHz), 
positive for one path and negative for the other. The flat gain is 1.2 for 
the one path and 0.8 for the other. The phase slope implements a half-period
delay (1GHz) and the magnitude slope effectively removes the 4GHz tone.
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
	rate = 64.0e9 # 
	num_of_samples = 128 # 
	time_start = 0 # sample from t=0
	tvec = np.arange(time_start,1.0*num_of_samples/rate,1.0/rate)
	# scale tvec to nanoseconds
	tvec = tvec*1.0e9
	
	# Create the multitone signal
	#
	soi_list = list()
	tone_frequencies = (1.0e9,4.0e9)
	tone_amplitudes = (1.0,0.1)
	tone_phases = 0.0
	for ii in range(0,len(tone_frequencies)):
		tone_frequency = tone_frequencies[ii]
		tone_amplitude = tone_amplitudes[ii]
		tone_generator = sg.SinusoidGenerator(tone_amplitude,tone_frequency,tone_phases)
		soi_list.append(sg.AnalogSignal(tone_generator))
	
	combiner_block = bl.AnalogCombiner()
	combiner_block.attach_source(soi_list)
	
	# Build the first chain (separate)
	# for antenna 1
	ant1_delay = 0.25/1.0e9 # quarter period delay at 1GHz
	ant1_flat_gain = 1.2
	chain1_1 = build_chain_separate(ant1_delay,ant1_flat_gain)
	# and for antenna 2
	ant2_delay = -0.25/1.0e9 # quarter period advance at 1GHz
	ant2_flat_gain = 0.8
	chain1_2 = build_chain_separate(ant2_delay,ant2_flat_gain)
	# Create Parallel using list - used when the blocks in each 
	# parallel path are different
	par1 = bl.Parallel(list((chain1_1,chain1_2)))
	
	# Build the second chain (common)
	both_phase_slope = 0.5e-9 # half period delay at 1GHz
	both_magnitude_slope = -10 # dB/GHz
	chain2 = build_chain_common(both_phase_slope,both_magnitude_slope)
	# Create Parallel using single block - used when the blocks in each
	# parallel path is the same
	par2 = bl.Parallel(chain2,n=2) # have to specify n, the number of paths
	
	# Connect the signal path
	par1.attach_source(combiner_block)
	par2.attach_source(par1)
	
	# Generate output
	out1 = par1.output()
	out2 = par2.output()
	
	# Plot the results
	plt.figure()
	plt.plot(tvec,combiner_block.output().sample(rate,num_of_samples,time_start),label='SoI')
	plt.legend()
	plt.title('Signal with large 1GHz and small 4GHz component, both start at 0deg phase.')
	plt.xlabel('Time [ns]')
	plt.show()
	
	plt.figure()
	plt.plot(tvec,out1[0].sample(rate,num_of_samples,time_start),'--',label='Chain 1 (channel 1)')
	plt.plot(tvec,out1[1].sample(rate,num_of_samples,time_start),'--',label='Chain 1 (channel 2)')
	plt.legend()
	plt.title('Output of first chain: 1.2 gain + delay (channel 1) and 0.8 gain + advance (channel 2)')
	plt.xlabel('Time [ns]')
	plt.show()
	
	plt.figure()
	plt.plot(tvec,out2[0].sample(rate,num_of_samples,time_start),'--',label='Chain 2 (channel 1)')
	plt.plot(tvec,out2[1].sample(rate,num_of_samples,time_start),'--',label='Chain 2 (channel 2)')
	plt.legend()
	plt.title('Output of second chain: 4GHz removed and phase inversion.')
	plt.xlabel('Time [ns]')
	plt.show()
	
	# wait for input before closing
	print "Press ENTER to exit."
	raw_input()
	
	
	return 0

def build_chain_separate(delay,flat_gain):
	"""
	Build and return a signal processing block chain.
	
	The chain comprises a delay block and a flat gain block,
	in that order.
	
	Arguments:
	delay -- Delay block parameter.
	flat_gain -- Flat gain block parameter
	
	"""
	
	chain = bl.Chain()
	chain.add_block(bl.AnalogDelay(delay))
	chain.add_block(bl.AnalogGain(flat_gain))
	
	return chain

def build_chain_common(phase_slope,magnitude_slope):
	"""
	Build and return a signal processing block chain.
	
	The chain comprises a frequency domain phase and magnitude slope.
	
	Arguments:
	phase_slope -- The phase gradient parameter.
	magnitude_slope -- The magnitude gradient parameter.
	
	"""
	
	chain = bl.Chain()
	chain.add_block(bl.AnalogFrequencyPhaseSlope(phase_slope))
	chain.add_block(bl.AnalogFrequencyGainSlope(magnitude_slope))
	
	return chain

if __name__ == '__main__':
	main()

