#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  example_slope.py
#  Jan 27, 2015 08:48:53 EST
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
Simple example that uses frequency magnitude slope.

The input signal contains a number of frequency components, and is
low-pass filtered using a frequency magnitude slope block.
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
	tone_amplitudes = 1.0
	tone_phases = 0.0
	for tone_frequency in tone_frequencies:
		tone_generator = sg.SinusoidGenerator(tone_amplitudes,tone_frequency,tone_phases)
		soi_list.append(sg.AnalogSignal(tone_generator))
	
	combiner_block = bl.AnalogCombiner()
	combiner_block.attach_source(soi_list)
	
	# Create the filter block
	filter_block = bl.AnalogFrequencyGainSlope(-10.0) # -10dB/GHz
	filter_block.attach_source(combiner_block)
	
	# Compare filter input and output
	filter_in = combiner_block.output()
	filter_out = filter_block.output()
	
	# plot results
	plt.figure()
	plt.plot(tvec,filter_in.sample(rate,num_of_samples,time_start),'--',label='Filter input')
	plt.plot(tvec,filter_out.sample(rate,num_of_samples,time_start),'--',label='Filter output')
	plt.legend()
	plt.xlabel('Time [ns]')
	plt.show()
	
	
	# wait for input before closing
	print "Press ENTER to exit."
	raw_input()
	
	return 0

if __name__ == '__main__':
	main()

