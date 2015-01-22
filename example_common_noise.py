#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  example_common_noise.py
#  Jan 21, 2015 15:52:09 EST
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
Simple example to illustrate common noise signals.

Gaussian noise source is received with different delays at three
different antennas.

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
	
	# construct noise source
	n_mean = 0.0
	n_variance = 0.1
	n_generator = sg.GaussianNoiseGenerator(n_mean,n_variance)
	n_signal = sg.AnalogSignal(n_generator)
	
	sample_period = 1.0/rate
	delays = np.array((-3,2,1.5,-1.5,8.75,np.pi,32,-16))*sample_period
	for d in delays:
		b = bl.AnalogDelay(d)
		b.attach_source(n_signal)
		output = b.output()
		plt.figure()
		plt.plot(tvec-d*1e9,n_signal.sample(rate,num_of_samples,time_start),'--',label='Input')
		plt.plot(tvec,output.sample(rate,num_of_samples,time_start),'-',label='Output')
		plt.xlabel('Time [ns]')
		plt.legend()
		plt.show()
		#break
	
	# wait for input before closing
	print "Press ENTER to exit."
	raw_input()
	
	return 0

if __name__ == '__main__':
	main()

