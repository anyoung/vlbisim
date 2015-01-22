#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  example_adc.py
#  Jan 21, 2015 17:00:20 EST
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
Simple example to illustrate the use of an adc.

Use as input a 1GHz sinusoid with amplitude of 7, and sample at
64 times the Nyquist rate (128Gsps), using 4.4b quantization, i.e. 4bit
integer and 4bit fractional part. Using this format for the binary 
output the maximum and minimum values that can be represented are
7.9375 and -8.0. Then, requantize the signal using 2.0b, for which
the minimum and maximum values are -2 and 1.

"""

# some useful libraries to import
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

# import SimSWARM modules
import SimSWARM.Signal as sg
import SimSWARM.Blocks as bl

# import FixedWidthBinary module
import FixedWidthBinary as fw

# turn interactive plot on
plt.interactive(True)

def main():
	
	
	# The signals herein are mostly encapsulated in AnalogSignal 
	# instances which need to be sampled in order to visualize.
	# The sampling characteristics for this purpose are defined below.
	rate = 128.0e9 # 16 samples per SoI period
	num_of_samples = 256 # total of 64 samples (4 SoI periods)
	time_start = 0 # sample from t=0
	tvec = np.arange(time_start,1.0*num_of_samples/rate,1.0/rate)
	# scale tvec to nanoseconds
	tvec = tvec*1.0e9
	
	# First create a signal generator and the input AnalogSignal.
	soi_frequency = 1.0e9
	soi_amplitude = 9.0
	soi_phase = 0.0
	soi_generator = sg.SinusoidGenerator(soi_amplitude,soi_frequency,soi_phase)
	soi_signal = sg.AnalogSignal(soi_generator)
	
	# Use the above sampling characteristics for time discretization
	# and define the amplitude discretization.
	adc_bits = 8
	adc_lsb_pot = -4 # least-significant bit as power of two
	adc_precision = fw.WordFormat(adc_bits,adc_lsb_pot)
	
	# Define an ADC block
	block_adc = bl.AnalogDigitalConverter(rate,num_of_samples,adc_precision)
	block_adc.attach_source(soi_signal)
	
	# Define a requantize block
	rq_bits = 2
	rq_lsb_pot = 0 # least-significant bit as power of two
	rq_precision = fw.WordFormat(rq_bits,rq_lsb_pot)
	block_rq = bl.Requantizer(rq_precision)
	block_rq.attach_source(block_adc)
	
	# Generate output
	adc_out = block_adc.output()
	rq_out = block_rq.output()
	
	# Analog-to-Digital and Digital block outputs are DigitalSignal instances,
	# which means the signal samples are immediately available.
	adc_samples = adc_out.samples
	rq_samples = rq_out.samples
	
	# Plot results
	plt.figure()
	plt.plot(tvec,soi_signal.sample(rate,num_of_samples,time_start),'--',label='Analog')
	plt.step(tvec,adc_samples,'-',label='ADC',where='post')
	plt.step(tvec,rq_samples,'-',label='RQ',where='post')
	plt.xlabel('Time [ns]')
	plt.legend()
	plt.show()
	
	# wait for input before closing
	print "Press ENTER to exit."
	raw_input()
	
	return 0

if __name__ == '__main__':
	main()

