#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  example_fft.py
#  Jan 27, 2015 12:26:53 EST
#  Copyright 2015
#         
#  Andre Young <andre.young@cfa.harvard.edu>
#  Harvard-Smithsonian Center for Astrophysics
#  60 Garden Street, Cambridge
#  MA 02138
#  
#  Changelog:
#  	AY: Created 2015-01-27
#	AY: Added use of DigitalRealFFT block

"""
Example script to illustrate the use of the FFT block.

"""


# some useful libraries to import
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

# import SimSWARM modules
import SimSWARM.Signal as sg
import SimSWARM.Blocks as bl
import FixedWidthBinary as fw

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
	filter_block = bl.AnalogFrequencyGainSlope(-3) # -3dB/GHz
	filter_block.attach_source(combiner_block)
	
	# Digitize the signal
	adc_bits = 18
	adc_lsb_pot = -8 # least-significant bit as power of two
	adc_precision = fw.WordFormat(adc_bits,adc_lsb_pot)
	adc_block = bl.AnalogDigitalConverter(rate,num_of_samples,adc_precision)
	adc_block.attach_source(filter_block)
	
	# Do FFT
	fft_block = bl.DigitalFFT(adc_precision)
	fft_block.attach_source(adc_block)
	spectrum = fft_block.output()
	alt_spectrum = fft_block._alt_output()
	
	# Do realFFT in parallel
	rfft_block = bl.DigitalRealFFT(adc_precision)
	rfft_block.attach_source(adc_block)
	rspectrum = rfft_block.output()
	alt_rspectrum = rfft_block._alt_output()
	
	# Get frequency sample points
	fmax = 0.5*rate
	fstep = rate/num_of_samples
	fvec = np.arange(-fmax,fmax,fstep)
	rfvec = np.arange(0,fmax,fstep)
	
	# plot results
	#raise RuntimeError("Break.")
	plt.figure()
	plt.plot(fvec/1e9,np.fft.fftshift(spectrum.samples.real),'o',label='Real (DiFFT)',markerfacecolor='w')
	plt.plot(fvec/1e9,np.fft.fftshift(spectrum.samples.imag),'s',label='Imag (DiFFT)',markerfacecolor='w')
	plt.plot(fvec/1e9,np.fft.fftshift(alt_spectrum.samples.real),'x',label='Real')
	plt.plot(fvec/1e9,np.fft.fftshift(alt_spectrum.samples.imag),'+',label='Imag')
	plt.legend()
	plt.title('Signal spectrum')
	plt.xlabel('Frequency [GHz]')
	plt.show()

	plt.figure()
	plt.plot(rfvec/1e9,rspectrum.samples.real,'o',label='Real (DirFFT)',markerfacecolor='w')
	plt.plot(rfvec/1e9,rspectrum.samples.imag,'s',label='Imag (DirFFT)',markerfacecolor='w')
	plt.plot(rfvec/1e9,alt_rspectrum.samples.real,'x',label='Real')
	plt.plot(rfvec/1e9,alt_rspectrum.samples.imag,'+',label='Imag')
	plt.legend()
	plt.title('Signal spectrum')
	plt.xlabel('Frequency [GHz]')
	plt.show()

	plt.figure()
	plt.plot(tvec,combiner_block.output().sample(rate,num_of_samples,time_start),'-',label='Input signal')
	plt.plot(tvec,filter_block.output().sample(rate,num_of_samples,time_start),'-',label='Filtered signal')
	plt.step(tvec,adc_block.output().samples,'-',label='Digitized signal',where='post')
	plt.legend()
	plt.title('Time domain signal')
	plt.xlabel('Time [ns]')
	plt.show()
	
	
	# wait for input before closing
	print "Press ENTER to exit."
	raw_input()
	
	return 0

if __name__ == '__main__':
	main()

