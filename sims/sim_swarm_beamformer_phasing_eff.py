#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  sim_swarm_beamformer_phasing_eff.py
#  Jan 28, 2015 09:50:37 EST
#  Copyright 2015
#         
#  Andre Young <andre.young@cfa.harvard.edu>
#  Harvard-Smithsonian Center for Astrophysics
#  60 Garden Street, Cambridge
#  MA 02138
#  
#  Changelog:
#  	AY: Created 2015-01-28

"""
Simulation to determine impact of SWARM beamformer on phasing efficiency.

"""

# some useful libraries to import
import sys
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

# import to time execution
import time

# interactive plotting
plt.interactive(True)

# add local path for custome modules
sys.path.append('../')

# import for handling fixed-width binary representation
import FixedWidthBinary as fw

# import for system description building blocks
import SimSWARM.Antenna as an
import SimSWARM.Blocks as bl
import SimSWARM.Signal as sg
import SimSWARM.Source as so

# Global parameters
#
# Mauna Kea lattitude in degrees. This is used as estimate for center
# of observatory (SMA).
LAT_OBS_DEG = 19.0 + 49.0/60.0 + 28.0/3600.0
#
# Number of antennas to use
NUM_ANT = 8
#
# Input signal characteristics
TARGET_MEAN = 0.0
TARGET_VARIANCE = 1.0
NOISE_MEAN = 0.0
NOISE_VARIANCE = 0.1#1e-8#0.0001
#
# ADC characteristics
ADC_RATE = 4576e6 # samples per second
ADC_NUM_OF_SAMPLES = 2**12
ADC_WIDTH = 8 # bits
#
# FFT characteristics
FFT_WIDTH = 18 # bits
#
# Weighting characteristics
WEIGHT_WIDTH = 25
#
# Requantization characteristics
RQ_WIDTH = 2

def main():
	
	# Initialize signal path
	chain = bl.Chain()
	
	# Create the array
	array = generate_array()
	
	# Create the target source
	target_source = generate_target_source()
	array.add_source(target_source)
	
	# Add independent noise to each antenna
	add_antenna_noise_to_array(array)
	
	# Create array block
	array_block = array.receiver_block()
	# and insert in path
	chain.add_block(array_block)
	
	# Create ADC
	# 
	# Calculate an estimate of the maximum signal input. Use 5sigma
	# of the target source signal + noise signal
	max_in = 5.0*np.sqrt(TARGET_VARIANCE + NOISE_VARIANCE)
	adc_block = generate_adc(max_in)
	# and insert in path
	chain.add_block(adc_block)
	
	# Create FFT engine
	fft_block = generate_fft(max_in,ADC_NUM_OF_SAMPLES)
	# and insert in path
	chain.add_block(fft_block)
	
	# Extract the per-channel spectrum as an intermediate result
	print "Computing per-channel spectrum..."
	t_start = time.time()
	out_fft = chain.output()
	t_end = time.time()
	print "done. Calculation took {:10.3f} seconds".format(t_end-t_start)
	
	# plot cross-correlations on raw FFT output
	plot_cross_correlations(out_fft)
	plt.title('Cross-correlations from FFT output')
	pe_fft, pea_f = calculate_phasing_efficiency(out_fft)
	
	# Create beamformer chain
	#
	# initialize new chain
	chain_beamformer = bl.Chain()
	#
	# compute the weighting of each channel
	weights = calculate_scanning_weights(array,target_source.position)
	weights_block = generate_weights(weights)
	# and insert in path
	chain_beamformer.add_block(weights_block)
	# then attach FFT output signal as source for beamformer chain
	chain_beamformer.attach_source(out_fft)
	
	# Extract weighted channel data as intermediate result
	out_weights = weights_block.output()
	plot_cross_correlations(out_weights)
	plt.title('Cross-correlations from weighted output')
	pe_weights, pea_w = calculate_phasing_efficiency(out_weights)
	
	# Requantize to 4bit
	#
	# build requantizer block
	rq_block = generate_requantizer(max_in*np.sqrt(ADC_NUM_OF_SAMPLES))
	# and insert in path
	chain_beamformer.add_block(rq_block)
	
	# Extract weighted channel data as intermediate result
	out_rq = rq_block.output()
	plot_cross_correlations(out_rq)
	plt.title('Cross-correlations from requantized output')
	pe_rq, pea_r = calculate_phasing_efficiency(out_rq)
	
	# Do beamforming summation
	combiner_block = bl.DigitalCombiner()
	# and insert in path
	chain_beamformer.add_block(combiner_block)
	
	# Extract beamformer output
	# 
	print "Computing beamformer output..."
	t_start = time.time()
	out_beamformer = chain_beamformer.output()
	t_end = time.time()
	print "done. Calculation took {:10.3f} seconds".format(t_end-t_start)
	
	# Phasing efficiency results
	plt.figure()
	plt.plot(pe_fft,label='FFT')
	plt.plot(pe_weights,label='Weighted')
	plt.plot(pe_rq,label='Requantized')
	plt.title('Phasing efficiencies at different parts in the processing chain.')
	print "Average phasing efficiencies across the band: {:10.9f} (FFT), {:10.9f} (weighted), {:10.9f} (requantized)".format(pea_f,pea_w,pea_r)
	
	# wait for input before closing
	print "Press ENTER to exit."
	raw_input()
	
	return out_fft,out_beamformer,pe_fft,pe_weights,pe_rq

def generate_array():
	"""
	Generate the array used for the observation.
	
	The array is based on the SMA configuration for March 2013 VLBI
	observation.
	
	"""
	
	# The antenna positions are stored in file, copied and pasted from
	# SMA wiki. Each line contains:
	#	ANT_NUM PAD_NUM X Y Z
	fh_ant_pos = open('ant_pos.txt')
	lines = fh_ant_pos.readlines()
	ant_list = list()
	for line in lines:
		line = line.split()
		x = float(line[2])
		y = float(line[3])
		z = float(line[4])
		ant_list.append(an.Antenna((x,y,z)))
		if (len(ant_list) == NUM_ANT):
			break

	array = an.Array(ant_list)
	
	return array

def add_antenna_noise_to_array(array):
	"""
	Generate independent noise signals and attach them to the antennas.
	
	"""
	
	# noise parameters
	mean = NOISE_MEAN
	variance = NOISE_VARIANCE
	for ant in array.antennas:
		generator = sg.GaussianNoiseGenerator(mean,variance)
		signal = sg.AnalogSignal(generator)
		position = so.LocalPosition()
		source = so.PointSource(signal,position)
		ant.add_source(source)

def generate_target_source(H=2.0):
	"""
	Generate the target source which is loosely based on SgrA*.
	
	Uses a gaussian noise signal model.
	
	Keyword arguments:
	H -- The hour angle of the source in hours. (default H=0.0)
	
	"""
	
	# Signal parameters
	mean = TARGET_MEAN
	variance = TARGET_VARIANCE
	# Source declination in degrees
	DECL_SRC_DEG = -1.0*(29.0 + 0.0/60.0 + 28.118/3600.0)
	
	# build signal
	generator = sg.GaussianNoiseGenerator(mean,variance)
	signal = sg.AnalogSignal(generator)
	
	# The source position is defined by (theta,phi)-coordinates where
	# theta measures from zenith to the horizon, and phi from North 
	# through East, South, and West. Using cosine rule for spherical 
	# triangles, these coordinates can be found through:
	# 	cos(theta) = cos(L)cos(d) + sin(L)sin(d)cos(H)
	#	cos(d) = cos(L)cos(theta) + sin(L)sin(theta)cos(phi)
	# where L is the observatory lattitude measured from North to South,
	# d the source lattitude measured from North to South, and h the 
	# hour angle. Corrections for phi are required as follows:
	#	for theta == 0.0, phi = 0.0
	#	for H == 0.0, phi = 0.0 (if d < L) or phi = pi (if d > L)
	#	for H > 0.0, phi is negated
	L = np.deg2rad(90.0 - LAT_OBS_DEG)
	d = np.deg2rad(90.0 - DECL_SRC_DEG)
	H = np.deg2rad(H*15.0)
	
	if (H == 0.0):
		theta = np.abs(d-L)
		if (d > L):
			phi = 180.0
		else:
			# this is valid for d < L AND d == L, i.e. theta == 0.0
			phi = 0.0
	else:
		cos_theta = np.cos(L)*np.cos(d) + np.sin(L)*np.sin(d)*np.cos(H)
		theta = np.rad2deg(np.arccos(cos_theta))
		if (theta == 0.0):
			phi = 0.0
		else:
			cos_phi = (np.cos(d) - np.cos(L)*cos_theta) / (np.sin(L)*np.sqrt(1.0-cos_theta**2.0)) # sin(x) = sqrt(1-cos(x)^2)
			phi = np.rad2deg(np.arccos(cos_phi))
			if (H > 0.0):
				phi = -phi
	
	position = so.SkyPosition((theta,phi))
	source = so.PointSource(signal,position)
	
	return source

def generate_adc(max_in):
	"""
	Generate an ADC parallel block based on the global characteristics.
	
	Arguments:
	max_in -- Estimate of the absolute maximum of the input signal.
	
	"""
	
	# calculate how many decimal places left
	resolution_pot = np.ceil(np.log2(np.abs(max_in))) - ADC_WIDTH
	
	precision = fw.WordFormat(ADC_WIDTH,resolution_pot)
	adc = bl.AnalogDigitalConverter(ADC_RATE,ADC_NUM_OF_SAMPLES,precision)
	adc_array = bl.Parallel(adc,NUM_ANT)
	
	return adc_array

def generate_fft(max_in,num_pts):
	"""
	Generate an FFT block.
	
	Arguments:
	max_in -- Estimate of the absolute maximum of the input signal.
	num_pts -- Number of input samples.
	
	"""
	
	# estimate optimal distribution of bits between integer and fractional
	resolution_pot = np.ceil(np.log2(np.abs(max_in*np.sqrt(num_pts)))) - FFT_WIDTH
	precision = fw.WordFormat(FFT_WIDTH,resolution_pot)
	print "Creating DigitalFFT with precision =",precision 
	fft = bl.DigitalFFT(precision)
	fft_array = bl.Parallel(fft,NUM_ANT)
	
	return fft_array

def generate_weights(weights):
	"""
	Generate a DigitalGain Parallel that applies spectral weighting to each parallel path.
	
	Arguments:
	weights -- The complex-valued weights to apply given as 2D numpy array.
	The first dimension is along antennas in the array, and the second 
	dimension is along the frequency bins.
	
	Notes:
	Weights are assumed to have unit magnitude so that the WordFormat only
	uses 2bits for integer values, and the rest of WEIGHT_WIDTH for 
	fractional bits.
	
	"""
	
	resolution_pot = 2 - WEIGHT_WIDTH
	precision = fw.WordFormat(WEIGHT_WIDTH,resolution_pot)
	weight_blocks = list()
	for ii in range(0,NUM_ANT):
		weight_blocks.append(bl.DigitalGain(weights[ii,:].squeeze(),precision))
	
	weight_block = bl.Parallel(weight_blocks)
	
	return weight_block

def calculate_scanning_weights(array,src_pos):
	"""
	Calculate the beamformer weights to implement a scanning array towards the given source position."
	
	Arguments:
	array -- Array implementation.
	src_pos -- SkyPosition instance that is the position of the source.
	
	Notes:
	The frequencies for which weights are calculated are determined based
	on the ADC sampling characteristics.
	
	"""
	
	fmax = ADC_RATE/2.0
	fstep = ADC_RATE/ADC_NUM_OF_SAMPLES
	fvec = np.arange(-fmax,fmax,fstep)
	x,y,z = array.position
	l,m,n = src_pos.coords_lmn
	weights = np.zeros((NUM_ANT,len(fvec)),dtype=np.complex)
	#plt.figure()
	for ii in range(0,NUM_ANT):
		b_dot_s = x[ii]*l + y[ii]*m + z[ii]*n
		weights[ii,:] = np.exp(1j*2.0*np.pi*fvec/sp.constants.c*b_dot_s)
		# weight -fmax component to zero to preserve hermitian symmetry in spectrum
		weights[ii,0] = 0.0
		# apply ifftshift since the FFT block output is not fftshifted
		weights[ii,:] = np.fft.ifftshift(weights[ii,:])
		#plt.plot(weights[ii,:].real)
		#lt.plot(weights[ii,:].imag,'--')

	return weights

def generate_requantizer(max_in):
	"""
	Generate a requantation block prior to beamforming.
	
	Arguments:
	max_in -- Estimate of the maximum signal to be quanitized.
	
	"""
	
	resolution_pot = np.ceil(np.log2(np.abs(max_in))) - RQ_WIDTH
	precision = fw.WordFormat(RQ_WIDTH,resolution_pot)
	rq_block = bl.Requantizer(precision)
	rq_parallel = bl.Parallel(rq_block,NUM_ANT)
	
	return rq_parallel

def calculate_phasing_efficiency(fd_signals):
	"""
	Calculate the phasing efficiency across the band for the given set
	of frequency-domain signals.
	
	Arguments:
	fd_signals -- List of frequency domain DigitalSignal instances.
	
	"""
	
	corr_out = calculate_correlator_products(fd_signals)
	# sum is over axis 0 and 1, but first sum reduces shape, so second sum is also over axis 0
	numer = np.abs(np.sum(np.sum(corr_out,axis=0),axis=0))
	denom = np.sum(np.sum(np.abs(corr_out),axis=0),axis=0)
	pe = numer/denom
	pea = np.sum(numer)/np.sum(denom)
	return pe, pea

def calculate_correlator_products(fd_signals):
	"""
	Calculate the correlator product for the given frequency domain signals.
	
	Arguments:
	fd_signals -- List of frequency domain DigitalSignal instances.
	
	"""
	
	corr_out = np.zeros((NUM_ANT,NUM_ANT,ADC_NUM_OF_SAMPLES),dtype=np.complex)
	for ii in np.arange(0,NUM_ANT):
		for jj in np.arange(0,NUM_ANT):
			corr_out[ii,jj,:] = fd_signals[ii].samples * np.conjugate(fd_signals[jj].samples)
	
	return corr_out

def plot_cross_correlations(fd_signals):
	"""
	Plot cross-correlations by multiplying in frequency domain and iFFT.
	
	Arguments:
	fd_signals -- Frequency-domain signals as a list of DigitalSignal.
	
	Notes:
	Returns a handle to the created figure.
	
	Can also print statistics relating to the delay (peak in 
	cross-correlation) by uncommenting the print statement below.
	"""
	
	# calculate cross-correlation spectrum
	fh = plt.figure()
	plt.title('Cross-correlations between Ant#0 and Ant#1-7')
	tvec = np.arange(0,ADC_NUM_OF_SAMPLES/ADC_RATE,1.0/ADC_RATE)
	for ii in np.arange(1,len(fd_signals)):
		R_0_x = fd_signals[0].samples * np.conjugate(fd_signals[ii].samples)
		r_0_x = np.fft.ifft(R_0_x)
		r_0_x_max = r_0_x.real.max()
		r_0_x_mean = np.mean(r_0_x.real)
		r_0_x_std = r_0_x.std()
		plt.plot(tvec*1e9,r_0_x.real,label="Ant#0-Ant#{:1d}".format(ii))
		#print "Peak is ", (r_0_x_max - r_0_x_mean)/r_0_x_std, "sigma above mean."
	
	plt.legend()
	plt.xlabel('Delay [ns]')
	plt.ylabel('Correlation')
	
	return fh
	
def plot_signals(sb_list):
	"""
	Plot the signals for the Block/Signal items in the list.
	
	Arguments:
	sb_list -- List of Block/Signal objects.
	
	"""
	
	for sb in sb_list:
		# create a new figure for each item
		plt.figure()
		title_str = str(sb)
		plt.title(title_str)
		
		samples_to_plot = get_samples_from_signal_or_block(sb)
		if (samples_to_plot == None):
			# impossible to plot, just put text that says so
			plt.text(0.5,0.5,'Unable to plot signal.',horizontalalignment='center',verticalalignment='center')
			plt.show()
			continue
		
		plt.plot(samples_to_plot)
		plt.show()
		#print "Samples to plot shape: ", samples_to_plot.shape
		
def get_samples_from_signal_or_block(sb):
	"""
	Return signal samples from given Block / Signal.
	
	Arguments:
	sb -- Signal or Block instance.
	
	Notes:
	If sb is a Block that returns a list of samples, then a 2D numpy 
	array is created so that a plot call will plot each sample string.
	
	"""
	
	num_samples_to_plot = min(1024,ADC_NUM_OF_SAMPLES)
	
	# if sb a block, first request output
	if (isinstance(sb,bl.Block)):
		sb = sb.output()
	
	
	if (isinstance(sb,sg.AnalogSignal)):
		# if analog signal, then first sample
		samples = sb.sample(ADC_RATE,num_samples_to_plot,0.0)
	elif (isinstance(sb,sg.DigitalSignal)):
		# if digital signal, select subset of samples
		samples = sb.samples[0:num_samples_to_plot]
	else:
		# only other valid option is some iterable (e.g. Parallel output)
		try:
			samples = None
			for sb2 in sb:
				this_samples = get_samples_from_signal_or_block(sb2)
				#print "this_samples.shape = ", this_samples.shape
				if (samples == None):
					samples = this_samples.reshape((-1,1))
				else:
					samples = np.concatenate((samples,this_samples.reshape((-1,1))),axis=1)
		except:
			# unable to find plotable samples
			return None
	
	return samples

if __name__ == '__main__':
	main()

