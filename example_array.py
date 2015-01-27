#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  example_array.py
#  Jan 27, 2015 10:16:08 EST
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
Sample script to illustrate an array simulator, using the Array class.

This basically copies the simulation example_antenna.py, but only using
Array to encapsulate the antenna array instead of handling the antennas
separately.
"""

# some useful libraries to import
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

# import SimSWARM modules
import SimSWARM.Signal as sg
import SimSWARM.Blocks as bl
import SimSWARM.Antenna as an
import SimSWARM.Source as sr

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
	
	# Create the sky source. This requires an AnalogSignal and a Position (for
	# a point-like source) to be created.
	# 
	# first the AnalogSignal
	soi_frequency = 1.0e9
	soi_amplitude = 1
	soi_phase = 0.0
	soi_generator = sg.SinusoidGenerator(soi_amplitude,soi_frequency,soi_phase)
	soi_signal = sg.AnalogSignal(soi_generator)
	# then the position
	theta = 30.0
	phi = 0.0
	soi_pos = sr.SkyPosition((theta,phi))
	# and finally the source
	soi_src = sr.PointSource(soi_signal,soi_pos)
	
	# Create the array. This requires the Antenna instances to be created,
	# which in turn need the coordinates of each antenna as a tuple.
	#
	# antenna 0
	x0 = 0.0
	y0 = 0.0
	z0 = 0.0
	ant0 = an.Antenna((x0,y0,z0))
	#
	# antenna 1
	x1 = sp.constants.c/soi_frequency # one wavelength along the x-axis from ant0
	y1 = 0.0
	z1 = 0.0
	ant1 = an.Antenna((x1,y1,z1))
	# Create the array
	arr = an.Array(list((ant0,ant1)))
	
	# Construct two inpependent noise sources
	n0_mean = 0.0
	n0_variance = 0.001
	n0_generator = sg.GaussianNoiseGenerator(n0_mean,n0_variance)
	n0_signal = sg.AnalogSignal(n0_generator)
	n0_pos = sr.LocalPosition()
	n0_src = sr.PointSource(n0_signal,n0_pos)
	#
	n1_mean = 0.0
	n1_variance = 0.001
	n1_generator = sg.GaussianNoiseGenerator(n1_mean,n1_variance)
	n1_signal = sg.AnalogSignal(n1_generator)
	n1_pos = sr.LocalPosition()
	n1_src = sr.PointSource(n1_signal,n1_pos)
	# and some common noise
	nc_mean = 0.0
	nc_variance = 0.1
	nc_generator = sg.GaussianNoiseGenerator(nc_mean,nc_variance)
	nc_signal = sg.AnalogSignal(nc_generator)
	nc_pos = sr.CartesianPosition(((x0+x1)*0.5,(y0+y1)*0.5,(z0+z1)*0.5)) # mid-point between ant0 and ant1, which means signal arrives simultaneously at both
	nc_src = sr.PointSource(nc_signal,nc_pos)
	
	# Add the independent noise sources to the antennas separately 
	# 
	# ant0
	ant0.add_source(n0_src)
	# 
	# ant1
	ant1.add_source(n1_src)
	# and add the sources common to all antennas to the array
	arr.add_source((soi_src,nc_src))
	
	# Now request the output signals for the array
	R = arr.receiver_block().output()
	r0 = R[0]
	r1 = R[1]
	
	# Plot some of the signals in the simulation.
	#
	# original signal and noise
	plt.figure()
	plt.plot(tvec,soi_signal.sample(rate,num_of_samples,time_start),label='SoI')
	plt.plot(tvec,r0.sample(rate,num_of_samples,time_start),label='ant0')
	plt.plot(tvec,r1.sample(rate,num_of_samples,time_start),label='ant1')
	plt.legend()
	plt.xlabel('Time [ns]')
	plt.show()
	
	# wait for input before closing
	print "Press ENTER to exit."
	raw_input()
	
	return 0

if __name__ == '__main__':
	main()

