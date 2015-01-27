#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  signal.py
#  Jan 20, 2015 11:17:05 EST
#  Copyright 2015
#         
#  Andre Young <andre.young@cfa.harvard.edu>
#  Harvard-Smithsonian Center for Astrophysics
#  60 Garden Street, Cambridge
#  MA 02138
#  
#  Changelog:
#  	AY: Created 2015-01-20
#	AY: Changed frequency magnitude slope to dB/GHz

"""
Defines various signal utilities.

"""

import numpy as np
import scipy.constants as const
import copy

import FixedWidthBinary as fw

pi = np.pi

class Signal(object):
	"""
	Baseclass for all signals.
	
	"""
	
	def __init__(self):
		"""
		Construct Signal instance.
		
		This implementation is empty, derived classes should define a 
		number of Arguments that control the nature of the signal.
		"""
		
		pass
	
	@classmethod
	def copy(cls,s):
		"""
		Construct and return a copy of the given signal.
		
		Arguments:
		s -- Signal instance to copy.
		
		Notes:
		Current implementation uses a deep copy.
		"""
		
		return copy.deepcopy(s)

# end class Signal


class AnalogSignal(Signal):
	"""
	Baseclass for all analog signals.
	
	AnalogSignal is characterized by a signal generator method that
	is called to obtain discrete samples of the signal, given a sample
	rate, number of samples, and time-offset.
	"""
	
	def __init__(self,gen):
		"""
		Construct an analog signal using the given generator method.
		
		Arguments:
		gen -- A Generator object instance.
		
		Notes:
		
		"""
		
		self._generator = gen
	
	def sample(self,r,n,t):
		"""
		Obtain signal samples by calling the Generator's generate method.
		
		Arguments:
		r -- Sample rate in samples per second.
		n -- Number of samples to generate.
		t -- Time offset of first signal.
		"""
		
		return self.generator.generate(r,n,t)
	
	@property
	def generator(self):
		"""
		Return the Generator for this signal.
		
		"""
		
		return self._generator

# end class AnalogSignal


class TransformedAnalogSignal(AnalogSignal):
	"""
	Extend functionality of AnalogSignal to account for analog transformations.
	
	Current implementation allows for a flat gain, time delay, frequency
	magnitude slope, and frequency phase slope. All methods are
	inherited as-is from AnalogSignal, except for the sample method 
	which applies the transformations to the signal before returning the
	result, and the constructor which sets default parameters for all
	transformations. Methods are also provided to set the parameters 
	related to these transformations.
	"""
	
	@property
	def time_delay(self):
		"""
		Return the time delay applied to the signal.
		
		"""
		
		return self._time_delay
	
	@property
	def flat_gain(self):
		"""
		Return the constant gain applied to the signal.
		
		"""
		
		return self._flat_gain
	
	@property
	def frequency_magnitude_slope(self):
		"""
		Return the slope of the magnitude envelope applied to the signal.
		
		"""
		
		return self._frequency_magnitude_slope
	
	@property
	def frequency_phase_slope(self):
		"""
		Return the phase gradient applied to the signal.
		
		"""
		
		return self._frequency_phase_slope
	
	def __init__(self,analog_signal):
		"""
		Construct a transformable analog signal instance.
		
		Arguments:
		analog_signal -- An AnalogSignal instance.
		
		Notes:
		The generator of analog_signal is inherited.
		"""
		
		if (not isinstance(analog_signal, AnalogSignal)):
			raise ValueError("TransformedAnalogSignal can only be created from instances of AnalogSignal.")
		elif (isinstance(analog_signal, TransformedAnalogSignal)):
			self._time_delay = analog_signal.time_delay
			self._flat_gain = analog_signal.flat_gain
			self._frequency_magnitude_slope = analog_signal.frequency_magnitude_slope
			self._frequency_phase_slope = analog_signal.frequency_phase_slope
		else:
			self._time_delay = 0.0
			self._flat_gain = 1.0
			self._frequency_magnitude_slope = None
			self._frequency_phase_slope = None
		
		super(TransformedAnalogSignal,self).__init__(analog_signal.generator)
	
	def sample(self,r,n,t):
		"""
		Obtain signal samples by calling the Generator's generate method.
		
		Arguments:
		r -- Sample rate in samples per second.
		n -- Number of samples to generate.
		t -- Time offset of first signal.
		
		Notes:
		The order of transformations applied to the signal are as follows.
		The generator generate method is called by adding the delay to
		the time offset parameter; this gets the delayed and sampled 
		time-domain signal. The flat gain is then applied by multiplying
		all samples with the constant parameter. An FFT is then performed,
		and the phase and magnitude slopes are applied, using the frequency
		samples corresponding to the time-domain sampling. Finally, the
		iFFT is applied to obtain the corresponding time-domain signal
		samples, which are then returned.
		"""
		
		td_samples = self.flat_gain * self.generator.generate(r,n,t + self.time_delay)
		if ((self.frequency_magnitude_slope == None) and (self.frequency_phase_slope == None)):
			return td_samples
		
		fd_samples = np.fft.fftshift(np.fft.fft(td_samples))
		fmax = r/2.0
		fstep = 1.0*r/n
		fvec = np.arange(-fmax,fmax,fstep)
		if (self.frequency_magnitude_slope != None):
			#fd_samples = fd_samples * (2.0*pi * self.frequency_magnitude_slope * np.abs(fvec))
			fd_samples = fd_samples * 10**((self.frequency_magnitude_slope/20.0) * (np.abs(fvec)/1.0e9))
		
		if (self.frequency_phase_slope != None):
			fd_samples = fd_samples * np.exp(1j*2.0*pi * self.frequency_phase_slope * fvec)
		
		td_samples = np.fft.ifft(np.fft.ifftshift(fd_samples)).real
		return td_samples
	
	def apply_delay(self,d):
		"""
		Apply a delay to the analog signal.
		
		Arguments:
		d -- The delay given in seconds, which can be negative.
		
		Notes:
		Additional time delays are additive.
		"""
		
		self._time_delay = self.time_delay + d
	
	def apply_gain(self,g):
		"""
		Apply a constant gain to the analog signal.
		
		Arguments:
		g -- The flat gain that is applied to all signal samples.
		
		Notes:
		Additional gains are multiplicative.
		"""
		
		self._flat_gain = self.flat_gain * g
	
	def apply_frequency_magnitude_slope(self,m):
		"""
		Apply a magnitude slope to the analog signal in the frequency domain.
		
		Arguments:
		m -- The slope in units dB/GHz
		
		Notes:
		The slope is applied with even symmetry in the frequency domain,
		i.e. the negative of the slope is applied to negative frequencies.
		
		The multiplier is calculated as follows:
			M = 10**(m/20 * f/1e9)
		and then the FFT result is multiplied per-point using the frequency
		points where the spectrum is sampled.
		
		Additional magnitude slopes are multiplicative (additive in dB).
		"""
		
		if (m == None):
			pass
		
		if (self.frequency_magnitude_slope == None):
			self._frequency_magnitude_slope = m
		else:
			self._frequency_magnitude_slope = self.frequency_magnitude_slope + m
	
	def apply_frequency_phase_slope(self,p):
		"""
		Apply a phase slope to the analog signal in the frequency domain.
		
		Arguments:
		p -- The slope in units Hz^-1
		
		Notes:
		The slope is applied uniformly over the frequency band, i.e.
		the same slope is used for both positive and negative frequencies.
		
		Given the unit of p the per-sample is adjusted by multiplying with
		exp(1j*2*pi*f * p )
		
		Additional phase slopes are multiplicative.
		"""
		
		if (p == None):
			pass
		
		if (self.frequency_phase_slope == None):
			self._frequency_phase_slope = p
		else:
			self._frequency_phase_slope = self.frequency_phase_slope * p

# end class TransformedAnalogSignal

class CompoundAnalogSignal(TransformedAnalogSignal):
	"""
	Represent a summation of (optionally transformed) analog signals.
	
	"""
	
	@property
	def components(self):
		"""
		Return the analog signals that comprise this compound signal as 
		a list of AnalogSignal instances.
		
		"""
		
		return self._components
	
	def __init__(self,signals):
		"""
		Construct a compound analog signal from the given list of signals.
		
		Arguments:
		signals -- List of AnalogSignal instances.
		
		Notes:
		The signal components are internally stored as a list of 
		TransformedAnalogSignal instances so that transformations can
		easily be applied to the compound signal.
		
		The components added are copies, so that the original signals
		are unaltered by transformations applied to the compound signal.
		"""
		
		self._components = list()
		for s in signals:
			if (isinstance(s,CompoundAnalogSignal)):
				# If compound signal in the list, just add its components
				# individually
				for t in s.components:
					self._components.append(TransformedAnalogSignal(t))
				
				# and continue to the next element in signals
				continue
			else:
				# Just add this signal
				self._components.append(TransformedAnalogSignal(s))

	def sample(self,r,n,t):
		"""
		Sample the compound analog signal.
		
		Arguments:
		r -- Sample rate in samples per second.
		n -- Number of samples to generate.
		t -- Time offset of first signal.
		
		Notes:
		The sampling is done one each signal component, and the result
		accumulated and returned.
		
		See TransformedAnalogSignal.sample for more information.
		"""
		
		# initialize result to the correct amount of zero samples
		result = np.zeros(n)
		for c in self.components:
			result += c.sample(r,n,t)

		return result
	
	def apply_delay(self,d):
		"""
		Apply a delay to the compound analog signal.
		
		Arguments:
		d -- The delay given in seconds, which can be negative.
		
		Notes:
		Delays are applied to each component signal individually.
		"""
		
		for c in self.components:
			c.apply_delay(d)

	def apply_gain(self,g):
		"""
		Apply a constant gain to the compound analog signal.
		
		Arguments:
		g -- The flat gain that is applied to all signal samples.
		
		Notes:
		Gains are applied to each component signal individually.
		"""
		
		for c in self.components:
			c.apply_gain(g)
	
	def apply_frequency_magnitude_slope(self,m):
		"""
		Apply a magnitude slope to the compound analog signal in the frequency domain.
		
		Arguments:
		m -- The slope in units dB/GHz
		
		Notes:
		Frequency magnitude slopes are applied to each component signal
		individually.
		
		See method in TransformedAnalogSignal for more information.
		"""
		
		if (m == None):
			pass
		
		for c in self.components:
			c.apply_frequency_magnitude_slope(m)
	
	def apply_frequency_phase_slope(self,p):
		"""
		Apply a phase slope to the compound analog signal in the frequency domain.
		
		Arguments:
		p -- The slope in units Hz^-1
		
		Notes:
		Frequency phase slopes are applied to each component signal
		individually.
		
		See method in TransformedAnalogSignal for more information.
		"""
		
		if (p == None):
			pass
		
		for c in self.components:
			c.apply_frequency_phase_slope(p)

# end class CompoundAnalogSignal


class Generator(object):
	"""
	This is the baseclass for all signal generators.
	
	Generator defines a parameterized signal generator and can be used
	to define signal generator methods used in AnalogSignal objects.
	"""
	
	def __init__(self):
		"""
		Construct a signal generator.
		
		This implementation is empty, derived classes should define
		the class behaviour through a number of arguments.
		"""
		
		pass
	
	def generate(self,r,n,t):
		"""
		Generate signal samples for the given sampling characteristics.
		
		Arguments:
		r -- Sample rate in samples per second.
		n -- Number of samples.
		t -- Time offset for the first sample.
		
		Notes:
		This method is called in AnalogSignal.sample() when signal 
		samples are requested.
		"""
		
		return np.zeros(n)

	def get_time_vector(self,r,n,t):
		"""
		Return the time-vector for given sampling characteristics.
		
		Arguments:
		r -- Sample rate in samples per second.
		n -- Number of samples.
		t -- Time offset for the first sample.
		"""
		
		delta_t = 1.0/r
		t_stop = n*delta_t
		
		return np.arange(0,t_stop,delta_t) + t

# end class Generator


class ConstantGenerator(Generator):
	"""
	Generator for a constant signal.
	
	"""
	
	def __init__(self,amplitude=1.0):
		"""
		Construct a constant signal generator with the given amplitude.
		
		Keyword arguments:
		amplitude -- The constant signal amplitude.
		
		Notes:
		The samples generated are simply an amplitude-scaled array of
		ones.
		"""
		
		self._amplitude = amplitude
	
	def generate(self,r,n,t):
		"""
		Generate samples for a constant signal.
		
		See the constructor method for signal parameters, and baseclass
		generate method for more information.
		"""
		
		return self.amplitude * np.ones(n)
		
	
	@property
	def amplitude(self):
		"""
		Return the amplitude of the signal.
		
		"""
		return self._amplitude

# end class ConstantGenerator


class SinusoidGenerator(Generator):
	"""
	Generator for a sinusoidal signal.
	
	"""
	
	def __init__(self,amplitude=1.0,frequency=1.0,phase=0.0):
		"""
		Construct a sinusoidal signal with the given characteristics.
		
		Keyword arguments:
		amplitude -- The sine wave amplitude.
		frequency -- The sine wave frequency, in cycles per second.
		phase -- The sine wave phase, in radians.
		
		Notes:
		For a given time-discretization t the signal generated is
		amplitude * sin(2*pi*frequency*t + phase).
		"""
		
		self._amplitude = amplitude
		self._frequency = frequency
		self._phase = phase
	
	def generate(self,r,n,t):
		"""
		Generate samples for a sinusoid signal.
		
		See the constructor method for signal parameters, and baseclass
		generate method for more information.
		"""
		
		tvec = self.get_time_vector(r,n,t)
		return self.amplitude * np.sin(2.0*pi*self.frequency*tvec + self.phase)
	
	@property
	def amplitude(self):
		"""
		Return amplitude for the signal.
		
		"""
		
		return self._amplitude

	@property
	def frequency(self):
		"""
		Return frequency for the signal.
		
		"""
		
		return self._frequency

	@property
	def phase(self):
		"""
		Return phase for the signal.
		
		"""
		
		return self._phase

# end class SinusoidGenerator

class GaussianNoiseGenerator(Generator):
	"""
	Generator for a gaussian noise signal.
	
	"""
	
	# Keeps a list of all the seeds used so far to reseed the random
	# number generator with each new instance of a GaussianNoiseGenerator
	# object. The list contains unsigned integers and new seeds are
	# added with increments of 2. This is to allow both positive and
	# negative time delays in the signal by using even-numbered seeds for
	# positive time sampling and odd-numbered seeds for negative time 
	# sampling.
	_seed_list = list([0])

	def __init__(self,mean=0.0,variance=1.0):
		"""
		Construct a gaussian noise signal with the given characteristics.
		
		Keyword arguments:
		mean -- Signal mean to pass to the random generator.
		variance -- Signal variance to pass to the random generator.
		
		Notes:
		The statistical properties are only pass to the random number
		generator and when calculated for the provided signal they may
		not match perfectly.
		
		On instanciation a unique random generator seed is assigned so
		that multiple samples of the same signal can be obtained by
		setting the random generator seed accordingly. This means that
		calling the generate method may impact on other code that uses
		numpy.random.
		"""
		
		# Create new seed and add to the list.
		this_seed = self._seed_list[-1]
		self._seed_list.append(this_seed+2)
		
		# Assign this instance seed
		self._seed = this_seed
		
		self._mean = mean
		self._variance = variance
	
	def generate(self,r,n,t):
		"""
		Generate samples for a gaussian noise signal.
		
		See the constructor method for signal parameters, and baseclass
		generate method for more information.
		
		Notes:
		Currently only integer multiples of the sample period are 
		supported, in either positive or negative directions.
		"""
		
		# get time-vector
		tvec = self.get_time_vector(r,n,t)
		#print "Time-vector = ", tvec
		# calculate sample count bounds
		s_min = int(np.floor(np.min(tvec)*r))
		s_max = int(np.ceil(np.max(tvec)*r))
		#print "Signal of interest between samples (",s_min,", ",s_max, ")"
		
		# this takes care of integer-multiples of sample period
		if (s_min < 0):
			np.random.seed(self.seed['neg'])
			# note that negative-time samples are flipud'ed
			samples_neg = np.flipud(np.random.randn(-s_min))
			if (s_max < 0):
				samples_neg = samples_neg[(-s_max-1):(-s_min)]
		else:
			samples_neg = np.zeros(0)
		
		if (s_max >= 0):
			np.random.seed(self.seed['pos'])
			samples_pos = np.random.randn(s_max+1)
			if (s_min >= 0):
				samples_pos = samples_pos[s_min:s_max+1]
		else:
			samples_pos = np.zeros(0)
		
		# concatenate positive and negative parts
		samples_all = np.concatenate((samples_neg,samples_pos))
		
		# Adjustment for fractional sample period delays if needed. One
		# way to check is if there are more samples than elements in the
		# time-vector. Fractional delays are handled via FFT.
		if (samples_all.size > tvec.size):
			#print "Mismatch in vector sizes"
			samples_fft = np.fft.fftshift(np.fft.fft(samples_all));
			delta_t = (tvec[0]-1.0*s_min/r)
			fmax = r/2.0
			fstep = 1.0*r/samples_all.size # note that n is not the correct denominator
			fvec = np.arange(-fmax,fmax,fstep)
			samples_fft = samples_fft * np.exp(1j*2.0*pi*fvec*delta_t)
			samples_all = np.fft.ifft(np.fft.ifftshift(samples_fft)).real
			# Due to rounding errors the length of samples_all may be more
			# than one element longer than that of tvec. In this case we
			# need to select the correct subset of elements from samples_all.
			# This is easily done by realizing that the fractional delay
			# is a shift in sampling to the right, so that the correct starting
			# point would be the one corresponding to the first original 
			# sampling point immediately to the left of tvec[0]. 
			idx_start = np.argwhere(tvec[0] - np.arange(s_min,s_max)/r >= 0.0)[0][-1]
			samples_all = samples_all[idx_start:(idx_start+tvec.size)] # samples_all = samples_all[0:-1]
			#raise RuntimeError("Break")
		
		# finally, adjust statistics
		samples_all *= np.sqrt(self.variance)
		samples_all += self.mean
		
		return samples_all

	@property
	def seed(self):
		"""
		Return the random generator seeds used for this instance.
		
		The returned object is a dictionary in which the entry 'pos' 
		is the seed used for positive-time random samples, and 'neg' is
		the seed used for negative-time random samples.
		"""
		
		seed_dict = {'pos': self._seed, 'neg': self._seed+1}
		
		return seed_dict

	@property
	def mean(self):
		"""
		Return the mean of this gaussian noise generator.
		
		"""
		
		return self._mean

	@property
	def variance(self):
		"""
		Return the variance of this gaussian noise generator.
		
		"""
		
		return self._variance

# end class GaussianNoiseGenerator


class DigitalSignal(Signal):
	"""
	Baseclass for all digital signals.
	
	DigitalSignal is characterized by a sample rate, precision defined
	by FixedWidthBinary.WordFormat instance, and the signal samples stored 
	as a FixedWidthBinary.Word instance.
	"""
	
	def __init__(self,rate,precision,svec):
		"""
		Construct a digital signal for the given rate, precision and samples.
		
		Arguments:
		rate -- Sampling rate in samples per second.
		precision -- FixedWidthType that defines the binary representation
		of signal samples
		svec -- An array of sampled values.
		
		Notes:
		svec should be real. If complex svec is given, the imaginary component
		is discarded by casting to real numeric type.
		"""
		
		self._sample_rate = rate
		self._precision = precision
		self._samples_word = fw.Word(svec,precision)
	
	@property
	def sample_rate(self):
		"""
		Return the sample rate for this signal.
		
		"""
		
		return self._sample_rate
	
	@property
	def precision(self):
		"""
		Return the precision as a FixedWidthBinary.WordFormat instance.
		"""
		
		return self._precision
	
	@property
	def samples(self):
		"""
		Return the signal samples as a numpy array.
		
		"""
		
		return self._samples_word.value
	
	@property
	def samples_word(self):
		"""
		Return the signal samples as a FixedWidthBinary.Word.
		
		"""
		
		return self._samples_word
	
	@property
	def number_of_samples(self):
		"""
		Return the number of signal samples.
		
		"""
		
		return self._samples_word.value.size

# end class DigitalSignal


class DigitalSignalComplex(DigitalSignal):
	"""
	Represent a complex digital signal.
	
	All methods except for the constructor are inherited as-is. The 
	samples are stored as FixedWidthBinary.WordComplex internally and the 
	complex nature of the signal is handled by that class.
	
	Apart from the constructor, all other class attributes are inherited
	as-is from DigitalSignal.
	"""
	
	def __init__(self,rate,precision,svec):
		"""
		Construct a digital signal for the given rate, precision and samples.
		
		Arguments:
		rate -- Sampling rate in samples per second.
		precision -- FixedWidthBinary.WordFormat that defines the binary 
		representation of signal samples
		svec -- An array of sampled values.
		
		Notes:
		svec can be real or complex, if real then zero imaginary part is
		assumed.
		
		"""
		
		self._sample_rate = rate
		self._precision = precision
		self._samples_word = fw.WordComplex(svec,precision)

# end class DigitalSignalComplex
