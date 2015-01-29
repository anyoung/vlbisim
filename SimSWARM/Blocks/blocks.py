#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  blocks.py
#  Jan 20, 2015 11:02:30 EST
#  Copyright 2015
#         
#  Andre Young <andre.young@cfa.harvard.edu>
#  Harvard-Smithsonian Center for Astrophysics
#  60 Garden Street, Cambridge
#  MA 02138
#  
#  Changelog:
#  	AY: Created 2015-01-20
#	AY: Changed frequency magnitude slope block to dB/GHz 2015-01-27
#	AY: Added Parallel class 2015-01-27
#	AY: Added DigitalFFT class 2015-01-27

"""
Defines various fundamental signal processing blocks.

"""

import SimSWARM.Signal as sg
import FixedWidthBinary as fw

import copy
import numpy as np

pi = np.pi

class Block(object):
	"""
	Baseclass for all signal processing blocks.
	
	"""
	
	@property
	def source(self):
		"""
		Return the source attached to this block.
		"""
		
		return self._source

	def __init__(self):
		"""
		Construct Block instance.
		
		This implementation is empty, derived classes should define
		a number of parameters that define the behaviour of this block.
		"""
		
		self._source = None
	
	def attach_source(self,src):
		"""
		Attach an input to this block.
		
		Arguments:
		src -- Can either be a Signal or Block instance.
		
		Notes:
		If src is a Block instance, then output() will call the output
		method of src.
		"""
		
		self._source = src
	
	def output(self):
		"""
		Generate and return the output of the block for the defined input.
		
		This method should always return a Signal instance. This specific
		implementation just returns the input.
		"""
		
		#~ # check the input is a Signal instance
		#~ if (not isinstance(s_in,sg.Signal)):
			#~ raise ValueError("Input to Block should be a Signal instance.")
		#~ 
		#~ # make default behaviour to just create and return a copy
		#~ self._output = sg.Signal.copy(s_in)
		#~ 
		#~ return self._output 
		
		src = self.source
		if (src == None):
			raise RuntimeError("No input defined for this Block; cannot generate output.")
		
		if (isinstance(src,Block)):
			src = src.output()
		
		# src should always reduce to some instance of Signal
		if (isinstance(src,sg.Signal)):
			return src
		
		raise RuntimeError("Input is not reducable to Signal instance.")
	
	@classmethod
	def copy(cls,b):
		"""
		Construct and return a copy of the given block.
		
		Arguments:
		b -- Block instance to copy.
		
		Notes:
		Current implementation uses a deep copy.
		"""
		
		return copy.deepcopy(b)

# end class Block

class Parallel(Block):
	"""
	Baseclass for representing parallel processes.
	
	"""
	
	@property
	def blocks(self):
		"""
		Return the list of parallel blocks.
		"""
		
		return self._blocks
	
	def __init__(self,parblock,n=None):
		"""
		Construct a parallel signal processing path.
		
		Arguments:
		parblock -- A Block that defines the processing that occurs in 
		each parallel path.
		
		Keyword Arguments:
		n -- The number of parallel paths.
		
		Notes:
		If n == None then parblock is assumed to be a list of Block 
		instances, and a separate parallel signal path is created using
		each of the items in the list. If n is an integer then Block is
		assumed to be a single Block instance, which is copied n times
		to create the separate signal paths.
		"""
		
		if (n == None):
			self._blocks = parblock
		else:
			self._blocks = list()
			for ii in range(0,n):
				self._blocks.append(Block.copy(parblock))
		

	def attach_source(self,src):
		"""
		Attach a source to this parallel processing path.
		
		Arguments:
		src -- The input source to the processing paths, either list
		of Signal/Block or a single Signal/Block.
		
		Notes:
		For src a list of objects, each item in the list is attached as
		a source to each item in the list of parallel blocks. For a single
		object, if it is a Parallel instance, then each of its parallel
		blocks is used as input for each of this Parallel's blocks. For
		any other single object instance, src is copied and attached
		separately for each parallel block.
		
		"""
		
		if (isinstance(src,list)):
			for ii in range(0,len(src)):
				self.blocks[ii].attach_source(src[ii])
		elif (isinstance(src,Parallel)):
			src = src.blocks
			for ii in range(0,len(src)):
				self.blocks[ii].attach_source(src[ii])
		elif (isinstance(src,Block)):
			for iblock in self.blocks:
				iblock.attach_source(Block.copy(src))
		elif (isinstance(src,sg.Signal)):
			for iblock in self.blocks:
				iblock.attach_source(sg.Signal.copy(src))
		else:
			raise TypeError("Invalid type for source to attach to Parallel.")
	
	def output(self):
		"""
		Return the output signals for the parallel paths.
		
		Notes:
		The returned value is a list containing the output method result
		of each of the parallel blocks individually.
		"""
		
		out = list()
		for iblock in self.blocks:
			out.append(iblock.output())
		
		return out
		
# end class Parallel


class Chain(Block):
	"""
	Baseclass for a serial execution of signal processing blocks.
	
	"""
	
	@property
	def source(self):
		"""
		Return the input source for this chain.
		
		The input source in this case is that of the first block in 
		the chain.
		"""
		
		return self.blocks[0].source
	
	@property
	def blocks(self):
		"""
		Return a list of blocks that comprise this signal chain.
		"""
		
		return self._blocks
	
	def __init__(self):
		"""
		Construct a signal processing chain.
		"""
		
		self._blocks = list()
	
	def attach_source(self,src):
		"""
		Attach source for the first block in this chain.
		
		Arguments:
		src -- A valid block input source to attach to first block in 
		chain.
		"""
		
		self.blocks[0].attach_source(src)
	
	def add_block(self,b):
		"""
		Add the block to the signal chain.
		
		Arguments:
		b -- Block instance to be appended (at the end) of the signal 
		chain.
		"""
		
		if (not isinstance(b,Block)):
			raise ValueError("Only Block or Chain instances can be added to a Chain.")
		
		#~ if (isinstance(b,Chain)):
			#~ for link in b:
				#~ self.add_block(l)
		#~ else:
			#~ self._blocks.append(b)
		
		# set current last entry as the input to this link
		if (len(self.blocks) > 0):
			b.attach_source(self.blocks[-1])
			
		self._blocks.append(b)
		
	
	def output(self):
		"""
		Return output signal for the chain.
		
		"""
		
		#~ s_out = s_in
		#~ for b in self.blocks():
			#~ s_out = b.output(s_out)
		#~ 
		#~ return s_out
		
		# call to output of last link in chain should recursively
		# invoke output of all links, down to the first
		return self.blocks[-1].output()

# end class Chain


#~ class Converge(Block):
	#~ """
	#~ Represent a multiple-input single-output processing block.
	#~ """
	#~ 
	#~ @property
	#~ def in_blocks(self):
		#~ """
		#~ Return a list of blocks of which the outputs are used as input.
		#~ 
		#~ """
		#~ 
		#~ return self._in_blocks
	#~ 
	#~ 
	#~ def __init__(self):
		#~ """
		#~ Construct a converge block.
		#~ 
		#~ Implementation creates an empty list of input blocks. Derived
		#~ classes can further define behaviour as needed.
		#~ """
		#~ 
		#~ self._in_blocks = list()
#~ 
	#~ def add_in_block(self,b):
		#~ """
		#~ Add block to the list of input blocks.
		#~ 
		#~ Arguments:
		#~ b -- Block instance to be added.
		#~ """
		#~ 
		#~ self._in_blocks.append(b)
	#~ 
	#~ def output(self

### Define a number of analog blocks

class AnalogDelay(Block):
	"""
	Define a delay along the propagation path of an analog signal.
	
	"""
	
	@property
	def delay(self):
		"""
		Return the delay applied by the block.
		
		"""
		
		return self._delay
	
	def __init__(self,d):
		"""
		Construct an analog delay block.
		
		Arguments:
		d -- The delay in seconds.
		"""
		
		self._delay = d
	
	def output(self):
		"""
		Return a transformed analog signal with the corresponding delay.
		
		"""
		s_in = self.source
		
		if (isinstance(s_in,Block)):
			s_in = s_in.output()
		
		if (not isinstance(s_in, sg.AnalogSignal)):
			raise ValueError("AnalogDelay can only operate on instances of AnalogSignal or derivative classes.")
		
		# Create output signal, which will be TransformedAnalogSignal instance.
		if (not isinstance(s_in, sg.TransformedAnalogSignal)):
			# If not TransformedAnalogSignal, make one
			s_out = sg.TransformedAnalogSignal(s_in)
		else:
			# If TransformedAnalogSignal, make a copy. This preserves
			# transformations already applied, and handles a CompoundAnalogSignal
			# instance correctly.
			s_out = sg.Signal.copy(s_in)
		
		# Apply the effect to the output signal
		s_out.apply_delay(self.delay)
		
		return s_out

# end class AnalogDelay

class AnalogGain(Block):
	"""
	Define a gain along the propagation path of an analog signal
	
	"""
	
	@property
	def gain(self):
		"""
		Return the gain that is applied by the block.
		
		"""
		
		return self._gain
	
	def __init__(self,g):
		"""
		Construct an analog gain block.
		
		Arguments:
		g -- The gain, which is dimensionless.
		"""
		
		self._gain = g
	
	def output(self):
		"""
		Return a transformed analog signal with the corresponding gain.
		
		"""
		s_in = self.source
		
		if (isinstance(s_in,Block)):
			s_in = s_in.output()
		
		if (not isinstance(s_in, sg.AnalogSignal)):
			raise ValueError("AnalogGain can only operate on instances of AnalogSignal or derivative classes.")
		
		# Create output signal, which will be TransformedAnalogSignal instance.
		if (not isinstance(s_in, sg.TransformedAnalogSignal)):
			# If not TransformedAnalogSignal, make one
			s_out = sg.TransformedAnalogSignal(s_in)
		else:
			# If TransformedAnalogSignal, make a copy. This preserves
			# transformations already applied, and handles a CompoundAnalogSignal
			# instance correctly.
			s_out = sg.Signal.copy(s_in)
		
		# Apply the effect to the output signal
		s_out.apply_gain(self.gain)
		
		return s_out

# end class AnalogGain

class AnalogFrequencyGainSlope(Block):
	"""
	Define a gain slope applied to an analog signal in the frequency domain.
	
	"""
	
	@property
	def slope(self):
		"""
		Return the gain slope that is applied by the block.
		
		"""
		
		return self._slope
	
	def __init__(self,m):
		"""
		Construct an analog spectrum gain slope block.
		
		Arguments:
		m -- The gain slope, in units dB/GHz.
		
		Notes:
		See SimSWARM.Signal.TransformedAnalogSignal.apply_frequency_magnitude_slope 
		for additional information.
		"""
		
		self._slope = m
	
	def output(self):
		"""
		Return a transformed analog signal with the corresponding gain.
		
		"""
		s_in = self.source
		
		if (isinstance(s_in,Block)):
			s_in = s_in.output()
		
		if (not isinstance(s_in, sg.AnalogSignal)):
			raise ValueError("AnalogFrequencyGainSlope can only operate on instances of AnalogSignal or derivative classes.")
		
		# Create output signal, which will be TransformedAnalogSignal instance.
		if (not isinstance(s_in, sg.TransformedAnalogSignal)):
			# If not TransformedAnalogSignal, make one
			s_out = sg.TransformedAnalogSignal(s_in)
		else:
			# If TransformedAnalogSignal, make a copy. This preserves
			# transformations already applied, and handles a CompoundAnalogSignal
			# instance correctly.
			s_out = sg.Signal.copy(s_in)
		
		# Apply the effect to the output signal
		s_out.apply_frequency_magnitude_slope(self.slope)
		
		return s_out

# end class AnalogSpectrumSlope

class AnalogFrequencyPhaseSlope(Block):
	"""
	Define a phase slope applied to an analog signal in the frequency domain.
	
	"""
	
	@property
	def slope(self):
		"""
		Return the phase slope that is applied by the block.
		
		"""
		
		return self._slope
	
	def __init__(self,p):
		"""
		Construct an analog spectrum phase slope block.
		
		Arguments:
		p -- The phase slope, in units Hz^-1.
		
		Notes:
		See SimSWARM.Signal.TransformedAnalogSignal.apply_frequency_phase_slope 
		for additional information.
		"""
		
		self._slope = p
	
	def output(self):
		"""
		Return a transformed analog signal with the corresponding gain.
		
		"""
		s_in = self.source
		
		if (isinstance(s_in,Block)):
			s_in = s_in.output()
		
		if (not isinstance(s_in, sg.AnalogSignal)):
			raise ValueError("AnalogFrequencyPhaseSlope can only operate on instances of AnalogSignal or derivative classes.")
		
		# Create output signal, which will be TransformedAnalogSignal instance.
		if (not isinstance(s_in, sg.TransformedAnalogSignal)):
			# If not TransformedAnalogSignal, make one
			s_out = sg.TransformedAnalogSignal(s_in)
		else:
			# If TransformedAnalogSignal, make a copy. This preserves
			# transformations already applied, and handles a CompoundAnalogSignal
			# instance correctly.
			s_out = sg.Signal.copy(s_in)
		
		# Apply the effect to the output signal
		s_out.apply_frequency_phase_slope(self.slope)
		
		return s_out

# end class AnalogFrequencyPhaseSlope


class AnalogCombiner(Block):
	"""
	Combines a number of analog signals by summation.
	
	The constructor is inherited as-is from Block, which is essentially 
	empty.
	"""
	
	def attach_source(self,s_in_list):
		"""
		Attach a list of input sources to this block.
		
		Arguments:
		s_in_list -- A list of valid Block input instances, i.e. either
		Block or Signal instances, or a Parallel instance.
		
		Notes:
		If s_in_list is a Parallel instance, that class' blocks attribute
		is accessed and the returned list of Block instances used as the
		source to attach, as if s_in_list was the same list of Block instances.
		"""
		
		if (isinstance(s_in_list,Parallel)):
			s_in_list = s_in_list.blocks
		
		if (not isinstance(s_in_list,list)):
			raise ValueError("Source for AnalogCombiner should be a list of Block and/or Signal instances.")
		
		super(AnalogCombiner,self).attach_source(s_in_list)
	
	def output(self):
		"""
		Return a combined analog signal from the given list of signals.

		"""
		
		s_list_in = list()
		for src in self.source:
			if (isinstance(src,Block)):
				src = src.output()
			
			if (not isinstance(src,sg.AnalogSignal)):
				raise RuntimeError("AnalogCombiner can only operate on AnalogSignal inputs.")
			
			s_list_in.append(src)
		
		return sg.CompoundAnalogSignal(s_list_in)

# end class AnalogCombiner

### End of analog blocks

### Define analog-to-digital blocks

class AnalogDigitalConverter(Block):
	"""
	Represent an ADC.
	
	"""
	
	@property
	def sample_rate(self):
		"""
		Return the sample rate for the ADC
		
		"""
		
		return self._sample_rate
	
	@property
	def number_of_samples(self):
		"""
		Return the number of samples collected by ADC.
		
		"""
		
		return self._number_of_samples
	
	@property
	def precision(self):
		"""
		Return the precision for the amplitude discretization.
		
		"""
		
		return self._precision
	
	def __init__(self,rate,length,precision):
		"""
		Construct an ADC characterized by the given sampling characteristics.
		
		Arguments:
		rate -- Sampling rate in samples per second
		length -- The number of samples to acquire
		precision -- Defines the amplitude discretization as a FixedWidthType
		instance.
		"""
		
		self._sample_rate = rate
		self._number_of_samples = length
		self._precision = precision
	
	def output(self):
		"""
		Return the digital output for the given analog input.
		
		Notes:
		The amplitude discretization is hanled by the FixedWidthNum 
		constructor method. That method raises an exception if values
		passed to it fall outside the range of values representable in 
		the given FixedWidthType format. This implementation catches
		the exception and then applies a hard limit so that all values
		are within the representable range.
		
		"""
		s_in = self.source
		
		if (isinstance(s_in,Block)):
			s_in = s_in.output()
		
		if (not isinstance(s_in,sg.AnalogSignal)):
			raise ValueError("Input to AnalogDigitalConverter should be an AnalogSignal instance.")
		
		# get machine precision samples of the analog input
		svec = s_in.sample(self.sample_rate,self.number_of_samples,0.0) # note: zero time-delay in sampling
		
		# Apply amplitude discretization. The constructor of a FixedWithNumber
		# may raise an error if the given values fall outside the range 
		# of values representable in the given format. In that case the 
		# ADC simply saturates to the nearest bound.
		try:
			svec_digital = fw.Word(svec,self.precision).value
		except fw.OverflowError:
			max_val = self.precision.maximum_value
			min_val = self.precision.minimum_value
			svec[svec > max_val] = max_val
			svec[svec < min_val] = min_val
			svec_digital = fw.Word(svec,self.precision).value
		
		return sg.DigitalSignal(self.sample_rate,self.precision,svec_digital)

### End analog-to-digital blocks

### Define a number of digital blocks

class Requantizer(Block):
	"""
	Define a requantization block.
	
	"""
	
	@property
	def precision(self):
		"""
		Return the precision used for this requantization.
		"""
		
		return self._precision
	
	def __init__(self,precision):
		"""
		Construct a requantizer block for the given precision.
		
		Arguments:
		precision -- FixedWidthType that defines the amplitude 
		discretization used in requantization.
		"""
		
		self._precision = precision
	
	def output(self):
		"""
		Return the output after requantization of the input source.
		
		Notes:
		Implementation for handling overflows is similar to the AnalogDigitalConverter
		block. See output method for that class for information.
		
		"""
		
		s_in = self.source
		
		if (isinstance(s_in,Block)):
			s_in = s_in.output()
		
		if (not isinstance(s_in,sg.DigitalSignal)):
			raise ValueError("Input to Requantize should be a DigitalSignal instance.")
		
		# extract samples from input signal
		svec = s_in.samples
		# get word type, either Word or WordComplex
		word_type = type(s_in.samples_word)
		
		try:
			svec_digital = word_type(svec,self.precision).value
		except fw.OverflowError:
			max_val = self.precision.maximum_value
			min_val = self.precision.minimum_value
			if (word_type == fw.WordComplex):
				svec[svec.real > max_val] = max_val + svec[svec.real > max_val].imag*1j
				svec[svec.real < min_val] = min_val + svec[svec.real < min_val].imag*1j
				svec[svec.imag > max_val] = max_val*1j + svec[svec.imag > max_val].real
				svec[svec.imag < min_val] = min_val*1j + svec[svec.imag < min_val].real
			else:
				svec[svec > max_val] = max_val
				svec[svec < min_val] = min_val
			
			svec_digital = word_type(svec,self.precision).value
		
		return sg.DigitalSignal(s_in.sample_rate,self.precision,svec_digital)

# end class Requantizer


class DigitalGain(Block):
	"""
	Apply a gain to a digital signal.
	
	"""
	
	@property
	def gain_word(self):
		"""
		Return the gain value as a FixedWidthBinary.Word.
		
		"""
		
		return self._gain_word
	
	@property
	def gain(self):
		"""
		Return the gain value.
		
		"""
		
		return self.gain_word.value

	def __init__(self,g,precision):
		"""
		Construct a digital gain block.
		
		Arguments:
		g -- The gain to be applied.
		precision -- The WordFormat instance that defines the binary
		representation of the gain parameters.
		
		Notes:
		The gain is stored internally as a FixedWidthBinary.Word (or 
		.WordComplex if g is complex-valued) the precision passed to the 
		constructor. Accessing the stored gain value will therefore be 
		limited by the precision.
		
		The gain can either be scalar or per-sample (if g is a vector
		with as many elements as samples are expected from the attached
		source).
		
		"""
		
		if (np.iscomplexobj(g)):
			self._gain_word = fw.WordComplex(g,precision)
		else:
			print ".?"
			self._gain_word = fw.Word(g,precision)
			print ".!"
	
	def output(self):
		"""
		Return the output after applying the gain to the input source.
		
		Notes:
		Implementation is based on the __mul__ method override in 
		FixedWidthBinary.Word, which automatically adjusts the result
		word format according to that of the inputs.  See the documentation
		for that method for more information.
		
		"""
		
		s_in = self.source
		
		if (isinstance(s_in,Block)):
			s_in = s_in.output()
		
		if (not isinstance(s_in,sg.DigitalSignal)):
			raise ValueError("Input to Requantize should be a DigitalSignal instance.")
		
		# do multiplication using Word instances
		out_word = s_in.samples_word * self.gain_word
	
		return sg.DigitalSignal(s_in.sample_rate,out_word.word_format,out_word.value)

# end class DigitalGain

class DigitalCombiner(Block):
	"""
	Combines a number of digital signals by summation.
	
	The constructor is inherited as-is from Block, which is essentially 
	empty.
	"""
	
	def attach_source(self,s_in_list):
		"""
		Attach a list of input sources to this block.
		
		Arguments:
		s_in_list -- A list of valid Block input instances, i.e. either
		Block or Signal instances, or a single Parallel instance.
		
		Notes:
		If s_in_list is a Parallel instance, then that class' blocks
		attribute is accessed to return a list of Blocks, which is then
		used as the list of sources to attach.
		"""
		
		if (isinstance(s_in_list,Parallel)):
			s_in_list = s_in_list.blocks
		
		if (not isinstance(s_in_list,list)):
			raise ValueError("Source for DigitalCombiner should be a list of Block and/or Signal instances.")
		
		super(DigitalCombiner,self).attach_source(s_in_list)
	
	def output(self):
		"""
		Return a digital signal which is the sum of the input signals.
		
		Notes:
		The addition operation is handled by the FixedWidthBinary.Word
		override of __add__ which automatically handles the output 
		WordFormat.
		
		"""
		
		result = None
		sample_rate = None
		for src in self.source:
			if (isinstance(src,Block)):
				src = src.output()
			
			if (not isinstance(src,sg.DigitalSignal)):
				raise RuntimeError("DigitalCombiner can only operate on DigitalSignal inputs.")
				
			if (sample_rate == None):
				sample_rate = src.sample_rate
			else:
				if (sample_rate != src.sample_rate):
					raise RuntimeError("Sample rates should match at DigitalCombiner input.")
			
			if (result == None):
				result = src.samples_word
			else:
				result = result + src.samples_word
		
		return sg.DigitalSignal(sample_rate,result.word_format,result.value)

# end class DigitalCombiner

class DigitalFFT(Block):
	"""
	Implements a digital FFT.
	
	"""
	
	@property
	def precision(self):
		"""
		Return the precision used for representing Fourier coefficients.
		"""
		
		return self._precision
	
	def __init__(self,precision):
		"""
		Create an FFT block.
		
		Arguments:
		precision -- The FixedWidthBinary.WordFormat instance that defines
		the binary representation of the Fourier coefficients.
		"""
		
		self._precision = precision
	
	def output(self):
		
		rate,result = self._fft_routine()
		
		return sg.DigitalSignal(rate,result.word_format,result.value,force_complex=True)
	
	def _fft_routine(self):
		"""
		Internal method that performs the FFT on the source signal.
		
		Notes:
		Raises an error when the number of samples is not a whole-numbered
		power of two.
		"""
		
		src = self.source
		
		if (isinstance(src,Block)):
			src = src.output()
		
		if (not isinstance(src,sg.DigitalSignal)):
			raise RuntimeError("FFTBlock can only operate on DigitalSignal instance.")
		
		N = src.number_of_samples
		log2_N = np.log2(N)
		if ( (log2_N - int(log2_N)) != 0.0 ):
			raise RuntimeError("Only radix-2 implemented, number of samples should be whole-numbered power of two.")
		
		# represent samples in the output WordFormat
		samples_word = fw.WordComplex(src.samples,self.precision)
		samples_fft_word = self._recursive_dft(samples_word,N)
		
		return (src.sample_rate,samples_fft_word)
	
	def _recursive_dft(self,full,N):
		"""
		Compute the N-point DFT using two N/2-point DFTs.
		
		Arguments:
		full -- The input vector, given as FixedWidthBinary.Word
		N -- Number of elements in the input vector.
		
		Notes:
		The result is returned as a FixedWidthBinary.Word. The automatic
		bit-growth for binary operations on Word instances is ignored in
		the current implementation. Instead, the calculations are performed
		on the value attributes of the Words involved, and the result
		used to create a Word instance using the same WordFormat of full.
		
		"""
		
		if (N == 1):
			return fw.WordComplex(full.value,full.word_format)
		else:
			# divide into two N/2-point DFTs
			even_half = fw.WordComplex(full.value[0::2],full.word_format)
			even_half = self._recursive_dft(even_half,N/2)
			odd_half = fw.WordComplex(full.value[1::2],full.word_format)
			odd_half = self._recursive_dft(odd_half,N/2)
			# compute twiddle factors
			twiddle_factors = self._compute_twiddle(N)
			# combine into N-point DFT
			result_lower = even_half.value + twiddle_factors.value * odd_half.value
			result_upper = even_half.value - twiddle_factors.value * odd_half.value
			return fw.WordComplex(np.concatenate((result_lower,result_upper)),full.word_format)

	def _compute_twiddle(self,N):
		"""
		Compute W_N^(n) = exp(-j*2*pi*n/N) for all n = 0, 1, ..., N/2-1
		"""
		
		nvec = np.arange(0,N/2)
		wvec = np.exp(-1j*2.0*pi*nvec/N)
		
		return fw.WordComplex(wvec,self.precision)

	def _alt_output(self):
		"""
		Alternative output method that is based on numpy's FFT implementation.
		
		This method is useful to check the impact of quantization *during*
		the FFT algorithm. Here the FFT is computed with machine precision
		using the quantized input samples, and the final result then 
		quantized according to the precision of the FFT block.
		
		"""
		
		rate,result = self._alt_fft_routine()
		
		return sg.DigitalSignal(rate,result.word_format,result.value)
	
	def _alt_fft_routine(self):
		"""
		Alternative internal computation based on numpy's FFT implementation.
		
		"""
		
		src = self.source
		
		if (isinstance(src,Block)):
			src = src.output()
		
		if (not isinstance(src,sg.DigitalSignal)):
			raise RuntimeError("FFTBlock can only operate on DigitalSignal instance.")
		
		samples_fft = np.fft.fft(src.samples)
		
		return (src.sample_rate,fw.WordComplex(samples_fft,self.precision))

# end class DigitalFFT

### End digital blocks
