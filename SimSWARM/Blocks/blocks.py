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

"""
Defines various fundamental signal processing blocks.

"""

import SimSWARM.Signal as sg
import FixedWidthBinary as fw

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
	
# end class Block


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
		m -- The gain slope, in units Hz^-1.
		
		Notes:
		See SimSWARM.Signal.TransformedAnalogSignal.apply_frequency_magnitude_slope 
		for additional information.
		"""
		
		self._slope = m
	
	def output(self,s_in):
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
		See SimSWARM.Signal.TransformedAnalogSignal.apply_frequency_magnitude_slope 
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
		Block or Signal instances.
		"""
		
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
		
		try:
			svec_digital = fw.Word(svec,self.precision).value
		except fw.OverflowError:
			max_val = self.precision.maximum_value
			min_val = self.precision.minimum_value
			svec[svec > max_val] = max_val
			svec[svec < min_val] = min_val
			svec_digital = fw.Word(svec,self.precision).value
		
		return sg.DigitalSignal(s_in.sample_rate,self.precision,svec_digital)

# end class Requantizer


class DigitalGain(Block):
	"""
	Apply a constant gain to a digital signal.
	
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
		g -- The real-valued gain to be applied.
		precision -- The WordFormat instance that defines the binary
		representation of the gain parameters.
		
		Notes:
		The gain is stored internally as a FixedWidthBinary.Word using
		the precision passed to the constructor. Accessing the stored 
		gain value will therefore be limited by the precision.
		
		"""
		
		self._gain_word = fw.Word(g,precision)
	
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
		
		return DigitalSignal(out_word,out_word.word_format)

# end class DigitalMultiplier

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
		Block or Signal instances.
		"""
		
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
		for src in self.source:
			if (isinstance(src,Block)):
				src = src.output()
			
			if (not isinstance(src,sg.DigitalSignal)):
				raise RuntimeError("DigitalCombiner can only operate on DigitalSignal inputs.")
			
			if (result == None):
				result = src
			else:
				result = result + src
		
		return result

# end class DigitalCombiner

### End digital blocks
