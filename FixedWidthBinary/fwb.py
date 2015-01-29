#!/usr/bin/env python
# -*- coding: utf-8 -*-
#       
#       fwb.py
#       Jan 7, 2015 12:25:15 EST
#       
#       Andre Young <andre.young@cfa.harvard.edu>
#       Harvard-Smithsonian Center for Astrophysics
#       60 Garden Street, Cambridge
#       MA 02138
#       
#       Copyright 2015
#       
"""
Provides classes for representing numbers with arbitrarily wide binary words.

The representation is based on Two's Complement.
"""

import numpy as np

class OverflowError(ValueError):
	"""
	Raised when input values are outside the range representable in the given format.
	
	"""

class WordFormat(object):
	"""
	Define a limited width binary word format.
	
	"""

	def __init__(self,width,lsb_pot):
		"""
		Construct a binary word format.
		
		Arguments:
		width -- The total number of bits in the word.
		lsb_pot -- The value of the least-significant bit as a power of 2.
		
		"""
		
		self._width = width
		self._lsb_value = 2**lsb_pot
		self._mask = 2**width - 1
		self._scaled_max = 2**(width-1) - 1
		self._scaled_min = -(2**(width-1))
	
	def value_in_range(self,val):
		"""
		Test whether value is within the representable range.
		
		Arguments:
		val -- Value to be tested.
		
		"""
		
		return ( (np.array(val) >= self.minimum_value).all() and (np.array(val) <= self.maximum_value).all() )

	def __repr__(self):
		"""
		Unambiguous string representation.
		
		"""
		
		return 'WordFormat(' + str(self.width) + '-bit, ' + str(self.lsb_value) + ' res)'

	def __str__(self):
		"""
		Friendly string representation.
		
		"""
		
		return str(self.width) + '-bit, ' + str(self.lsb_value) + ' res'

	@property
	def width(self):
		"""
		Return the total number of bits used in representation.
		
		"""
		
		return self._width

	@property
	def lsb_value(self):
		"""
		Return the value represented by the least-significant bit
		
		"""
		
		return self._lsb_value
	
	@property
	def mask(self):
		"""
		Return the bit-mask used to trim a binary word down to the correct width.
		
		"""
		
		return self._mask

	@property
	def maximum_scaled_value(self):
		"""
		Return the maximum scaled value that can be represented.
		
		"""
		
		return self._scaled_max

	@property
	def minimum_scaled_value(self):
		"""
		Return the minimum scaled value that can be represented.
		
		"""
		
		return self._scaled_min

	@property
	def maximum_value(self):
		"""
		Return the maximum value that can be represented.
		
		"""
		
		return 1.0*self.maximum_scaled_value * self.lsb_value

	@property
	def minimum_value(self):
		"""
		Return the minimum value that can be represented.
		
		"""
		
		return 1.0*self.minimum_scaled_value * self.lsb_value

# end class WordFormat

class Word(object):
	"""
	Represent a number in a limited width binary format.
	
	"""

	def __init__(self,val,fmt):
		"""
		Construct a limited width binary representation of a number.
		
		Arguments:
		val -- The value to be represented.
		fmt -- A WordFormat instance that defines the binary representation 
		used.
		
		Notes:
		val can be any real-valued numeric type that is valid input
		for constructing a numpy array. It can also be an instance of
		Word, in which case its value is simply extracted.
		
		This method raises an OverflowError if the given value falls 
		outside the range of numbers that can be represented in the 
		given format.
		
		"""
		
		self._word_format = fmt
		
		# if value given as Word instance, extract underlying value
		if (isinstance(val,Word)):
			val = val.value
		
		# assign the integer representation of the number
		self._scaled_value = np.array(np.array(val) / fmt.lsb_value, dtype=np.int64)
		# test if it is in the allowable range
		if (not fmt.value_in_range(val)):
			raise OverflowError("Given value is not in the range for the given format.")
		
		# mask out all bits outside the word width
		self._scaled_value =  self._scaled_value & fmt.mask

	def __repr__(self):
		"""
		Unambiguous string representation.
		
		"""
		
		if (self.scaled_value.size < 9):
			return str(type(self)) + '(scaled value=' + str(self.scaled_value) + ', format=' + str(self.word_format) + ')'
		else:
			return str(type(self)) + '(scaled value=array' + str(self.scaled_value.shape) + ', format=' + str(self.word_format) + ')'

	def __str__(self):
		"""
		String representation.
		
		"""
		
		if (self.scaled_value.size == 1):
			return self._bitstr(self.scaled_value)
		else:
			return repr(self)

	def _bitstr(self,sval):
		"""
		Internal method used to represent sval as bit-string.
		
		Notes:
		Not implemented yet.
		
		"""
		
		#~ bits_int = sval >> self.fixed_width_type.decimal_bits
		#~ bits_int_mask = 2**self.fixed_width_type.integer_bits - 1
		#~ bits_int_str = bin(bits_int & bits_int_mask)[2:]
		#~ bits_int_str = '0' * (self.fixed_width_type.integer_bits - len(bits_int_str)) + bits_int_str
		#~ if (self.fixed_width_type.decimal_bits > 0):
			#~ bits_dec = sval & (2**self.fixed_width_type.decimal_bits-1)
			#~ bits_dec_str = bin(bits_dec)[2:]
			#~ bits_dec_str = '0' * (self.fixed_width_type.decimal_bits - len(bits_dec_str)) + bits_dec_str
		#~ else:
			#~ bits_dec_str = ''
		#~ 
		#~ return bits_int_str + '.' + bits_dec_str
		
		return "????????"

	def __neg__(self):
		"""
		Negate operator.
		
		"""
		
		new_val = -self.value
		return type(self)(new_val,self.word_format)

	def __pos__(self):
		"""
		Identity operator.
		
		"""
		
		return self

	def __add__(self,other):
		"""
		Addition operation.
		
		Notes:
		Both arguments should be Word instances.
		
		The result is returned as a Word instance, and the WordFormat
		for the result is automatically derived to have the least lsb_value
		of either argument, and to have a width equal to one more than the
		maximum width of either argument.
		
		The exact type of the result is determined by the further decendent
		of Word among the arguments.
		
		"""
		
		if (not isinstance(other,Word)):
			raise TypeError("Addition only defined for Word instances.")
		
		# the smallest lsb_value should be preserved
		new_lsb_value = min(self.word_format.lsb_value, other.word_format.lsb_value)
		new_lsb_pot = int(np.log2(new_lsb_value))
		
		# the width should be the largest width + 1
		new_width = max(self.word_format.width, other.word_format.width) + 1
		
		new_fmt = WordFormat(new_width,new_lsb_pot)
		new_val = self.value + other.value
		
		# check which type to use for result
		type_other = type(other)
		type_self = type(self)
		if (isinstance(other,type_self)):
			return type_other(new_val,new_fmt)
		else:
			return type_self(new_val,new_fmt)

	def __sub__(self,other):
		"""
		Subtraction operation.
		
		Notes:
		Both arguments should be Word instances.
		
		The result is returned as a Word instance, and the WordFormat
		for the result is automatically derived to have the least lsb_value
		of either argument, and to have a width equal to one more than the
		maximum width of either argument.
		
		The exact type of the result is determined by the further decendent
		of Word among the arguments.
		
		"""
		
		if (not isinstance(other,Word)):
			raise TypeError("Subtraction only defined for Word instances.")
		
		# the smallest lsb_value should be preserved
		new_lsb_value = min(self.word_format.lsb_value, other.word_format.lsb_value)
		new_lsb_pot = int(np.log2(new_lsb_value))
		
		# the width should be the largest width + 1
		new_width = max(self.word_format.width, other.word_format.width) + 1
		
		new_fmt = WordFormat(new_width,new_lsb_pot)
		new_val = self.value - other.value
		
		# check which type to use for result
		type_other = type(other)
		type_self = type(self)
		if (isinstance(other,type_self)):
			return type_other(new_val,new_fmt)
		else:
			return type_self(new_val,new_fmt)

	def __mul__(self,other):
		"""
		Multiplication operation.
		
		Notes:
		Both arguments should be Word instances.
		
		The result is returned as a Word instance, and the WordFormat
		for the result is automatically derived to have the an lsb_value
		equal to the product of the lsb_values for the two arguments and
		a width equal to the sum of the widths of the two arguments.
		
		The exact type of the result is determined by the further decendent
		of Word among the arguments.
		
		"""
		
		if (not isinstance(other,Word)):
			raise TypeError("Multiplication only defined for Word instances.")
		
		# the lsb_value is the product of that of the two arguments
		new_lsb_value = self.word_format.lsb_value * other.word_format.lsb_value
		new_lsb_pot = int(np.log2(new_lsb_value))
		
		# the width should be the sum of the widths of the two arguments
		new_width = self.word_format.width + other.word_format.width
		
		new_fmt = WordFormat(new_width,new_lsb_pot)
		new_val = self.value * other.value
		
		# check which type to use for result
		type_other = type(other)
		type_self = type(self)
		if (isinstance(other,type_self)):
			return type_other(new_val,new_fmt)
		else:
			return type_self(new_val,new_fmt)

	@property
	def word_format(self):
		"""
		Return the WordFormat instance that defines the representation.
		
		"""
		
		return self._word_format

	@property 
	def scaled_value(self):
		"""
		Return the scaled value represenation of the number.
		
		"""
		
		return self._scaled_value

	@property
	def value(self):
		"""
		Return the actual number represented.
		
		"""
		
		neg_value = self.scaled_value & (-self.word_format.minimum_scaled_value)
		pos_value = self.scaled_value & self.word_format.maximum_scaled_value
		
		return 1.0 * (pos_value - neg_value) * self.word_format.lsb_value

# End class Word(object):

class WordComplex(Word):
	"""
	Extension of Word to represent complex-valued numbers.

	"""

	def __init__(self,val,fmt,dbg=False):
		"""
		Construct a limited width binary representation of a complex number.
		
		Arguments:
		val -- The number to be represented in the given format.
		fmt -- The WordFormat instance that defines the representation.
		
		Notes:
		val can be real-valued, in which case a zero imaginary component
		is assumed.
		
		See the constructor method of Word for additional information.
		"""
		
		self._word_format = fmt
		
		# if val is a Word instance, extract the true value
		if (isinstance(val,Word)):
			val = val.value
			
		if (dbg):
			while True:
				input("DBG:")
		
		# cast val to complex
		val = np.array(val,dtype=np.complex)
		
		# store real and imaginary parts separately
		self._scaled_value_real = np.array(np.array(val.real) / fmt.lsb_value, dtype=np.int64)
		self._scaled_value_imag = np.array(np.array(val.imag) / fmt.lsb_value, dtype=np.int64)
		
		# test if value components are in the allowable range
		if (not (fmt.value_in_range(val.real) and fmt.value_in_range(val.imag))):
			raise OverflowError("Given value is not in the range for the given format.")
		
		# mask out all bits outside the word width
		self._scaled_value_real =  self._scaled_value_real & fmt.mask
		self._scaled_value_imag =  self._scaled_value_imag & fmt.mask

	def real(self):
		"""
		Return the real component of the complex number as a Word.
		
		Notes:
		The returned type is Word and NOT WordComplex.
		"""
		
		return Word(self.value.real,self.word_format)

	def imag(self):
		"""
		Return the imaginary component of the complex number as a Word.
		
		Notes:
		The returned type is Word and NOT WordComplex.
		"""
		
		return Word(self.value.imag,self.word_format)

	#~ def __add__(self,other):
		#~ """Addition operation overload."""
		#~ new_nbits_dec = max(self.fixed_width_type.GetDecimal(),other.fixed_width_type.GetDecimal())
		#~ new_nbits_int = max(self.fixed_width_type.GetInteger(),other.fixed_width_type.GetInteger())
		#~ new_nbits_width = 1 + new_nbits_dec + new_nbits_int
		#~ new_fwtype = FixedWidthType(nbits_width=new_nbits_width,nbits_decimal=new_nbits_dec);
		#~ add_val = self.GetValue() + other.GetValue()
		#~ return FixedWidthNumComplex(rval=add_val.real,ival=add_val.imag,fwtype=new_fwtype)
#~ 
	#~ def __sub__(self,other):
		#~ """Addition operation overload."""
		#~ new_nbits_dec = max(self.fixed_width_type.GetDecimal(),other.fixed_width_type.GetDecimal())
		#~ new_nbits_int = max(self.fixed_width_type.GetInteger(),other.fixed_width_type.GetInteger())
		#~ new_nbits_width = 1 + new_nbits_dec + new_nbits_int
		#~ new_fwtype = FixedWidthType(nbits_width=new_nbits_width,nbits_decimal=new_nbits_dec);
		#~ sub_val = self.GetValue() - other.GetValue()
		#~ return FixedWidthNumComplex(rval=sub_val.real,ival=sub_val.imag,fwtype=new_fwtype)
#~ 
	#~ def __mul__(self,other):
		#~ """Multiply operation overload."""
		#~ new_nbits_dec = self.fixed_width_type.GetDecimal() + other.fixed_width_type.GetDecimal()
		#~ new_nbits_int = self.fixed_width_type.GetInteger() + other.fixed_width_type.GetInteger()
		#~ new_nbits_width = new_nbits_dec + new_nbits_int
		#~ new_fwtype = FixedWidthType(nbits_width=new_nbits_width,nbits_decimal=new_nbits_dec);
		#~ mul_val = self.GetValue() * other.GetValue()
		#~ return FixedWidthNumComplex(rval=mul_val.real,ival=mul_val.imag,fwtype=new_fwtype)

	#~ def TestScaledValueWithinBounds(self):
		#~ if (self.scaledval_real.size == 1):
			#~ if (self.scaledval_real > self.fixed_width_type.scaled_max or self.scaledval_real < self.fixed_width_type.scaled_min or 
			#~ self.scaledval_imag > self.fixed_width_type.scaled_max or self.scaledval_imag < self.fixed_width_type.scaled_min):
				#~ raise FixedWidthOverflowError
				#~ return False
			#~ else:
				#~ return True
		#~ else:
			#~ if ((self.scaledval_real > self.fixed_width_type.scaled_max).any() or (self.scaledval_real < self.fixed_width_type.scaled_min).any() or 
			#~ (self.scaledval_imag > self.fixed_width_type.scaled_max).any() or (self.scaledval_imag < self.fixed_width_type.scaled_min).any()):
				#~ raise FixedWidthOverflowError
				#~ return False
			#~ else:
				#~ return True

	@property
	def scaled_value_real(self):
		"""
		Return the scaled value represenation of the real component of the number.
		
		"""
		
		return self._scaled_value_real

	@property
	def scaled_value_imag(self):
		"""
		Return the scaled value represenation of the imaginary component of the number.
		
		"""
		
		return self._scaled_value_imag

	@property
	def scaled_value(self):
		"""
		Return the scaled value represenation of the number.
		
		"""
		
		return self.scaled_value_real + 1j*self.scaled_value_imag

	@property
	def value(self):
		"""
		Return the actual number represented.
		
		"""
		
		neg_value_real = self.scaled_value_real & (-self.word_format.minimum_scaled_value)
		pos_value_real = self.scaled_value_real & self.word_format.maximum_scaled_value
		neg_value_imag = self.scaled_value_imag & (-self.word_format.minimum_scaled_value)
		pos_value_imag = self.scaled_value_imag & self.word_format.maximum_scaled_value
		return np.array(((pos_value_real - neg_value_real) + 1.0j*(pos_value_imag - neg_value_imag))*self.word_format.lsb_value,dtype=np.complex)

# End class WordComplex

