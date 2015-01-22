#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  untitled.py
#  Jan 22, 2015 14:54:55 EST
#  Copyright 2015
#         
#  Andre Young <andre.young@cfa.harvard.edu>
#  Harvard-Smithsonian Center for Astrophysics
#  60 Garden Street, Cambridge
#  MA 02138
#  
#  Changelog:
#  	AY: Created 2015-01-22
"""
Defines antenna related classes and utilities.

"""

import numpy as np
import scipy.constants as const
import collections

import SimSWARM.Blocks as bl
import SimSWARM.Signal as sg
import SimSWARM.Source as sr

class Antenna(object):
	"""
	Define a single antenna.
	
	"""
	
	@property
	def position(self):
		"""
		Return the position of the antenna as a tuple.
		
		"""
		
		return tuple(self._position)
	
	@property
	def sources(self):
		"""
		Return a list of sources visible to the antenna.
		
		"""
		
		return self._sources
		
	def __init__(self,pos):
		"""
		Create an antenna at the given position.
		
		Arguments:
		pos -- (x,y,z)-coordinates of the antenna provided in any form
		that can be used as input to numpy array.
		
		"""
		
		self._position = np.array(pos)
		
		# initialize visible sources
		self._sources = list()
	
	def add_source(self,src):
		"""
		Add a source to this antenna's received signals.
		
		Arguments:
		src -- SimSWARM.Source.Source instance that defines the source, 
		or an iterable object of such instances.
		
		"""
		
		if (isinstance(src,collections.Iterable)):
			for s in src:
				self.add_source(s)
		else:
			if (not isinstance(src,sr.Source)):
				raise TypeError("Only Source instances can be added to Antenna source list.")
			
			self.sources.append(src)
	
	def receiver_block(self):
		"""
		Generate the signal received by this antenna.
		
		The result is returned as a Block instance on which the output
		method may be called to yield the signal received by the antenna.
		
		"""
		
		# first create a list of received signals
		received_signals = list()
		for src in self.sources:
			if (isinstance(src,sr.PointSource)): # no other source types available yet
				src_pos = src.position
				if (isinstance(src_pos,sr.LocalPosition)):
					# for LocalPosition sources, the signal is received as-is
					received_signals.append(src.signal)
				elif (isinstance(src_pos,sr.SkyPosition)):
					# add propagation delay; other effects can be added later
					# and can include atmospheric delay / attenuation, gain
					# pattern of the antenna, etc.
					l,m,n = src_pos.coords_lmn
					ant_x,ant_y,ant_z = self.position
					delay = -(l*ant_x + m*ant_y + n*ant_z)/const.c
					delay_block = bl.AnalogDelay(delay)
					delay_block.attach_source(src.signal)
					received_signals.append(delay_block)
				elif (isinstance(src_pos,sr.CartesianPosition)):
					# add propagation delay
					src_x,src_y,src_z = src_pos.coords
					ant_x,ant_y,ant_z = self.position
					delay = -np.sqrt((ant_x-src_x)**2 + (ant_y-src_y)**2 + (ant_z-src_z)**2)/const.c
					delay_block = bl.AnalogDelay(delay)
					delay_block.attach_source(src.signal)
					received_signals.append(delay_block)
		
		result = bl.AnalogCombiner()
		result.attach_source(received_signals)
		
		return result
	
# end class Antenna

class Array(Antenna):
	"""
	Define an array of antennas.
	
	This class derives from Antenna so that the attributes of all 
	antenna elements can easily be set by the same attribute for the 
	array instance.
	
	"""
	
	@property
	def antennas(self):
		"""
		Return the antennas as a list.
		
		"""
		
		return self._antennas
	
	@property
	def positions(self):
		"""
		Return the positions of the antennas.
		
		The returned result is a tuple of lists, where the outer tuple
		contains (x_all,y_all,z_all) and each *_all is a list containing
		the specific coordinate value for all antennas.
		"""
		
		x_all,y_all,z_all = list(),list(),list()
		for ant in self.antennas:
			x,y,z = ant.position
			x_all.append(x)
			y_all.append(y)
			z_all.append(z)
		
		return (x_all,y_all,z_all)
		
	
	def __init__(self,ant):
		"""
		Construct an array comprising the given list of antennas.
		
		Arguments:
		ant -- List of Antenna instances.
		"""
		
		self._antennas = ant

# end class Array
