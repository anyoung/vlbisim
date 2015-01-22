#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  untitled.py
#  Jan 22, 2015 15:26:44 EST
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
Defines a number of classes and utilities that are used to describe sources.

"""

import numpy as np
import scipy.constants as const

import SimSWARM.Signal as sg

class Position(object):
	"""
	Define an abstract position for source placement.
	
	"""
	
	def __init__(self):
		"""
		Construct an abstract position.
		
		Nothing to see here.
		"""

# end class Position

class LocalPosition(Position):
	"""
	Define a local position.
	
	This class represents a positition local to some object. It derives
	all attributes from Position as-is.
	"""
	
# end class LocalPosition

class CartesianPosition(Position):
	"""
	Define a position in 3D Cartesian space.
	
	"""
	
	@property
	def x(self):
		"""
		Return x-coordinate.
		
		"""
		
		return self._x
	
	@property
	def y(self):
		"""
		Return y-coordinate.
		
		"""
		
		return self._y

	@property
	def z(self):
		"""
		Return z-coordinate.
		
		"""
		
		return self._z
	
	@property
	def coords(self):
		"""
		Return (x,y,z)-coordinates as a tuple.
		
		"""
		
		return (self._x,self._y,self._z)

	def __init__(self,coords):
		"""
		Construct a position at the specified coordinates.
		
		Arguments:
		coords -- Tuple containing (x,y,z) coordinates in meters.
		
		"""
		
		self._x = coords[0]
		self._y = coords[1]
		self._z = coords[2]

# end class CartesianPosition

class SkyPosition(Position):
	"""
	Define a position on the sky.
	
	"""
	
	@property
	def theta(self):
		"""
		Return theta-coordinate.
		
		"""
		
		return self._theta
	
	@property
	def phi(self):
		"""
		Return phi-coordinate.
		
		"""
		
		return self._phi

	@property
	def coords(self):
		"""
		Return (theta,phi)-coordinates as a tuple.
		
		"""
		
		return (self._theta,self._phi)
	
	@property
	def coords_lmn(self):
		"""
		Return (l,m,n) coordinates, assuming (theta,phi) origin and (l=0,m=0,n=1) coincide
		
		"""
		
		l = np.sin(np.deg2rad(self.theta))*np.cos(np.deg2rad(self.phi))
		m = np.sin(np.deg2rad(self.theta))*np.sin(np.deg2rad(self.phi))
		n = np.cos(np.deg2rad(self.theta))
		
		return (l,m,n)

	def __init__(self,coords):
		"""
		Construct a position on the sky at the given coordinates.
		
		Arguments:
		coords -- (theta,phi)-coordinates on the sky sphere. Theta measures
		from zenith towards the horizon, and phi=0 is aligned with the 
		x-axis. theta and phi are given in degrees.
		
		"""
		
		self._theta = coords[0]
		self._phi = coords[1]

# end class SkyPosition

class Source(object):
	"""
	Abstract representation of source.
	
	"""
	
	@property
	def signal(self):
		"""
		Return the signal for this source.
		
		"""
		
		return self._signal
	
	def __init__(self,asig):
		"""
		Construct a source that produces the given signal.
		
		Arguments:
		sig -- An AnalogSignal instance that defines the source.
		
		"""
		
		self._signal = asig

# end class Source

class PointSource(Source):
	"""
	Define a celestial source.
	
	"""
	
	@property
	def position(self):
		"""
		Return the position of the source.
		
		"""
		
		return self._position
	
	def __init__(self,asig,pos):
		"""
		Construct a point source at the given position.
		
		Arguments:
		asig -- AnalogSignal instance that defines the source.
		pos -- Position instance that locates the source.
		
		"""
		
		self._signal = asig
		self._position = pos
	
# end class PointSource
