import os
import subprocess
import sys

#### MACHINE-INDEPENDENT CONFIGURATION

# bounds : included min and max
vols = {
    '01/': {'AB':39},
    '02/': {'AB':41},
    '03/': {'AB':42},
    '04/': {'AB':40},
    '05/': {'AB':42},
    '06/': {'AB':44},
    '07/': {'AB':45},
    '08/': {'AB':44},
    '09/': {'AB':40},
    '10/': {'AB':44},
    '11/': {'AB':46},
    '12/': {'AB':44},
    '13/': {'AB':44},
    #'14/': {'AB':49},
    '15/': {'AB':40},
    '16/': {'AB':49},
    '17/': {'AB':46},
    '18/': {'AB':44},
    'F23/': {'AB':39},
    'F26/': {'AB':32},
    'F27_2/': {'AB':32},
    'F32/': {'AB':29},
    'F37/': {'AB':31},
    'F42/': {'AB':29},
    'F60/': {'AB':33},
    'M23/': {'AB':23},
    'M26/': {'AB':41},
    'M26_2/': {'AB':32},
    'M28/': {'AB':31},
    'M29/': {'AB':31},
    'M44/': {'AB':28},
    }
    
labelset = [0,13,14,15,16]
gray = 'gray.hdr'
seg = 'seg.hdr'
water = 'water.hdr'


def is_danny_laptop():
  return "HOME" in os.environ  and os.environ["HOME"] == '/Users/dannygoodman'

def is_py_machine():
  return "COMPUTERNAME" in os.environ and os.environ["COMPUTERNAME"] == 'JACQUES'

def is_TWIX():
  return "COMPUTERNAME" in os.environ and os.environ["COMPUTERNAME"] == 'TWIX-C411B-WIN'
  
def is_danny_igloo():
  return "HOME" in os.environ and os.environ["HOME"] == '/home/goodmand'

def is_py_igloo():
  return "HOME" in os.environ and os.environ["HOME"] == '/home/baudinpy'


#### MACHINE-DEPENDENT CONFIGURATION

if is_danny_laptop():
  dir_reg     = '/Users/dannygoodman/Sites/ecp/svmdata/01_register/'
  dir_work    = '/Users/dannygoodman/Sites/ecp/svmdata/segmentation_out'
elif is_py_igloo():
  dir_reg     = '/workdir/baudinpy/01_register/'
  dir_work    = '/workdir/baudinpy/segmentation_out/'
elif is_danny_igloo():
  dir_reg     = '/workdir/baudinpy/01_register/'
  dir_work    = '/workdir/goodmand/segmentation_out/'
elif is_py_machine():
  dir_reg     = '..\\rwtrain\\01_register\\'
  dir_work    = './'
elif is_TWIX():
  dir_reg     = '..\\rwtrain\\01_register\\'
  dir_work    = './'
else:
  raise Exception("Did not recognize the machine I'm running on")






















