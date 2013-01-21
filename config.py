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
 
gray = 'gray.hdr'
seg = 'seg.hdr'
water = 'water.hdr'


def is_danny_laptop():
  return "HOME" in os.environ  and os.environ["HOME"] == '/Users/dannygoodman'

def is_py_machine():
  return "COMPUTERNAME" in os.environ and os.environ["COMPUTERNAME"] == 'JACQUES'
  
def is_py_machine_idm():
  return "USERDOMAIN" in os.environ and os.environ["USERDOMAIN"] == 'MYOLOGIE_DOMAIN'
  
def is_TWIX():
  return "COMPUTERNAME" in os.environ and os.environ["COMPUTERNAME"] == 'TWIX-C411B-WIN'
  
def is_danny_igloo():
  return "HOME" in os.environ and os.environ["HOME"] == '/home/goodmand'

def is_puneet_igloo():
  return "HOME" in os.environ and os.environ["HOME"] == '/home/puneet'

def is_pawan_igloo():
  return "HOME" in os.environ and os.environ["HOME"] == '/home/mudigondpk'

def is_py_igloo():
  return "HOME" in os.environ and os.environ["HOME"] == '/home/baudinpy'\
    and os.path.isdir('/home/mudigondpk') and not os.path.isdir('/home/pawan')

def is_pawan_desktop():
  return "HOME" in os.environ and os.environ["HOME"] == '/home/baudinpy'\
    and os.path.isdir('/home/pawan') and not os.path.isdir('/home/mudigondpk')


#### MACHINE-DEPENDENT CONFIGURATION

if is_danny_laptop():
  dir_reg     = '/Users/dannygoodman/Sites/ecp/svmdata/01_register/'
  dir_work    = '/Users/dannygoodman/Sites/ecp/svmdata/segmentation_out/'
elif is_py_igloo():
  dir_reg     = '/workdir/baudinpy/01_register/'
  dir_work    = '/workdir/baudinpy/segmentation_out/'
elif is_danny_igloo():
  dir_reg     = '/workdir/baudinpy/01_register/'
  dir_work    = '/workdir/goodmand/segmentation_out/'
elif is_puneet_igloo():
  dir_reg     = '/workdir/baudinpy/01_register/'
  dir_work    = '/workdir/puneet/segmentation_out/'
elif is_pawan_igloo():
  dir_reg     = '/workdir/baudinpy/01_register/'
  dir_work    = '/workdir/mudigondpk/segmentation_out/'
elif is_py_machine():
  dir_reg     = '..\\rwtrain\\01_register\\'
  dir_work    = './'
elif is_py_machine_idm():
  dir_reg     = '..\\rwtrain\\01_register\\'
  dir_work    = './'
elif is_TWIX():
  dir_reg     = '../01_register/'
  dir_work    = './'
elif is_pawan_desktop():
  dir_reg     = '/home/baudinpy/svmdata/01_register/'
  dir_work    = '/home/baudinpy/svmdata/segmentation_out/'
else:
  raise Exception("Did not recognize the machine I'm running on")



debug = False      
if '-d' in sys.argv or '--debug' in sys.argv:
    debug = True
if '--folder' in sys.argv:
    ind = sys.argv.index('--folder')
    folder = sys.argv[ind+1]
else:
    folder = None
 
## Set up global logging to file
from rwsegment import utils_logging
## output paths
if sys.platform[:3]=='win':
    current_code_version = subprocess.check_output(['git','rev-parse', 'HEAD'],shell=True)[:-2]
else:
    current_code_version = subprocess.check_output(['git','rev-parse', 'HEAD'])[:-2]
if folder is None:
    folder = current_code_version

if sys.argv[0]=='learn_svm_batch.py':
    dir_log = dir_work + 'learning/{}'.format(folder)
    dir_inf = dir_log  + '/inference/'.format(folder)
    dir_svm = dir_log  + '/svm/'.format(folder)
    if not debug:
        utils_logging.LOG_OUTPUT_DIR = dir_log
elif sys.argv[0]=='segmentation_batch.py':
    dir_log = dir_work + 'segmentation/{}'.format(folder)
    dir_seg = dir_log    
    if not debug:
        utils_logging.LOG_OUTPUT_DIR = dir_log
elif sys.argv[0]=='batch_rwpca.py':
    dir_log = dir_work + 'segmentation_pca/{}'.format(folder)
    dir_seg = dir_log    
    if not debug:
        utils_logging.LOG_OUTPUT_DIR = dir_log

print utils_logging.LOG_OUTPUT_DIR

#logger = utils_logging.get_logger('config', utils_logging.DEBUG)
#logger.info('current code version: {}'.format(current_code_version))
basis = 'default'
if '--basis' in sys.argv:
    ind = sys.argv.index('--basis')
    basis = sys.argv[ind+1]
print 'using basis = {}'.format(basis)

if basis=='default':
    dir_prior = dir_work + '/prior/'
    dir_pca_prior = dir_work + '/prior_pca/'
    dir_prior_edges = dir_work + '/prior_edges/'
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
    folds = [ 
       ['01/', '03/', '05/', '07/', '10/', '12/'],
       ['09/', '11/', '13/', '15/', 'M29/', '16/'],
       ['17/', 'F23/', 'F27_2/', 'F37/','18/', 'F26/'],
       ['F60/', 'M26/', 'M28/', 'M44/', 'F32/', 'F42/'],
       ['02/', '04/', '06/', '08/', 'M23/', 'M26_2/'],
       ]
   

elif basis=='allmuscles':
    dir_prior = dir_work + '/prior_allm/'
    dir_pca_prior = dir_work + '/prior_pca_allm/'
    dir_prior_edges = dir_work + '/prior_edges_allm/'
    vols = {
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
    labelset = [0,1,3,4,5,7,8,10,11,12,13,14,15,16]
    folds = [[key] for key in vols]
         
else:
    raise Exception('unknown basis {}'.format(basis))
    sys.exit(1)

gray = 'gray.hdr'
seg = 'seg.hdr'
water = 'water.hdr'



gray = 'gray.hdr'
seg = 'seg.hdr'
water = 'water.hdr'












