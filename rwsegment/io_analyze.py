'''
    Work in progress code.
    Do not copy, do not distribute without permission !
    
    Pierre-Yves Baudin 2012
'''

import os
import sys
import numpy as np
from struct import unpack, pack

#from volume import Volume


FORMATS = {
    0: 'unknown',
    1: 'int1',
    2: 'uint8',
    4: 'int16',
    8: 'int32',
    16: 'float32',
    64: 'float64'
    }
    


ORIENTS = {
    '\x00': 'axial',
    '\x01': 'coronal',
    '\x02': 'sagittal',
    '\x03': 'axial flipped',
    '\x04': 'coronal flipped',
    '\x05': 'sagittal flipped',
    }
    
DEFAULT_HEADER_KEY = {
    'sizeof_hdr':       pack('>i',  348),
    'data_type':        pack('>cccccccccc',
                             *tuple(['\0' for i in range(10)])),
    'db_name':          pack('>cccccccccccccccccc',
                             *tuple(['\0' for i in range(18)])),
    'extents':          pack('>i',  16384),
    'session_error':    pack('>h',  0),
    'regular':          pack('>c',  'r'),
    'hkey_un0':         pack('>c',  '\0'),
    }
    
DEFAULT_IMAGE_DIM = {
    'dim':          pack('>hhhhhhhh', 4,1,1,1,1,1,1,1),
    'unused8':      pack('>h', 0),
    'unused9':      pack('>h', 0),
    'unused10':     pack('>h', 0),
    'unused11':     pack('>h', 0),
    'unused12':     pack('>h', 0),
    'unused13':     pack('>h', 0),
    'unused14':     pack('>h', 0),
    'datatype':     pack('>h', 4),  #short int
    'bitpix':       pack('>h', 16),
    'dim_un0':      pack('>h', 0),
    'pixdim':       pack('>ffffffff', 0.,1.,1.,1.,1.,1.,1.,1.),
    'vox_offset':   pack('>f', 0), 
    'funused1':     pack('>f', 0),
    'funused2':     pack('>f', 0),
    'funused3':     pack('>f', 0),
    'cal_max':      pack('>f', 0),
    'cal_min':      pack('>f', 0), 
    'compressed':   pack('>f', 0), 
    'verified':     pack('>f', 0), 
    'glmax':        pack('>i', 0), 
    'glmin':        pack('>i', 0),
    }
    
DEFAULT_DATA_HIST = {
    'descrip':      pack('>'+''.join(['c' for i in range(80)]), 
                         *tuple(['\0' for i in range(80)]) ),
    'aux_file':     pack('>'+''.join(['c' for i in range(24)]), 
                         *tuple(['\0' for i in range(24)]) ),
    'orient':       pack('>c', '\0'),
    'originator':   pack('>cccccccccc', *tuple(['\0' for i in range(10)])),
    'generated':    pack('>cccccccccc', *tuple(['\0' for i in range(10)])),
    'scannum':      pack('>cccccccccc', *tuple(['\0' for i in range(10)])),
    'patient_id':   pack('>cccccccccc', *tuple(['\0' for i in range(10)])),
    'exp_date':     pack('>cccccccccc', *tuple(['\0' for i in range(10)])),
    'exp_time':     pack('>cccccccccc', *tuple(['\0' for i in range(10)])),
    'hist_un0':     pack('>ccc', '\0','\0','\0'),
    'views':        pack('>i', 0),
    'vols_added':   pack('>i', 0),
    'start_field':  pack('>i', 0),
    'field_skip':   pack('>i', 0),
    'omax':         pack('>i', 0),
    'omin':         pack('>i', 0),
    'smax':         pack('>i', 0),
    'smin':         pack('>i', 0),
    }
    
    
#-------------------------------------------------------------------------------
def load(filename, asdict=False):
    ''' load analyze file
    
    filename must be either the '.hdr' file or the '.img' file
    
    returns: dict with keys: 'data', 'spacing', etc.
    
    TODO:
    - load origin ?
    - load orient ?
    
    '''

    base,ext = os.path.splitext(filename)
    if ext not in ('.hdr', '.img'):
        print 'not analyze file'
        sys.exit(1)
        
    header_file= base + '.hdr'
    data_file= base + '.img'
    
    if not os.path.isfile(header_file):
        print 'could not open file:', header_file
        sys.exit(1)
        
    if not os.path.isfile(data_file):
        print 'could not open file:', data_file
        sys.exit(1)
        
    ''' read header '''
    f = open(header_file,'r')


    h_size = f.read(4)
    E = '>' #big endian
    endian = 'big'
    if int(unpack(E+'i', h_size)[0] ) != 348:
        E = '<' #little endian
        endian = 'little'
        if int(unpack(E+'i', h_size)[0]) != 348:
            print 'error in file:', header_file
            
    header_key = {}
    header_key['sizeof_hdr'] =      unpack(E+'i', h_size)[0]
    header_key['data_type'] =       unpack('>cccccccccc', f.read(10))
    header_key['db_name'] =         unpack(E+'cccccccccccccccccc', 
                                           f.read(18))
    header_key['extents'] =         unpack(E+'i', f.read(4))[0]
    header_key['session_error'] =   unpack(E+'h', f.read(2))[0]
    header_key['regular'] =         unpack(E+'c', f.read(1))[0]
    header_key['hkey_un0'] =        unpack(E+'c', f.read(1))[0]
        
    image_dimension = {}
    image_dimension['dim'] =            unpack(E+'hhhhhhhh', f.read(16))
    image_dimension['unused8'] =        unpack(E+'h', f.read(2))[0]
    image_dimension['unused9'] =        unpack(E+'h', f.read(2))[0]
    image_dimension['unused10'] =       unpack(E+'h', f.read(2))[0]
    image_dimension['unused11'] =       unpack(E+'h', f.read(2))[0]
    image_dimension['unused12'] =       unpack(E+'h', f.read(2))[0]
    image_dimension['unused13'] =       unpack(E+'h', f.read(2))[0]
    image_dimension['unused14'] =       unpack(E+'h', f.read(2))[0]
    image_dimension['datatype'] =       unpack(E+'h', f.read(2))[0]
    image_dimension['bitpix'] =         unpack(E+'h', f.read(2))[0]
    image_dimension['dim_un0'] =        unpack(E+'h', f.read(2))[0]
    image_dimension['pixdim'] =         unpack(E+'ffffffff', f.read(32))
    image_dimension['vox_offset'] =     unpack(E+'f', f.read(4))[0]
    image_dimension['funused1'] =       unpack(E+'f', f.read(4))[0]
    image_dimension['funused2'] =       unpack(E+'f', f.read(4))[0]
    image_dimension['funused3'] =       unpack(E+'f', f.read(4))[0]
    image_dimension['cal_max'] =        unpack(E+'f', f.read(4))[0]
    image_dimension['cal_min'] =        unpack(E+'f', f.read(4))[0]
    image_dimension['compressed'] =     unpack(E+'f', f.read(4))[0]
    image_dimension['verified'] =       unpack(E+'f', f.read(4))[0]
    image_dimension['glmax'] =          unpack(E+'i', f.read(4))[0]
    image_dimension['glmin'] =          unpack(E+'i', f.read(4))[0]
        
    data_history = {}
    data_history['descrip']     = unpack(E+''.join(['c' for i in range(80)]),
                                         f.read(80))
    data_history['aux_file']    = unpack(E+''.join(['c' for i in range(24)]),
                                         f.read(24))
    data_history['orient']      = unpack(E+'c', f.read(1))[0]
    data_history['originator']  = unpack(E+'cccccccccc', f.read(10))
    data_history['generated']   = unpack(E+'cccccccccc', f.read(10))
    data_history['scannum']     = unpack(E+'cccccccccc', f.read(10))
    data_history['patient_id']  = unpack(E+'cccccccccc', f.read(10))
    data_history['exp_date']    = unpack(E+'cccccccccc', f.read(10))
    data_history['exp_time']    = unpack(E+'cccccccccc', f.read(10))
    data_history['hist_un0']    = unpack(E+'ccc', f.read(3))
    data_history['views']       = unpack(E+'i', f.read(4))[0]
    data_history['vols_added']  = unpack(E+'i', f.read(4))[0]
    data_history['start_field'] = unpack(E+'i', f.read(4))[0]
    data_history['field_skip']  = unpack(E+'i', f.read(4))[0]
    data_history['omax']        = unpack(E+'i', f.read(4))[0]
    data_history['omin']        = unpack(E+'i', f.read(4))[0]
    data_history['smax']        = unpack(E+'i', f.read(4))[0]
    data_history['smin']        = unpack(E+'i', f.read(4))[0]
    
    f.close()
    
    ''' extract info from header: data type '''
    type_index = image_dimension['datatype']
    bit_pixels = image_dimension['bitpix']
    if type_index not in FORMATS:
        print 'unauthorized data type: %d (%d bits), ' %(type_index,bit_pixels)
        if 'int'+str(bit_pixels) in FORMATS.values():
            idx = FORMATS.values().index('int'+str(bit_pixels))
            data_type_str = FORMATS.values()[idx]
            print 'try loading data as', data_type_str
        else:
            sys.exit(0)
    else:
        data_type_str = FORMATS[type_index]
        
    data_type = np.dtype(data_type_str)
    if data_type.str[0] in '<>' and data_type.str != E:
        data_type = E + data_type.str[1:]
    
    ''' shape '''
    shape = image_dimension['dim'][1:4]
    shape = [shape[len(shape)-i-1] for i in range(len(shape))]
    
    ''' spacing '''
    spacing = image_dimension['pixdim'][1:4]
    
    ''' origin ? '''
    origin = (0,0,0)
    
    ''' orient ? '''
    orient = ORIENTS[data_history['orient']]
    
    ''' load data file '''
    im = open(data_file, 'rb')
    size = np.product(shape)
    data = np.fromfile(im, dtype=data_type, count=size)
    im.close()
    data = data.reshape(shape)[:,::-1,:].squeeze()
    

            
    if asdict:
        ''' make dict '''
        volume = {
            'data': data,
            'spacing': spacing,
            'orient': orient,
            'origin': origin,
            'format header': {
                'format': 'analyze75',
                'endian': endian,
                'header_key': header_key,
                'image_dimension': image_dimension,
                'data_history': data_history,
                }
            }
        return volume
    else:
        return data

#-------------------------------------------------------------------------------
    
    
#-------------------------------------------------------------------------------
def save(filename, volume):
    ''' save volume data to analyze file
    
    filename must end by '.img' or '.hdr'
    
    volume is a dictionary and must have the following key:
    - 'data' (numpy array)
    
    additionally, you can specify :
    - 'spacing' (if not set, will default to (1,1,1)
    - 'origin' (default = (0,0,0)
    
    TODO:
    - orientation
    - origin
    
    tip: you can use class Volume in rmn.io
    
    '''

    base,ext = os.path.splitext(filename)
    if ext not in ('.hdr', '.img'):
        print 'not analyze file'
        sys.exit(1)
        
    ''' make file names '''
    header_file= base + '.hdr'
    data_file= base + '.img'
    
    dirname = os.path.dirname(filename)
    if len(dirname)==0:
        dirname = './'
        
    
    if not os.path.isdir(dirname):
        print 'could not find output directory:', dirname
        sys.exit(1)
    
    
    ''' '''
    if not type(volume) is dict:
        datavol = {'data': volume}
    else:
        datavol = volume
    
    ''' byte order '''
    E = '>'
    
    ''' if has the same format and header, copy all '''
    if datavol.has_key('format header') and \
        datavol['format header']['format'] == 'analyze75':
        
        v = datavol['format header']['header_key']
        
        endian = datavol['format header'].get('endian','big')
        if endian == 'little':
            E = '<'
        
        header_key = {
            'sizeof_hdr':       pack(E+'i',                 v['sizeof_hdr']),
            'data_type':        pack(E+'cccccccccc',        *v['data_type']),
            'db_name':          pack(E+'cccccccccccccccccc',*v['db_name']),
            'extents':          pack(E+'i',                 v['extents']),
            'session_error':    pack(E+'h',                 v['session_error']),
            'regular':          pack(E+'c',                 v['regular']),
            'hkey_un0':         pack(E+'c',                 v['hkey_un0']),
            }
            
        v = datavol['format header']['image_dimension']
        image_dimension = {
            'dim':          pack(E +'hhhhhhhh',     *v['dim']),
            'unused8':      pack(E+'h',             v['unused8']),
            'unused9':      pack(E+'h',             v['unused9']),
            'unused10':     pack(E+'h',             v['unused10']),
            'unused11':     pack(E+'h',             v['unused11']),
            'unused12':     pack(E+'h',             v['unused12']),
            'unused13':     pack(E+'h',             v['unused13']),
            'unused14':     pack(E+'h',             v['unused14']),
            'datatype':     pack(E+'h',             v['datatype']),  #short int
            'bitpix':       pack(E+'h',             v['bitpix']),
            'dim_un0':      pack(E+'h',             v['dim_un0']),
            'pixdim':       pack(E+'ffffffff',      *v['pixdim']),
            'vox_offset':   pack(E+'f',             v['vox_offset']), 
            'funused1':     pack(E+'f',             v['funused1']),
            'funused2':     pack(E+'f',             v['funused2']),
            'funused3':     pack(E+'f',             v['funused3']),
            'cal_max':      pack(E+'f',             v['cal_max']),
            'cal_min':      pack(E+'f',             v['cal_min']), 
            'compressed':   pack(E+'f',             v['compressed']), 
            'verified':     pack(E+'f',             v['verified']), 
            'glmax':        pack(E+'i',             v['glmax']), 
            'glmin':        pack(E+'i',             v['glmin']),
            }
            
        v = datavol['format header']['data_history']
        data_history = {
            'descrip':      pack(E+''.join(['c' for i in range(80)]), 
                                 *v['descrip'] ),
            'aux_file':     pack(E+''.join(['c' for i in range(24)]), 
                                 *v['aux_file'] ),
            'orient':       pack(E+'c', v['orient']),
            'originator':   pack(E+'cccccccccc', *v['originator']),
            'generated':    pack(E+'cccccccccc', *v['generated']),
            'scannum':      pack(E+'cccccccccc', *v['scannum']),
            'patient_id':   pack(E+'cccccccccc', *v['patient_id']),
            'exp_date':     pack(E+'cccccccccc', *v['exp_date']),
            'exp_time':     pack(E+'cccccccccc', *v['exp_time']),
            'hist_un0':     pack(E+'ccc', *v['hist_un0']),
            'views':        pack(E+'i', v['views']),
            'vols_added':   pack(E+'i', v['vols_added']),
            'start_field':  pack(E+'i', v['views']),
            'field_skip':   pack(E+'i', v['start_field']),
            'omax':         pack(E+'i', v['omax']),
            'omin':         pack(E+'i', v['omin']),
            'smax':         pack(E+'i', v['smax']),
            'smin':         pack(E+'i', v['smin']),
            }
    else:
        ''' make default analyze header '''
        header_key = dict(DEFAULT_HEADER_KEY)
        image_dimension = dict(DEFAULT_IMAGE_DIM)
        data_history = dict(DEFAULT_DATA_HIST)
    
    ''' set header info: shape (try reshaping) and dtype'''
    shape = datavol['data'].shape
    dtype = datavol['data'].dtype
    data = datavol['data']

    
    while data.ndim < 3:
        data = data[np.newaxis,:]
        
    
    dim = [4,1,1,1,1,1,1,1]
    dim[1:len(shape)+1] = shape[::-1]
    image_dimension['dim'] = pack(E+'hhhhhhhh',*dim)
    
    ''' spacing '''
    if datavol.has_key('spacing'):
        spacing = tuple(datavol['spacing'])
    else:
        spacing = (1,1,1)
    pixdim = (0,) + spacing + (1,1,1,1)
    image_dimension['pixdim'] = pack(E+'ffffffff', *pixdim)
    
    
    ''' dtype '''
    typename = np.dtype(dtype).name
    if typename not in FORMATS.values():
        print 'error, unrecognized format:', typename
        sys.exit(1)

    id_fmt = FORMATS.values().index(typename)
    image_dimension['datatype'] = pack(E+'h', FORMATS.keys()[id_fmt])
    image_dimension['bitpix'] = pack(E+'h', dtype.itemsize*8)
    data_type = ['\0' for a in range(10)]
    data_type[:len(typename)] = typename
    header_key['data_type'] = pack(E+'cccccccccc', *data_type)
    
    ''' write header '''
    f = open(header_file, 'wb')
    f.write(header_key['sizeof_hdr'])
    f.write(header_key['data_type'])
    f.write(header_key['db_name'])
    f.write(header_key['extents'])
    f.write(header_key['session_error'])
    f.write(header_key['regular'])
    f.write(header_key['hkey_un0'])
    
    f.write(image_dimension['dim'])
    f.write(image_dimension['unused8'])
    f.write(image_dimension['unused9'])
    f.write(image_dimension['unused10'])
    f.write(image_dimension['unused11'])
    f.write(image_dimension['unused12'])
    f.write(image_dimension['unused13'])
    f.write(image_dimension['unused14'])
    f.write(image_dimension['datatype'])
    f.write(image_dimension['bitpix'])
    f.write(image_dimension['dim_un0'])
    f.write(image_dimension['pixdim'])
    f.write(image_dimension['vox_offset'])
    f.write(image_dimension['funused1'])
    f.write(image_dimension['funused2'])
    f.write(image_dimension['funused3'])
    f.write(image_dimension['cal_max'])
    f.write(image_dimension['cal_min'])
    f.write(image_dimension['compressed'])
    f.write(image_dimension['verified'])
    f.write(image_dimension['glmax'])
    f.write(image_dimension['glmin'])
    
    f.write(data_history['descrip'])
    f.write(data_history['aux_file'])
    f.write(data_history['orient'])
    f.write(data_history['originator'])
    f.write(data_history['generated'])
    f.write(data_history['scannum'])
    f.write(data_history['patient_id'])
    f.write(data_history['exp_date'])
    f.write(data_history['exp_time'])
    f.write(data_history['hist_un0'])
    f.write(data_history['views'])
    f.write(data_history['vols_added'])
    f.write(data_history['start_field'])
    f.write(data_history['field_skip'])
    f.write(data_history['omax'])
    f.write(data_history['omin'])
    f.write(data_history['smax'])
    f.write(data_history['smin'])
    
    f.close()

    ''' write data. WARNING! reshape and cast to dtype '''
    im = open(data_file, 'wb')

    if dtype.str[0] in '><' and dtype.str[0] != E:
        dtype_ = np.dtype(E + dtype.str[1:])
        data[:,::-1,:].astype(dtype_).tofile(im)
    else:
        data[:,::-1,:].tofile(im)
    im.close()
    
#-------------------------------------------------------------------------------


if __name__=='__main__':
    pass
    
    