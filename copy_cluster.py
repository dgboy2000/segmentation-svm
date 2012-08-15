import os

address = 'baudinpy@igloo.calcul.ecp.fr'
targetdir = '/home/baudinpy/sources/'

if 'password' not in dir():
    password = raw_input('password:')

exclude_list = ['config.py']

def func (arg,dirname,fnames):
    for fname in fnames:
        if fname[-3:] in ['.py']:
            if fname in exclude_list:
                continue
            if dirname=='.': 
                path = ''
            elif dirname[0]=='.':
                path = dirname[2:] + '/'
            else:
                path = dirname + '//'
            sourcepath = os.path.relpath(path+fname,'.')
            cmd = "pscp -r -pw {} {} {}:{}{}".format(
                password,sourcepath,address,targetdir,path+fname)
            print cmd
            os.system(cmd)
    
os.path.walk('.',func,None)