import sys
# from IPython.Debugger import Pdb
from IPython.core.debugger import Pdb
# from IPython.Shell import IPShell
from IPython.core import ipapi

# shell = IPShell(argv=[''])

def set_trace():
    ip = ipapi.get()
    def_colors = ip.colors
    Pdb(def_colors).set_trace(sys._getframe().f_back)