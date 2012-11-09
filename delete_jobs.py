import sys
import subprocess


if __name__=='__main__':
    kill_run = False
    if '-r' in sys.argv:
       #kill running
       kill_run = True
 
    output = subprocess.check_output('qstat').split()
    output = output[14:] # remove header
    
    for i in range(len(output)/6):
        line = ' '.join(output[i*6:i*6+6])
        name = output[i*6+2]
        service = output[i*6]
        if name=='baudinpy':
            if output[i*6+4]=='R' and not kill_run:
                continue
            print name, service
            command = ['qdel', service]
            subprocess.call(command)
