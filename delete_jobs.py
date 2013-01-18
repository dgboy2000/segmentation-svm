import sys
import subprocess


if __name__=='__main__':
    kill_run = False
    iname = sys.argv.index('-n') + 1
    jname = sys.argv[iname]
    if '-r' in sys.argv:
       #kill running
       kill_run = True
 
    if '-s' in sys.argv:
        istart = sys.argv.index('-s')+1
        start = int(sys.argv[istart])
    else:
        start = 0

    output = subprocess.check_output('qstat').split()
    output = output[14:] # remove header
    
    for i in range(len(output)/6):
        line = ' '.join(output[i*6:i*6+6])
        name = output[i*6+2]
        service = output[i*6]
        if int(service[:6])<start: continue
        if name==jname:
            if output[i*6+4]=='R' and not kill_run:
                continue
            print name, service
            command = ['qdel', service]
            subprocess.call(command)
