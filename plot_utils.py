import os
import numpy as np

#import matplotlib
#matplotlib.use('Agg')
from matplotlib import pyplot


labelset = [1,3,4,5,7,8,10,11,12,13,14,15,16]

def plot_dice_labels( 
        dices_list,
        labelset=labelset,
        perlabel=True, 
        figsize=10,
        space_l=0.15,
        space_t=0.1,
        ratio=1):
        
    ## make figure
    size = [figsize,int(ratio*figsize)]
    fig = pyplot.figure(figsize=size)
    ax1 = fig.add_subplot(111)
    dpi = fig.get_dpi()
    size_p = [size[0]*dpi, size[1]*dpi]
    
    ## prepare boxes
    nmethod = len(dices_list)
    width = 0.8/(nmethod)
    
    if perlabel: 
        nbox = len(labelset)
        yticks = [muscles[l] for l in labelset]
    else:
        nbox = 1
        yticks = ['allmuscles']
    
    ## init
    legends = []
     
    for imethod in range(nmethod):
        method = dices_list[imethod]
        title = method['title']
        print title
        if perlabel:
            values = []
            for label in labelset:
                values.append(method['values'][label])
        else:
            values = [[]]
            for label in labelset:
                values[0].extend(method['values'][label])
                
        ## boxplot
        import ipdb; ipdb.set_trace()
        positions = [i + -0.4 + (imethod + 0.5)*width for i in range(nbox)]
        bp = pyplot.boxplot(
            values,
            positions=positions,
            widths=width,
            vert=0,
            whis=1.5,
            notch=1,
            bootstrap=1000,
            )
            
        ## improve boxes
        pyplot.setp(bp['boxes'],    color=colors[imethod], zorder=0)
        pyplot.setp(bp['medians'],  color=colors[imethod])
        pyplot.setp(bp['whiskers'], color='black')
        pyplot.setp(bp['fliers'],   color='red', marker='+')
        for box in bp['boxes']:
            pyplot.fill(box.get_xdata(), box.get_ydata(), color=colors[imethod], alpha=0.3)
        
        ## make legend
        r = pyplot.Rectangle(
            (0,0),0,0,
            facecolor=colors[imethod], 
            edgecolor=colors[imethod],
            alpha=0.3)
        legends.append((title,r))
     
    pyplot.title(r'$\mathrm{Dice\ coefficients}$')
    pyplot.yticks(range(nbox), yticks, rotation=0)
    
    pyplot.subplots_adjust(left=space_l, right=0.95, top=1-space_t, bottom=0.1)
    
    ## plot grid
    from matplotlib.ticker import MultipleLocator
    minorLocator   = MultipleLocator(0.05)
    ax1.xaxis.set_minor_locator(minorLocator) 
    ax1.grid(True, linestyle='-', which='both',color='lightgrey', alpha=0.9)

    pyplot.axis([0, 1, nbox-0.5, -0.5])
    pyplot.legend(
        [l[1] for l in legends], 
        [l[0] for l in legends], 
        loc=3, prop={'size':12})
    
    return fig
    
def plot_dice_series( 
        dices_series_list,
        labelset=labelset,
        figsize=10,
        space_l=0.15,
        space_t=0.1,
        ratio=1):
        
    ## make figure
    size = [figsize,int(ratio*figsize)]
    fig = pyplot.figure(figsize=size)
    ax1 = fig.add_subplot(111)
    dpi = fig.get_dpi()
    size_p = [size[0]*dpi, size[1]*dpi]
    
    ## prepare boxes
    nseries = len(dices_series_list)
    
    ## init
    legends = []
    bounds = [np.inf,-np.inf,np.inf,-np.inf]
   
    xticks = []
    nticks = 0
    for iseries in range(nseries):
        series = dices_series_list[iseries]
        title = series['title']
        print  'name of the series {} ({} methods)'.format(series['name'], len(series['methods']))
       
        xvals = [] 
        avg = []
        std = []
        for method in series['methods']:
            values = []
            print 'name of the method {}'.format(method['name']),
            for label in labelset:
                if not label in method['values']: continue
                values.extend(method['values'][label])
            print '({} samples)'.format(len(values))
            avg.append(np.mean(values))
            std.append(np.std(values))
            xvals.append(method['x'])

        xs = np.arange(len(xvals)) + iseries*5e-2
        if nticks < len(xs): nticks = len(xs)
        xticks.append(xvals)

        if np.min(xs) < bounds[0]: bounds[0] = np.min(xs)
        if np.max(xs) > bounds[1]: bounds[1] = np.max(xs)
        if (np.min(avg)-np.max(std)) < bounds[2]: bounds[2] = np.min(avg)-np.max(std)
        if (np.max(avg)+np.max(std)) > bounds[3]: bounds[3] = np.max(avg)+np.max(std)
            
        ## boxplot    
        p = pyplot.errorbar(
            xs,
            avg,
            yerr=std,
            color=colors[iseries],
            linestyle='-',
            marker='o',
            )
            
        ## make legend
        r = pyplot.Rectangle(
            (0,0),0,0,
            facecolor=colors[iseries], 
            edgecolor=colors[iseries],
            alpha=0.3)
        legends.append((title,r))
    
    pyplot.title(r'$\mathrm{Dice\ coefficients}$')
    #pyplot.yticks(range(nbox), yticks, rotation=0)
    
    pyplot.subplots_adjust(left=space_l, right=0.95, top=1-space_t, bottom=0.1)
    
    ## plot grid
    from matplotlib.ticker import MultipleLocator
    minorLocator   = MultipleLocator(0.5)
    ax1.xaxis.set_minor_locator(minorLocator) 
    ax1.grid(True, linestyle='-', which='both',color='lightgrey', alpha=0.9)

    strticks = []
    #print xticks
    for i in range(nseries):
        ntick = len(xticks[i])
        if len(strticks) < ntick: strticks += ['']*(ntick-len(strticks))
        for j in range(ntick):
            strticks[j] += '\n{}'.format( xticks[i][j])
    #print strticks
    #strticks = ['\n'.join([str(xticks[i][j]) for i in range(nseries)]) for j in range(nticks)]
    pyplot.xticks(range(nticks), strticks)
    pyplot.xlabel(r'$C\ or\ C^{\prime}$')

    bounds = [b - 0.1*(-1)**i for i,b in enumerate(bounds)]
    #print bounds
    pyplot.axis(bounds)
    pyplot.legend(
        [l[1] for l in legends], 
        [l[0] for l in legends], 
        loc=3, prop={'size':12})

    return fig


muscles = [
   '',
   'tensor\nfasciae\nlatae',
   '',
   'semiten-\ndinosus',
   'semimem-\nbranosus',
   'adductor\nmagnus',
   '',
   'adductor\nbrevis',
   '',
   'sartorius',
   'biceps\nfemoris (L)',
   'biceps\nfemoris (S)',
   'gracilis',
   'rectus\nfemoris',
   'vastus\nlateralis',
   'vastus\nintermedius',
   'vastus\nmedialis',
   ]

''' colormap '''
colorlist = np.asarray([
    [0,0,0,0],
    [160,212,205,255],
    [164,154,219,255],
    [0,0,255,255],
    [24,203,255,255],
    [231,255,0,255],
    [231, 66, 0, 255],
    [226,143,122,255],
    [228,0,145,255],
    [226,164,43,255],
    [255,0,0,255],
    [255,151,0,255],
    [206,73,255,255],
    [142,5,10,255],
    [18,95,0,255],
    [8,212,125,255],
    [0,255,59,255],
    ])/255.1
from matplotlib.colors import ListedColormap
mycmap = ListedColormap(colorlist[:,:3], 'segmentation')
mycmap.set_bad(color=(0,0,0,0))

colors = [r'blue', r'green', r'red', r'yellow', r'purple', r'orange']



def plot_dices_per_slice(dices_list):
    fig = pyplot.figure()
    ax1 = fig.add_subplot(111)
    nbox = len(dices_list)
    width = 0.8/(nbox) 
    legends   = []
    legends_box   = []

    for i, dices in enumerate(dices_list):
        legend = dices['title']
        print 'nsample:',dices['nsample']
        left = []
        vals = []
        err = []
        minv = []
        maxv = []
        for l in dices.keys():
            if str(l).isdigit():
                n = len(dices[l])
                if n == 0:
                    minv.append(0)
                    maxv.append(0)
                    vals.append(0)
                elif n<=4:  
                    minv.append(0)
                    maxv.append(np.mean(dices[l]))
                    vals.append(np.mean(dices[l]))
                else:
                    v = np.sort(dices[l]) 
                    vals.append(np.median(dices[l]))
                    minv.append(np.mean(v[:n/4]))
                    maxv.append(np.mean(v[(n*3)/4:]))
                left.append(l)
        bp = pyplot.barh(
            np.array(left),
            np.array(maxv),
            height=[width for w in left],
            color=(0,0,1),
            alpha=0.3,
            )
       
        bp = pyplot.barh(
            np.array(left),
            np.array(vals),
            height=[width for w in left],
            color=(0,0,1),
            alpha=0.8,
            )
        bp = pyplot.barh(
            np.array(left),
            np.array(minv),
            height=[width for w in left],
            color=(0,0,0.5),
            alpha=1,
            )
        legends.append(legend)
        r = pyplot.Rectangle((0,0),0,0,facecolor=colors[i], edgecolor=colors[i],alpha=0.3)
        legends_box.append(r)
   
    pyplot.xlabel(r'$\mathrm{slices}$') 
    pyplot.subplots_adjust(left=0.1, right=0.98, top=0.95, bottom=0.1)
    pyplot.title(r'$\mathrm{Dice\ coefficients\ per\ slice\ for\ method:' + legend + '}$')

    #pyplot.yticks(range(nset), yticks)
    from matplotlib.ticker import MultipleLocator
    minorLocator   = MultipleLocator(0.05)
    ax1.xaxis.set_minor_locator(minorLocator) 
    ax1.xaxis.grid(True, linestyle='-', which='both',color='lightgrey', alpha=0.9)


    #pyplot.axis([0.5,1,nset -  0.5,-0.5])
    #pyplot.legend(legends_box, legends, loc=3)


def plot_cross(gray, sol, seg):
    figsize = np.asarray(gray.shape)*(10.0/gray.shape[0])
    pyplot.figure(figsize=figsize)
    pyplot.imshow(gray, interpolation='nearest', cmap='gray')
   
    mseg = np.ma.masked_array(sol, mask=(sol==0), fill_value=0)
    pyplot.imshow(mseg, interpolation='nearest', cmap=mycmap, vmin=0, vmax=16, alpha=0.5) 

    pyplot.subplots_adjust(left=0, right=1, top=1, bottom=0)
    print 'done cross'


''' test script '''
if __name__=='__main__':
    dices = [
        {'name': 'thing1', 'values': {1: np.random.random(100), 2: np.random.random(100)**2 }},
        {'name': 'thing2', 'values': {1: np.random.random(100)/2, 2: np.random.random(100)/2+0.5}},
        ]
        
    fig = plot_dice_labels(dices, perlabel=True, labelset=[1,2])
    pyplot.show()
