import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import re
import statistics as stats
import time
import numpy as np

folder = "../output_measures_report/"
file_name = "06-03.23:37:35-time"
full_name =folder + file_name

file_name_bla = "06-04.00:05:07-bla-time-N"
full_name_bla= folder+file_name_bla


file_name_umdhmm = "06-04.12:18:58-umdhmm-time-N"
full_name_umdhmm= folder+file_name_umdhmm
#parameter im Performance plot auf x-achse
wichtiger_param = 3
#0 = flag	1 = states
#2 = dO	3 = T

#welche work und memory access fuction
#aktuelle_version = 'reo'

#machine specs
scalar_pi=4
vector_pi=scalar_pi*4
mem_beta=25
#compiler
compiler =''





plt.rcParams.update({'figure.autolayout': True})

plt.rcParams.update({'font.size': 21})

styles=['-','--','-.',':']
colors=['b', 'g', 'r', 'c', 'm', 'y', 'k','brown']
#comp_colors=['darkblue', 'softblue', 'darkgreen', 'soft green', 'dark red', 'salmon', 'dark yellow', 'light yellow' ]
comp_colors = ['#030764','#6488ea','#054907','#6fc276','#840000','#ff796c','#d5b60a','#fffe7a', '#08787f','#08787f','#9900fa']
markers=[".","v","^","<",">","1","2","3","4","8","s","p","P","*","h","H","+","x","X","D","d","|","_"]


def to_back(arrays, index):
    value0=arrays[0][index]
    del arrays[0][index]
    arrays[0].append(value0)
    value1=arrays[1][index]
    del arrays[1][index]
    arrays[1].append(value1)
    return (arrays[0],arrays[1])

def to_order(array, index):
    value=array[-1]
    del array[-1]
    array.insert(index,value)
    return array






f = open(full_name+'.txt')
text=f.read()
flags=dict()
#read the file
while(re.search('FLAG', text)):
    
    text=re.split('FILE', text,1)[1:][0]
    file=re.split('FLAG', text, maxsplit=1)[0].strip()
    text=re.split('FLAG', text,1)[1:][0]
    flag=re.split('SEED', text, maxsplit=1)[0].strip()
    text=re.split('SEED', text,1)[1:][0]
    seed=int(re.split('N', text)[0].strip())
    text=re.split('N', text,1)[1:][0]
    n=int(re.split('Median', text)[0].strip())
    cycles=re.split('cycles', text)[0].strip()
    cycles=re.split('Time:', cycles)[1].strip()
    cycles=int(re.split('\.',cycles)[0].strip())
    text=re.split('cycles', text,1)[1:][0]
    
    if file not in flags:
        flags[file]=dict()
    if flag not in flags[file]:
        flags[file][flag]=dict()
    if n not in flags[file][flag]:
        flags[file][flag][n]=[]
    flags[file][flag][n].append(cycles)
    
f = open(full_name_bla+'.txt')
text=f.read()

#read the file
while(re.search('FLAG', text)):
    
    text=re.split('FILE', text,1)[1:][0]
    file=re.split('FLAG', text, maxsplit=1)[0].strip()
    text=re.split('FLAG', text,1)[1:][0]
    flag=re.split('SEED', text, maxsplit=1)[0].strip()
    text=re.split('SEED', text,1)[1:][0]
    seed=int(re.split('N', text)[0].strip())
    text=re.split('N', text,1)[1:][0]
    n=int(re.split('Median', text)[0].strip())
    cycles=re.split('cycles', text)[0].strip()
    cycles=re.split('Time:', cycles)[1].strip()
    cycles=int(re.split('\.',cycles)[0].strip())
    text=re.split('cycles', text,1)[1:][0]
    
    if file not in flags:
        flags[file]=dict()
    if flag not in flags[file]:
        flags[file][flag]=dict()
    if n not in flags[file][flag]:
        flags[file][flag][n]=[]
    flags[file][flag][n].append(cycles)
        
f = open(full_name_umdhmm+'.txt')
text=f.read()

#read the file
while(re.search('FLAG', text)):
    
    text=re.split('FILE', text,1)[1:][0]
    file=re.split('FLAG', text, maxsplit=1)[0].strip()
    text=re.split('FLAG', text,1)[1:][0]
    flag=re.split('SEED', text, maxsplit=1)[0].strip()
    text=re.split('SEED', text,1)[1:][0]
    seed=int(re.split('N', text)[0].strip())
    text=re.split('N', text,1)[1:][0]
    n=int(re.split('Median', text)[0].strip())
    cycles=re.split('cycles', text)[0].strip()
    cycles=re.split('Time:', cycles)[1].strip()
    cycles=int(re.split('\.',cycles)[0].strip())
    text=re.split('cycles', text,1)[1:][0]
    
    if file not in flags:
        flags[file]=dict()
    if flag not in flags[file]:
        flags[file][flag]=dict()
    if n not in flags[file][flag]:
        flags[file][flag][n]=[]
    flags[file][flag][n].append(cycles)
    

   
   

fig = plt.figure()
ax=plt.subplot(111)

ax.set_xlabel('N')
ax.set_ylabel('cycles/iteration',rotation='horizontal')
ax.yaxis.set_label_coords(0.25,1.0)



marker=0
color=0
style=0
for file in flags.keys():
    #for flag in flags[file].keys():
        filename = file
        flag = 'g-O2 -mfma'
        plot_flag = flag
        x=[]
        y=[]
        for n in flags[file][flag]:
            x.append(n)
            y.append(stats.median(flags[file][flag][n]))
            if(file != 'vec'):
                plot_flag = re.sub('\ -mfma$','',flag)
            if(file == 'vec-op'):
                filename = 'umd'
        plt.plot(x,y, marker=markers[marker], color=comp_colors[color], linestyle=styles[style], label= filename+': '+ compiler+' ' +str(plot_flag))
        color+=2
        marker+=2
        if(file=='bla'):
           marker+=2
           
plt.legend(frameon=False,fancybox=True, shadow=True)
figure = plt.gcf()
figure.set_size_inches(9,6)
timestr = time.strftime("%d-%m_%H;%M")
plt.savefig('../report_plots/N-' +timestr+"-cycles-alternatives.png",dpi=600)
#plt.show()
plt.clf()
