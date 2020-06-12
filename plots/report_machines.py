import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import re
import statistics as stats
import time
import numpy as np
import glob

folder = "../output_measures_comp2/"

file_name = "*N-time*"
full_name_luca =glob.glob(folder + file_name)

folder = "../output_measures/"
file_name = "06-10.13:13:29-time"
full_name_jannik =folder + file_name

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

plt.rcParams.update({'font.size': 24})


styles=['-','--','-.',':']
colors=['b', 'g', 'r', 'c', 'm', 'y', 'k','brown']
#comp_colors=['darkblue', 'softblue', 'darkgreen', 'soft green', 'dark red', 'salmon', 'dark yellow', 'light yellow' ]
comp_colors = ['#030764','#054907','#840000','#6488ea','#6fc276','#ff796c','#d5b60a','#7d7103']
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

def reo_work(params):
    (flag,hiddenstate,differentObservables,T)=params
    #KN^2+ 2KN+ 6N^2T−4N^2+ 7NT+ 4N+ 3T−2) [Latex]
    return differentObservables*hiddenstate*hiddenstate+2*differentObservables*hiddenstate+6*hiddenstate*hiddenstate*T-4*hiddenstate*hiddenstate+7*hiddenstate*T +4*hiddenstate+ 3*T-2

def base_work(params):
    (flag,hiddenstate,differentObservables,T)=params
    #4∗N+ 7∗T∗N∗N+ 4∗T∗N+ 3∗(T−1)∗N∗N+ 3∗T+ 3+ 3∗T∗K∗N+ (T−1)∗N+ N∗N+ N∗K [Latex]
    return (4*hiddenstate+7*T*hiddenstate*hiddenstate+4*T*hiddenstate+3*(T-1)*hiddenstate*hiddenstate+3*T+3+3*T*hiddenstate*differentObservables+(T-1)*hiddenstate+hiddenstate*hiddenstate+hiddenstate*differentObservables)
work_functions=dict()
work_functions['std']=base_work
work_functions['stb']=base_work
work_functions['cop']=base_work
work_functions['reo']=reo_work
work_functions['vec']=reo_work
work_functions['vec-op']=reo_work

def reo_memory(params):
    (flag,hiddenstate,differentObservables,T)=params
    #2KN^2+ 2KN+ 5N^2T+ 5NT+ 2N+ 4T−5 reads + KN^2+ 2KN+N^2T+ 3N^2+ 6NT+ 4N+T−3 writes[Latex]
    return 3*differentObservables*hiddenstate*hiddenstate+4*differentObservables*hiddenstate+6*hiddenstate*hiddenstate*T+11*hiddenstate*T+6*hiddenstate+5*T-8+3*hiddenstate*hiddenstate

def base_memory(params):
    (flag,hiddenstate,differentObservables,T)=params
    #12∗N+ 3 + 5N∗N∗T+ 9∗N∗T+ 4∗T+ 9∗N∗N∗(T−1)+ 3∗N∗(T−1)+ 2∗T∗K∗N+ N∗N+ N∗K [Latex]
    return  8*((12*hiddenstate+3+5*hiddenstate*hiddenstate*T+9*hiddenstate*T+4*T+9*hiddenstate*hiddenstate*(T-1)+3*hiddenstate*(T-1)+2*T*hiddenstate*differentObservables+hiddenstate*hiddenstate+hiddenstate*differentObservables))
memory_functions=dict()
memory_functions['std']=base_memory
memory_functions['stb']=base_memory
memory_functions['cop']=base_memory
memory_functions['reo']=reo_memory
memory_functions['vec']=reo_memory
memory_functions['vec-op']=reo_memory


def base_memory_compulsory(params):

    (flag,hiddenstate,differentObservables,T)=params
    #Compulsory misses only:
    return 3*hiddenstate*T + hiddenstate*hiddenstate*T+T + hiddenstate + hiddenstate*hiddenstate + hiddenstate * differentObservables







f = open(full_name_luca[0])
text=f.read()
flags_luca=dict()
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
    
    if file not in flags_luca:
        flags_luca[file]=dict()
    if flag not in flags_luca[file]:
        flags_luca[file][flag]=dict()
    if n not in flags_luca[file][flag]:
        flags_luca[file][flag][n]=[]
    flags_luca[file][flag][n].append(cycles)


f = open(full_name_jannik+'.txt')
text=f.read()
flags_jannik=dict()
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
    
    if file not in flags_jannik:
        flags_jannik[file]=dict()
    if flag not in flags_jannik[file]:
        flags_jannik[file][flag]=dict()
    if n not in flags_jannik[file][flag]:
        flags_jannik[file][flag][n]=[]
    flags_jannik[file][flag][n].append(cycles)





fig = plt.figure()
ax=plt.subplot(111)

ax.set_xlabel('N')
ax.set_ylabel('Performance [flops/cycle]',rotation='horizontal')
ax.yaxis.set_label_coords(0.268,1.02)
#ax.set_title('Performance comparision gcc vs. icc')
ax.grid(which='minor', linestyle=':', linewidth='0.5', color='black')



marker=0
color=0
style=0

        

for file in flags_jannik.keys():
    for flag in flags_jannik[file].keys():
        if(flag[0] == 'i'):
            continue
        x=[]
        y=[]
        for n in flags_jannik[file][flag]:
            params=(flag,n,n,n*n)
            (flag,hiddenstate,differentObservables,T)=params
            work=work_functions[file](params)
            x.append(n)
            y.append(work/stats.median(flags_jannik[file][flag][n]))
            if(flag[0] == 'i'):
                plot_flag = 'icc '
            else:
                plot_flag = 'gcc '
            plot_flag = plot_flag + flag[1:]
            
            
        ax.plot(x,y, marker=markers[marker], color=comp_colors[color], linestyle=styles[style], label= file[:6]+': '+ compiler+' ' +str(plot_flag) + ' on system 1')
        
        color+=1
        marker+=1
       

for file in flags_luca.keys():
    if(file == 'vec'):
        continue
        
    for flag in flags_luca[file].keys():
        if(flag[0] == 'i'):
            continue
        #flag = 'g-O2 -mfma'
        plot_flag = flag
        x=[]
        y=[]
        for n in flags_luca[file][flag]:
            params=(flag,n,n,n*n)
            (flag,hiddenstate,differentObservables,T)=params
            work=work_functions[file](params)
            x.append(n)
            y.append(work/stats.median(flags_luca[file][flag][n]))
            if(plot_flag[0] == 'i'):
                plot_flag = 'icc '
            else:
                plot_flag = 'gcc '
            plot_flag = plot_flag + flag[1:]
            
            
        ax.plot(x,y, marker=markers[marker], color=comp_colors[color], linestyle=styles[style], label= file[:6]+': '+ compiler+' ' +str(plot_flag) + ' on system 2')
        
        color+=1
        marker+=1      
 
 
        
        
box=ax.get_position()

ax.set_position([box.x0, box.y0 + box.height * 0.05, box.width, box.height * 0.95])
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),fancybox=True, shadow=True, ncol=1)

fig.set_size_inches(9,12)
timestr = time.strftime("%d-%m_%H;%M")
plt.savefig('../report2_plots/N-' +timestr+"-perf-machines.png",dpi=600)
#plt.show()
plt.close('all')









