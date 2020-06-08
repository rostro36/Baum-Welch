import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import re
import statistics as stats
import time
import numpy as np

folder = "../output_measures_report/"
file_name_novec = "06-07.12:33:11-time-novec"
full_name_novec =folder + file_name_novec

file_name = "06-03.23:37:35-time"
full_name =folder + file_name

file_name_flags = "06-04.15:49:00-time-flags"
full_name_flags = folder+file_name_flags
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

plt.rcParams.update({'font.size': 13})


styles=['-','--','-.',':']
colors=['b', 'g', 'r', 'c', 'm', 'y', 'k','brown']
#comp_colors=['darkblue', 'softblue', 'darkgreen', 'soft green', 'dark red', 'salmon', 'dark yellow', 'light yellow' ]
comp_colors = ['#030764','#6488ea','#054907','#6fc276','#840000','#ff796c','#d5b60a','#fffe7a']
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







f = open(full_name_novec+'.txt')
text=f.read()
flags_novec=dict()
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
    
    if file not in flags_novec:
        flags_novec[file]=dict()
    if flag not in flags_novec[file]:
        flags_novec[file][flag]=dict()
    if n not in flags_novec[file][flag]:
        flags_novec[file][flag][n]=[]
    flags_novec[file][flag][n].append(cycles)
    
f = open(full_name_flags+'.txt')
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
        
f = open(full_name+'.txt')
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
ax.set_ylabel('Performance [flops/cycle]',rotation='horizontal')
ax.yaxis.set_label_coords(0.163,1.02)
#ax.set_title('Performance comparision gcc vs. icc')
ax.grid(which='minor', linestyle=':', linewidth='0.5', color='black')

marker=0
color=0
style=1
for file in flags_novec.keys():
    tmp = ['i-O2 -fno-tree-vectorize','g-O2 -fno-tree-vectorize']
    if(file == 'vec'):
        tmp = ['i-O2 -mfma','g-O2 -mfma']
    for flag in tmp:
        #flag = 'g-O2 -mfma'
        plot_flag = flag
        x=[]
        y=[]
        for n in flags_novec[file][flag]:
            params=(flag,n,n,n*n)
            (flag,hiddenstate,differentObservables,T)=params
            work=work_functions[file](params)
            x.append(n)
            y.append(work/stats.median(flags_novec[file][flag][n]))
            if(file != 'vec'):
                plot_flag = re.sub('\ -mfma$','',flag)
        ax.plot(x,y, marker=markers[marker], color=comp_colors[color], linestyle=styles[style], label= file[:6]+': '+ compiler+' ' +str(plot_flag))
        
        color+=1
        marker+=1
        
box=ax.get_position()

ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25),fancybox=True, shadow=False, ncol=2)

fig.set_size_inches(8,4.5)
timestr = time.strftime("%d-%m_%H;%M")
plt.savefig('../report_plots/N-' +timestr+"-perf-compilers.png",dpi=200)
#plt.show()
plt.close('all')



fig = plt.figure()
ax=plt.subplot(111)

ax.set_xlabel('N')
ax.set_ylabel('Performance [flops/cycle]',rotation='horizontal')
ax.yaxis.set_label_coords(0.163,1.02)
#ax.set_title('Performance comparision flags')
ax.grid(which='minor', linestyle=':', linewidth='0.5', color='black')

marker=0
color=0
style=1
for file in flags.keys():
    for flag in ['g-O2 -mfma','g-O3 -march=native -mfma']:
        #flag = 'g-O2 -mfma'
        plot_flag = flag
        x=[]
        y=[]
        for n in flags[file][flag]:
            params=(flag,n,n,n*n)
            (flag,hiddenstate,differentObservables,T)=params
            work=work_functions[file](params)
            x.append(n)
            y.append(work/stats.median(flags[file][flag][n]))
            if(file != 'vec'):
                plot_flag = re.sub('\ -mfma$','',flag)
        ax.plot(x,y, marker=markers[marker], color=comp_colors[color], linestyle=styles[style], label= file[:6]+': '+ compiler+' ' +str(plot_flag))
        
        color+=1
        marker+=1

box=ax.get_position()

ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25),fancybox=True, shadow=False, ncol=2)

fig.set_size_inches(8,4.5)
timestr = time.strftime("%d-%m_%H;%M")
plt.savefig('../report_plots/N-' +timestr+"-perf-flags.png",dpi=200)
#plt.show()
plt.close('all')





#Plot base model (empty rooflie model)


fig = plt.figure()
ax=plt.subplot(111)

ax.set_xlabel('Operational Intensity [flops/byte]')
ax.set_ylabel('Performance [flops/cycle]',rotation='horizontal')
ax.yaxis.set_label_coords(0.163,1.02)


#plt.title('Roofline Model')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')


ridge_point = scalar_pi/mem_beta
#print(ridge_point)
I = []
flops_byte_scal = []
flops_byte_vec = []
for i in np.arange(0.001,100*ridge_point,0.01):
	I.append(i)
	flops_byte_scal.append(min(scalar_pi, i * mem_beta))
	flops_byte_vec.append(min(vector_pi, i * mem_beta))
	

plt.plot(I,flops_byte_scal,color='k',linewidth=0.8)
plt.plot(I,flops_byte_vec,color='k',linewidth = 0.8)
plt.yscale('log')
plt.xscale('log')
figure = plt.gcf()
figure.set_size_inches(8,4.5)
plt.legend()
#plt.xlabel('Operational Intensity [flops/byte]')
#plt.ylabel('Performance [flops/cycle]')


m=["+","x"]
marker=0
color=0
style=1
#compiler = 'gcc'
for count, file in enumerate(flags.keys()):
    #for flag in list(flags[file].keys())[0]:
        flag = 'g-O2 -mfma'
        plot_flag = flag
        x=[]
        y=[]
        for n in flags[file][flag]:
            cycles=stats.median(flags[file][flag][n])
            params=(flag,n,n,n*n)
            (flag,hiddenstate,differentObservables,T)=params
            work=work_functions[file](params)
            memory=memory_functions[file](params)
            x.append(work/memory)
            y.append(work/cycles)
            if(file != 'vec'):
                plot_flag = re.sub('\ -mfma$','',flag)

        plt.plot(x,y, marker=m[marker], color=comp_colors[color], linestyle=styles[style], label= file[:6]+': '+compiler + ' '+str(plot_flag))
        color+=2
        #color=0
        marker=(marker + 1)%2



plt.legend()
#plt.show()
timestr = time.strftime("%d-%m_%H;%M")
plt.savefig('../report_plots/N-'+timestr+"-roof-gcc.png",dpi=200)
plt.clf()






