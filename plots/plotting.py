import matplotlib.pyplot as plt
import re
import statistics as stats
import time
import numpy as np



folder = "../output_measures/"
file_name = "reo-time"
full_name =folder + file_name

#parameter im Performance plot auf x-achse
wichtiger_param = 3
#0 = flag	1 = states
#2 = dO	3 = T

#welche work und memory access fuction
aktuelle_version = 'reo'

#machine specs
scalar_pi=4
vector_pi=scalar_pi*4
mem_beta=25
#compiler
compiler ='gcc'




plt.rcParams.update({'figure.autolayout': True})

plt.rcParams.update({'font.size': 14})


styles=['-','--','-.',':']
colors=['b', 'g', 'r', 'c', 'm', 'y', 'k']
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


def base_memory_compulsory(params):

    (flag,hiddenstate,differentObservables,T)=params
    #Compulsory misses only:
    return 3*hiddenstate*T + hiddenstate*hiddenstate*T+T + hiddenstate + hiddenstate*hiddenstate + hiddenstate * differentObservables



f = open(full_name+".txt")
text=f.read()
flags=dict()
#read the file
while(re.search('FLAG', text)):
    
    text=re.split('FLAG', text,1)[1:][0]
    flag=re.split('SEED', text, maxsplit=1)[0].strip()
    text=re.split('SEED', text,1)[1:][0]
    seed=int(re.split('HIDDENSTATE', text)[0].strip())
    text=re.split('HIDDENSTATE', text,1)[1:][0]
    hiddenstate=int(re.split('DIFFERENTOBSERVABLES', text)[0].strip())
    text=re.split('DIFFERENTOBSERVABLES', text,1)[1:][0]
    dO=int(re.split('T', text)[0].strip())
    text=re.split('T ', text,1)[1:][0]
    T=int(re.split('Median', text)[0].strip())
    cycles=re.split('cycles', text)[0].strip()
    cycles=re.split('Time:', cycles)[1].strip()
    cycles=int(re.split('\.',cycles)[0].strip())
    text=re.split('cycles', text,1)[1:][0]
    
    parameters=[flag,hiddenstate,dO,T]
    parameter_names=['flag','hiddenstate','different observables','T']
    (order_params,order_names)=to_back([parameters,parameter_names],wichtiger_param)
    
    if order_params[0] not in flags:
        flags[order_params[0]]=dict()
    if order_params[1] not in flags[order_params[0]]:
        flags[order_params[0]][order_params[1]]=dict()
    if order_params[2] not in flags[order_params[0]][order_params[1]]:
        flags[order_params[0]][order_params[1]][order_params[2]]=dict()
    if order_params[3] not in flags[order_params[0]][order_params[1]][order_params[2]]:
        flags[order_params[0]][order_params[1]][order_params[2]][order_params[3]]=[]
    flags[order_params[0]][order_params[1]][order_params[2]][order_params[3]].append(cycles)
    #print(flag)
    #print(seed)
    #print('HiddenState')
    #print(hiddenstate)
    #print(dO)
    #print(T)
    #print(cycles)

#print(flags)

#PLOTTING PERFORMANCE
plt.xlabel(order_names[-1])
plt.ylabel('cycles/iteration')
plt.title('Impact of '+ order_names[-1])
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')

marker=0
color=0
style=0
for flag in flags.keys():
    for dO in flags[flag].keys():
        for T in flags[flag][dO]:
            x=[]
            y=[]
            for hiddenstate in flags[flag][dO][T]:
                x.append(hiddenstate)
                y.append(stats.median(flags[flag][dO][T][hiddenstate]))
                
            plt.plot(x,y, marker=markers[marker], color=colors[color], linestyle=styles[style], label= compiler + ' ' + flag+', '+str(dO)+', '+str(T))
            style=(style+1)%len(styles)
        style=0
        color=(color+1)%len(colors)
    color=0
    marker= (marker + 1) % len(markers)
plt.legend()

figure = plt.gcf()
figure.set_size_inches(8,4.5)
#plt.show()
timestr = time.strftime("%d-%m_%H;%M")
plt.savefig(file_name +'-' + timestr+ 'perf.png',dpi=200)
plt.clf()



#Plot base model (empty rooflie model)

ridge_point = scalar_pi/mem_beta
#print(ridge_point)
I = []
flops_byte_scal = []
flops_byte_vec = []
for i in np.arange(0.001,10000*ridge_point,0.01):
	I.append(i)
	flops_byte_scal.append(min(scalar_pi, i * mem_beta))
	flops_byte_vec.append(min(vector_pi, i * mem_beta))
	

plt.plot(I,flops_byte_scal)
plt.plot(I,flops_byte_vec)
plt.yscale('log')
plt.xscale('log')
figure = plt.gcf()
figure.set_size_inches(8,4.5)
plt.legend()
plt.title('Roofline Model')
plt.xlabel('Operational Intensity [flops/byte]')
plt.ylabel('Performance [flops/cycle]')



#plotting with operational intensity that counts every memory access (no chache model)
marker=0
color=0
style=0
for flag in flags.keys():
    for dO in flags[flag].keys():
        for T in flags[flag][dO]:
            x=[]
            y=[]
            for hiddenstate in flags[flag][dO][T]:
                params=to_order([flag,dO,T,hiddenstate], wichtiger_param)
                work=work_functions[aktuelle_version](params)
                memory=memory_functions[aktuelle_version](params)
                cycles=stats.median(flags[flag][dO][T][hiddenstate])
                x.append(work/memory)
                y.append(work/cycles)
            plt.plot(x,y, marker=markers[marker], color=colors[color], linestyle=styles[style], label=compiler + ' ' +  flag+', '+str(dO)+', '+str(T))
            style=(style+1)%len(styles)
        style=0
        color=(color+1)%len(colors)
    color=0
    marker= (marker + 1) % len(markers)





plt.legend()
#plt.show()
timestr = time.strftime("%d-%m_%H;%M")
plt.savefig(file_name +'-'+ timestr+"-roof.png",dpi=200)
plt.clf()


