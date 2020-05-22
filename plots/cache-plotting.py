wichtiger_param=3
parameter_names=['flag','hiddenstate','different observables','T']
aktuelle_version=0
file_name="reo"
#machine specs
scalar_pi=4
vector_pi=16
mem_beta=25



import matplotlib.pyplot as plt
import re
import statistics as stats
import time
import numpy as np


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
    return differentObservables*hiddenstate*hiddenstate+2*differentObservables*hiddenstate+6*hiddenstate*hiddenState*T-4*hiddenstate*hiddenstate+7*hiddenstate*T +4*hiddenstate+ 3*T-2

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


f = open(file_name+'-time.txt')
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

f = open(file_name+'-cache.txt')
text=f.read()
cache=dict()
#read the file
while(re.search('FLAG', text)):
    #[Ir, I1mr, ILmr, Dr, D1mr,    DLmr, Dw, D1mw, DLmw, Bc, Bcm, Bi, Bim] 
    #0  47391964 86   86  12110242 30259 0  4521166 26175 649 5028326 206065 0 0
    text=re.split('FLAG', text,1)[1:][0]
    flag=re.split('SEED', text, maxsplit=1)[0].strip()
    text=re.split('SEED', text,1)[1:][0]
    seed=int(re.split('HIDDENSTATE', text)[0].strip())
    text=re.split('HIDDENSTATE', text,1)[1:][0]
    hiddenstate=int(re.split('DIFFERENTOBSERVABLES', text)[0].strip())
    text=re.split('DIFFERENTOBSERVABLES', text,1)[1:][0]
    dO=int(re.split('T', text)[0].strip())
    text=re.split('T ', text,1)[1:][0]
    T=int(re.split('\n', text)[0].strip())
    text=re.split('\n', text,1)[1:][0]
    
    Ir=int(re.split('\s', text)[0].strip())
    text=re.split('\s', text,1)[1:][0]
    
    I1mr=int(re.split('[\s\t]', text)[0].strip())
    text=re.split('[\s\t]', text,1)[1:][0]
    
    ILmr=int(re.split('\s', text)[0].strip())
    text=re.split('\s', text,1)[1:][0]
    
    Dr=int(re.split('\s', text)[0].strip())
    text=re.split('\s', text,1)[1:][0]
    
    D1mr=int(re.split('\s', text)[0].strip())
    text=re.split('\s', text,1)[1:][0]
    
    DLmr=int(re.split('\s', text)[0].strip())
    text=re.split('\s', text,1)[1:][0]
    
    Dw=int(re.split('\s', text)[0].strip())
    text=re.split('\s', text,1)[1:][0]
    
    D1mw=int(re.split('\s', text)[0].strip())
    text=re.split('\s', text,1)[1:][0]
    
    DLmw=int(re.split('\s', text)[0].strip())
    text=re.split('\s', text,1)[1:][0]
    
    Bc=int(re.split('\s', text)[0].strip())
    text=re.split('\s', text,1)[1:][0]
    
    Bcm=int(re.split('\s', text)[0].strip())
    text=re.split('\s', text,1)[1:][0]
    
    Bi=int(re.split('\s', text)[0].strip())
    text=re.split('\s', text,1)[1:][0]
    
    Bim=int(re.split('\s', text)[0].strip())
    text=re.split('\s', text,1)[1:][0]
    
    parameters=[flag,hiddenstate,dO,T]
    parameter_names=['flag','hiddenstate','different observables','T']
    (order_params,order_names)=to_back([parameters,parameter_names],wichtiger_param)
    
    if order_params[0] not in cache:
        cache[order_params[0]]=dict()
    if order_params[1] not in cache[order_params[0]]:
        cache[order_params[0]][order_params[1]]=dict()
    if order_params[2] not in cache[order_params[0]][order_params[1]]:
        cache[order_params[0]][order_params[1]][order_params[2]]=dict()
    if order_params[3] not in cache[order_params[0]][order_params[1]][order_params[2]]:
        cache[order_params[0]][order_params[1]][order_params[2]][order_params[3]]=[]
    cache[order_params[0]][order_params[1]][order_params[2]][order_params[3]].append([Ir, I1mr, ILmr, Dr, D1mr, DLmr, Dw, D1mw, DLmw, Bc, Bcm, Bi, Bim])

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
figure.set_size_inches(16,9)
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
                print(params)
                work=work_functions[aktuelle_version](params)
                memory=memory_functions[aktuelle_version](params)
                cycles=stats.median(flags[flag][dO][T][hiddenstate])
                x.append(work/memory)
                y.append(work/cycles)
            plt.plot(x,y, marker=markers[marker], color=colors[color], linestyle=styles[style], label= flag+','+str(dO)+','+str(T))
            style=(style+1)%len(styles)
        style=0
        color=(color+1)%len(colors)
    color=0
    marker= (marker + 1) % len(markers)

#plotting with operational intensity that counts only compulsory misses
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
                print(params)
                work=work_functions[aktuelle_version](params)
                memory=base_memory_compulsory(params)
                cycles=stats.median(flags[flag][dO][T][hiddenstate])
                x.append(work/memory)
                y.append(work/cycles)
            plt.plot(x,y, marker=markers[marker], color=colors[color], linestyle=styles[style], label= flag+','+str(dO)+','+str(T))
            style=(style+1)%len(styles)
        style=0
        color=(color+1)%len(colors)
    color=0
    marker= (marker + 1) % len(markers)



plt.legend()
plt.show()
timestr = time.strftime("%d-%m_%H;%M")
plt.savefig(timestr+"-roof.png")
plt.clf()