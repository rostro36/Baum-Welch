import matplotlib.pyplot as plt
import re
import statistics as stats
import time

file_name="../output_measures/05-31.21:38:21-cache.txt"
#machine specs
scalar_pi=6
vector_pi=12
mem_beta=5

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
    return differentObservables*hiddenstate*hiddenstate+2*differentObservables*hiddenstate+6*hiddenstate*hiddenState*T-4*hiddenstate*hiddenstate+7*hiddenstate*T +4*hiddenstate+ 3*T-2

def base_work(params):
    (flag,hiddenstate,differentObservables,T)=params
    return (4*hiddenstate+7*T*hiddenstate*hiddenstate+4*T*hiddenstate+3*(T-1)*hiddenstate*hiddenstate+3*T+3+3*T*hiddenstate*differentObservables+(T-1)*hiddenstate+hiddenstate*hiddenstate+hiddenstate*differentObservables)
work_functions=dict()
work_functions['std']=base_work
work_functions['stb']=base_work
work_functions['cop']=base_work
work_functions['reo']=reo_work
work_functions['vec']=reo_work

def reo_memory(params):
    (flag,hiddenstate,differentObservables,T)=params
    return 3*differentObservables*hiddenstate*hiddenstate+4*differentObservables*hiddenstate+6*hiddenstate*hiddenstate*T+11*hiddenstate*T+6*hiddenstate+5*T-8+3*hiddenstate*hiddenstate

def base_memory(params):
    (flag,hiddenstate,differentObservables,T)=params
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


f = open(file_name)
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

    
f = open(file_name+'-cache.txt')
text=f.read()
cache=dict()
#read the file
while(re.search('FLAG', text)):
    #[Ir, I1mr, ILmr, Dr, D1mr,    DLmr, Dw, D1mw, DLmw, Bc, Bcm, Bi, Bim] 
    #0  47391964 86   86  12110242 30259 0  4521166 26175 649 5028326 206065 0 0
    text=re.split('FILE', text,1)[1:][0]
    file=re.split('FLAG', text, maxsplit=1)[0].strip()
    text=re.split('FLAG', text,1)[1:][0]
    flag=re.split('SEED', text, maxsplit=1)[0].strip()
    text=re.split('SEED', text,1)[1:][0]
    seed=int(re.split('N', text)[0].strip())
    text=re.split('N', text,1)[1:][0]
    n=int(re.split('\n', text)[0].strip())
    text=re.split('\n', text,1)[1:][0]
    Ir=0
    I1mr=0
    ILmr=0
    Dr=0
    D1mr=0
    DLmr=0
    Dw=0
    D1mw=0
    DLmw=0
    Bc=0
    Bcm=0
    Bi=0
    Bim=0
    while(text.strip()!='' and text.strip()[:3]!='DAS'):
        junk=int(re.split('\s', text)[0].strip())
        text=re.split('\s', text,1)[1:][0]
        
        Ir+=int(re.split('\s', text)[0].strip())
        text=re.split('\s', text,1)[1:][0]
        
        I1mr+=int(re.split('[\s\t]', text)[0].strip())
        text=re.split('[\s\t]', text,1)[1:][0]
        
        ILmr+=int(re.split('\s', text)[0].strip())
        text=re.split('\s', text,1)[1:][0]
        
        Dr+=int(re.split('\s', text)[0].strip())
        text=re.split('\s', text,1)[1:][0]
        
        D1mr+=int(re.split('\s', text)[0].strip())
        text=re.split('\s', text,1)[1:][0]
        
        DLmr+=int(re.split('\s', text)[0].strip())
        text=re.split('\s', text,1)[1:][0]
        
        Dw+=int(re.split('\s', text)[0].strip())
        text=re.split('\s', text,1)[1:][0]
        
        D1mw+=int(re.split('\s', text)[0].strip())
        text=re.split('\s', text,1)[1:][0]
        
        DLmw+=int(re.split('\s', text)[0].strip())
        text=re.split('\s', text,1)[1:][0]
        
        Bc+=int(re.split('\s', text)[0].strip())
        text=re.split('\s', text,1)[1:][0]
        
        Bcm+=int(re.split('\s', text)[0].strip())
        text=re.split('\s', text,1)[1:][0]
        
        Bi+=int(re.split('\s', text)[0].strip())
        text=re.split('\s', text,1)[1:][0]
        
        Bim+=int(re.split('\s', text)[0].strip())
        print(text)
        if text.strip()!='':
            text=re.split('\s', text,1)[1:][0]
        
    
    parameters=[flag,seed,n]
    parameter_names=['flag',seed,'N']
    
    if file not in cache:
        cache[file]=dict()
    if flag not in cache[file]:
        cache[file][flag]=dict()
    if n not in cache[file][flag]:
        cache[file][flag][n]=[]
    cache[file][flag][n].append([Ir, I1mr, ILmr, Dr, D1mr, DLmr, Dw, D1mw, DLmw, Bc, Bcm, Bi, Bim])
    
plt.xlabel('N')
plt.ylabel('cycles/iteration')
plt.title('Different implementations')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')

marker=0
color=0
style=0
for file in flags.keys():
    for flag in flags[file].keys():
        x=[]
        y=[]
        for n in flags[file][flag]:
            x.append(n)
            y.append(stats.median(flags[file][flag][n]))
        plt.plot(x,y, marker=markers[marker], color=colors[color], linestyle=styles[style], label= file[:3]+','+str(flag))
        color+=1
    color=0
    marker+=1
plt.legend()
timestr = time.strftime("%d-%m_%H;%M")
plt.savefig(timestr+".png")
plt.clf()

plt.xlabel('flops/byte')
plt.ylabel('flops/cycle')
plt.title('Roofline impact of N')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')



##ROOFLINE NOCH NICHT GUT, DA NOCH MEMORY FUNKTIONEN FEHLEN
marker=0
color=0
style=0
for file in flags.keys():
    for flag in flags[file].keys():
        x=[]
        y=[]
        for n in flags[file][flag]:
            params=(None,n,n,n*n)
            (flag,hiddenstate,differentObservables,T)=params
            work=work_functions[file](params)
            memory=memory_functions[file](params)
            cycles=stats.median(flags[file][flag][n])
            x.append(work/memory)
            y.append(work/cycles)
        plt.plot(x,y, marker=markers[marker], color=colors[color], linestyle=styles[style], label= file[:3]+','+str(flag))
        color+=1
    color=0
    marker+=1

                         
xstart, xend = plt.xlim()
#Memory bound
x=[xstart,vector_pi/mem_beta]
y=[xstart*mem_beta, vector_pi]
plt.fill_between(x, y, 0, alpha=0.2, color='r')
#Vector line
x=[xstart,xend]
plt.fill_between(x, vector_pi, 0, alpha=0.2, color='y')
#Scalar line
plt.fill_between(x, scalar_pi, 0, alpha=0.2, color='b')



plt.legend()
timestr = time.strftime("%d-%m_%H;%M")
plt.savefig(timestr+"-roof.png")
plt.clf()
