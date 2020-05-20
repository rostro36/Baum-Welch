wichtiger_param=3
parameter_names=['flag','hiddenstate','different observables','T']
aktuelle_version=0
file_name="../output_measures/model_output_20_5_10_55.txt"
#machine specs
scalar_pi=4
vector_pi=16
mem_beta=25



import matplotlib.pyplot as plt
import re
import statistics as stats
import time


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

def base_work(params):
    (flag,hiddenstate,differentObservables,T)=params
    #4∗N+ 7∗T∗N∗N+ 4∗T∗N+ 3∗(T−1)∗N∗N+ 3∗T+ 3+ 3∗T∗K∗N+ (T−1)∗N+ N∗N+ N∗K [Latex]
    return (4*hiddenstate+7*T*hiddenstate*hiddenstate+4*T*hiddenstate+3*(T-1)*hiddenstate*hiddenstate+3*T+3+3*T*hiddenstate*differentObservables+(T-1)*hiddenstate+hiddenstate*hiddenstate+hiddenstate*differentObservables)
work_functions=[base_work]

def base_memory(params):
    (flag,hiddenstate,differentObservables,T)=params
    #Compulsory misses only: 3*hiddenstate*T + hiddenstate*hiddenstate*T+T + hiddenstate + hiddenstate*hiddenstate + hiddenstate * differentObservables
    #12∗N+ 3 + 5N∗N∗T+ 9∗N∗T+ 4∗T+ 9∗N∗N∗(T−1)+ 3∗N∗(T−1)+ 2∗T∗K∗N+ N∗N+ N∗K [Latex]
    return  8*((12*hiddenstate+3+5*hiddenstate*hiddenstate*T+9*hiddenstate*T+4*T+9*hiddenstate*hiddenstate*(T-1)+3*hiddenstate*(T-1)+2*T*hiddenstate*differentObservables+hiddenstate*hiddenstate+hiddenstate*differentObservables))
memory_functions=[base_memory]


f = open(file_name)
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
plt.xlabel(order_names[-1])
plt.ylabel('cycles/iteration')
plt.title('Impact of '+order_names[-1])
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
                
            plt.plot(x,y, marker=markers[marker], color=colors[color], linestyle=styles[style], label= flag+','+str(dO)+','+str(T))
            style=(style+1)%len(styles)
        style=0
        color=(color+1)%len(colors)
    color=0
    marker= (marker + 1) % len(markers)
plt.legend()

figure = plt.gcf()
figure.set_size_inches(16,9)

timestr = time.strftime("%d-%m_%H;%M")
plt.savefig(timestr+".png")
plt.clf()

plt.xlabel('flops/byte')
plt.ylabel('flops/cycle')
plt.title('Roofline impact of '+order_names[-1])

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
    
xstart, xend = plt.xlim()

#Vector line
#Probably have to change in the future
end=min(xend,vector_pi/mem_beta)
#print(xend)
#print(vector_pi/mem_beta)
x=[xstart,end]
plt.fill_between(x, vector_pi, 0, alpha=0.2, color='y')
#Scalar line
plt.fill_between(x, scalar_pi, 0, alpha=0.2, color='g')
#Memory bound
print(vector_pi/mem_beta)
x=[xstart,end]
#x=[xstart, xend]
y=[xstart*mem_beta, vector_pi]
y=[xstart*mem_beta, end*mem_beta]
plt.fill_between(x, y, 0, alpha=0.2, color='b')
plt.yscale('log')
plt.xscale('log')
figure = plt.gcf()
figure.set_size_inches(16,9)
#plt.show()

plt.legend()
#plt.show()
timestr = time.strftime("%d-%m_%H;%M")
plt.savefig(timestr+"-roof.png")
plt.clf()
