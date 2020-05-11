import matplotlib.pyplot as plt
import re
import statistics as stats
import time

file_name="N.txt"
#machine specs
scalar_pi=6
vector_pi=12
mem_beta=5



styles=['-','--','-.',':']
colors=['b', 'g', 'r', 'c', 'm', 'y', 'k']
markers=[".","v","^","<",">","1","2","3","4","8","s","p","P","*","h","H","+","x","X","D","d","|","_"]

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