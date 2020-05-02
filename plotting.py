import matplotlib.pyplot as plt
import re
import statistics as stats
import time

f = open("model_output.txt")
text=f.read()
flags=dict()
#parse the file into dict of dicts
while(re.search('FLAG', text)):
    text=re.split('FLAG', text,1)[1:][0]
    flag=re.split('SEED', text, maxsplit=1)[0].strip()
    text=re.split('SEED', text,1)[1:][0]
    hiddenstate=int(re.split('HIDDENSTATE', text)[0].strip())
    text=re.split('HIDDENSTATE', text,1)[1:][0]
    dO=int(re.split('DIFFERENTOBSERVABLES', text)[0].strip())
    text=re.split('DIFFERENTOBSERVABLES', text,1)[1:][0]
    T=int(re.split('T', text)[0].strip())
    text=re.split('T ', text,1)[1:][0]
    cycles=re.split('cycles', text)[0].strip()
    cycles=re.split('Time:', cycles)[1].strip()
    cycles=int(re.split('\.',cycles)[0].strip())
    text=re.split('cycles', text,1)[1:][0]
    if flag not in flags:
        flags[flag]=dict()
    if dO not in flags[flag]:
        flags[flag][dO]=dict()
    if T not in flags[flag][dO]:
        flags[flag][dO][T]=dict()
    if hiddenstate not in flags[flag][dO][T]:
        flags[flag][dO][T][hiddenstate]=[]
    flags[flag][dO][T][hiddenstate].append(cycles)
    #print(flag)
    #print(hiddenstate)
    #print(dO)
    #print(T)
    #print(cycles)

#make plot variables
plt.xlabel('T')
plt.ylabel('cycles/iteration')
plt.title('Performance plot')
markers=[".","v","^","<",">","1","2","3","4","8","s","p","P","*","h","H","+","x","X","D","d","|","_"]
marker=0
colors=['b', 'g', 'r', 'c', 'm', 'y', 'k']
color=0
styles=['-','--','-.',':']
style=0
#unravel dicts and make lines
for flag in flags.keys():
    for dO in flags[flag].keys():
        for T in flags[flag][dO]:
            x=[]
            y=[]
            for hiddenstate in flags[flag][dO][T]:
                x.append(hiddenstate)
                y.append(stats.median(flags[flag][dO][T][hiddenstate]))
            plt.plot(x,y, marker=markers[marker], color=colors[color], linestyle=styles[style], label= flag+','+str(hiddenstate)+','+str(dO)+','+str(T))
            style=style+1
        style=0
        color+=1
    color=0
    marker+=1
plt.legend()
timestr = time.strftime("%d-%m_%H;%M")
plt.savefig(timestr+".png")
