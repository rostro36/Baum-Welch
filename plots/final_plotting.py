import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import re
import statistics as stats
import time
import numpy as np
import glob


folder = "../output_measures/"
file_name1 = "bla-do*"
full_name1 =glob.glob(folder + file_name1)
file_name2 = "bla-hs*"
full_name2 =glob.glob(folder + file_name2)
file_name3 = "bla-T*"
full_name3 =glob.glob(folder + file_name3)
file_name4 = "cop-do*"
full_name4 =glob.glob(folder + file_name4)
file_name5 = "cop-hs*"
full_name5 =glob.glob(folder + file_name5)
file_name6 = "cop-T*"
full_name6 =glob.glob(folder + file_name6)
file_name7 = "reo-do*"
full_name7 =glob.glob(folder + file_name7)
file_name8 = "reo-hs*"
full_name8 =glob.glob(folder + file_name8)
file_name9 = "reo-T*"
full_name9 =glob.glob(folder + file_name9)
file_name10 = "stb-do*"
full_name10 =glob.glob(folder + file_name10)
file_name11 = "stb-hs*"
full_name11 =glob.glob(folder + file_name11)
file_name12 = "stb-T*"
full_name12 =glob.glob(folder + file_name12)
file_name13 = "umdhmm-do*"
full_name13 =glob.glob(folder + file_name13)
file_name14 = "umdhmm-hs*"
full_name14 =glob.glob(folder + file_name14)
file_name15 = "umdhmm-T*"
full_name15 =glob.glob(folder + file_name15)
file_name16 = "vec-do*"
full_name16 =glob.glob(folder + file_name16)
file_name17 = "vec-hs*"
full_name17 =glob.glob(folder + file_name17)
file_name18 = "vec-T*"
full_name18 =glob.glob(folder + file_name18)


file_names = [full_name1,full_name2,full_name3,full_name4,full_name5,full_name6,full_name7,full_name8,full_name9,full_name10,full_name11,full_name12,full_name13,full_name14,full_name15,full_name16,full_name17,full_name18]
files = ['bla','bla','bla','cop','cop','cop', 'reo', 'reo', 'reo', 'stb', 'stb', 'stb','umdhmm','umdhmm','umdhmm', 'vec', 'vec', 'vec']

#parameter im Performance plot auf x-achse
wichtiger_param = 2
#0 = flag	1 = states #2 = dO	3 = T

param=['hs','dO', 'T']

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
comp_colors = ['#030764','#054907','#840000','#6488ea','#6fc276','#ff796c','#fffe7a']
markers=[".","v","^","<",">","1","2","3","4","8","s","p","P","*","h","H","+","x","X","D","d","|","_"]


def translate(p,order):

    hiddenstate = 0
    dO = 0
    T = 0
    for i in [1,2,3]:
        if(order[i] == 'hiddenstate'):
            hiddenstate = p[i]
        elif(order[i] == 'different observables'):
            dO = p[i]
        else:
            T = p[i]
    return (p[0],hiddenstate,dO,T)

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

all_flags =[]
all_order_names = []

wichtiger_param = 2
for i in range(len(file_names)):
    print(file_names[i][0])
    
    f = open(file_names[i][0])
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
    all_flags.append(flags)
    all_order_names.append(order_names)
    wichtiger_param = wichtiger_param -1
    if(wichtiger_param == 0):
        wichtiger_param = 3

    
wichtiger_param = 2

for i in range(len(file_names)):
    
    #PLOTTING PERFORMANCE
    plt.xlabel(all_order_names[i][-1])
    plt.ylabel('cycles/iteration')
    plt.title('Impact of '+ all_order_names[i][-1]+" " + files[i])
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')

    marker=0
    color=0
    style=0


    flags = all_flags[i]
    #for flag in flags.keys():
    if(i < 3):
        flag = 'g-O2'
    else:
        flag = 'g-O2 -mfma'
    for p1 in flags[flag].keys():
        for p2 in flags[flag][p1]:
            x=[]
            y=[]
            for p3 in flags[flag][p1][p2]:
                x.append(p3)
                y.append(stats.median(flags[flag][p1][p2][p3]))
                
            plt.plot(x,y, marker=markers[marker], color=comp_colors[color], linestyle=styles[style], label= compiler + ' ' + flag+', '+str(dO)+', '+str(T))
            style=(style+1)%len(styles)
        style=0
        color=(color+1)%len(colors)
    # color=0
    marker= (marker + 1) % len(markers)
    plt.legend()

    figure = plt.gcf()
    figure.set_size_inches(8,4.5)
    #plt.show()
    timestr = time.strftime("%d-%m_%H;%M")
    plt.savefig(files[i] +'-'+param[wichtiger_param-1]+ '-cycles'+'-' + timestr+'.png',dpi=200)
    plt.clf()
    wichtiger_param = wichtiger_param-1
    if(wichtiger_param <1):
        wichtiger_param = 3

wichtiger_param = 2
for i in range(len(file_names)):
    
    if(i<3):
        continue
    if(i == 12 or i == 13 or i==14):
        continue
    #PLOTTING PERFORMANCE
    plt.xlabel(all_order_names[i][-1])
    plt.ylabel('Performance')
    plt.title('Performance: Impact of '+ all_order_names[i][-1] + " " + files[i])
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')

    marker=0
    color=0
    style=0


    flags = all_flags[i]
    #for flag in flags.keys():
        #print(flag)
    flag = 'g-O2 -mfma'
    for p1 in flags[flag].keys():
        for p2 in flags[flag][p1]:
            #print(T)
            x=[]
            y=[]
            for p3 in flags[flag][p1][p2]:
                #print(hiddenstate)
 
                params = translate([flag,p1,p2,p3],all_order_names[i])
                #params=(flag,hiddenstate,dO,T)               
                (flag,hiddenstate,differentObservables,T)=params
               
                work=work_functions[files[i]](params)
                x.append(p3)
                y.append(work/stats.median(flags[flag][p1][p2][p3]))
                
            plt.plot(x,y, marker=markers[marker], color=comp_colors[color], linestyle=styles[style], label= compiler + ' ' + flag+', '+str(dO)+', '+str(T))
            style=(style+1)%len(styles)
        style=0
        color=(color+1)%len(colors)
   # color=0
    marker= (marker + 1) % len(markers)
    plt.legend()

    figure = plt.gcf()
    figure.set_size_inches(8,4.5)
    #plt.show()
    timestr = time.strftime("%d-%m_%H;%M")
    plt.savefig(files[i] +'-'+ param[wichtiger_param-1]+ '-perf' +'-'+ timestr+'.png',dpi=200)
    plt.clf()

    wichtiger_param = wichtiger_param-1
    if(wichtiger_param <1):
        wichtiger_param = 3
