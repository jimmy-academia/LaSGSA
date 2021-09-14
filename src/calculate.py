import os
import argparse

parser = argparse.ArgumentParser(description='a new parser')
parser.add_argument('--tasknames', type=str, help='comma seperated list of tasknames')
opt = parser.parse_args()

dirslist = opt.tasknames.split(',')

for taskdir in dirslist:
    print('== ', taskdir, ' ==')
    epslist = os.listdir(taskdir)
    max_epochs_list = []
    try:
        epslist.remove('all')
    except:
        pass
    try:        
        epslist.remove('log.txt')
    except:
        pass
    epslist = [int(x) for x in epslist]
    max_epochs_list.append(max(epslist))

    max_epochs = min(max_epochs_list) +1

    thresh = 75
    passcts = passqurs = 0
    for e in range(max_epochs):
        scs_qurs = [(x.split('-')[2], x.split('-')[-1][:-4]) for x in os.listdir(os.path.join(taskdir, str(e))) if 'y' in x]
        scs_qurs = [ (int(x[0]), x[1]) if x[0]!='' else (0, x[1]) for x in scs_qurs]
        scs = [x[0] for x in scs_qurs]
        if max(scs) >= thresh:
            passcts += 1
            overthresh = [x for x in scs_qurs if x[0] >= thresh]
            passqurs += int(min(overthresh, key=lambda x: x[1])[1])
        else:
            passqurs += 100000
        
    msg = ["thresh: ", thresh, "ASR: ", (passcts*10000//max_epochs)/100, "attempt query average", passqurs//max_epochs]
    print(*msg)

