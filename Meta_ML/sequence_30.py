import RNA
import sys
import os
import numpy as np
import pandas as pd
from collections import defaultdict, OrderedDict
from Bio.SeqUtils import MeltingTemp as mt
import itertools
from itertools import chain

def targetsequence(file1):
    sequence = open(file1, 'r')
    out = open("CRISPRi_machine_learning_30nt.csv", 'w')
    header = list(map(lambda x: 'A' + str(x), range(1, 31))) + list(map(lambda x: 'T' + str(x), range(1, 31))) + list(map(lambda x: 'G' + str(x), range(1, 31))) + list(map(lambda x: 'C' + str(x), range(1, 31))) + list(map(lambda x: 'AA'+str(x), range(1, 30))) + list(map(lambda x: 'TA' + str(x), range(1, 30))) + list(map(lambda x: 'GA' + str(x), range(1, 30))) + list(map(lambda x: 'CA' + str(x), range(1, 30))) + list(map(lambda x: 'AT' + str(x), range(1, 30))) + list(map(lambda x: 'TT' + str(x), range(1, 30))) + list(map(lambda x: 'GT' + str(x), range(1, 30))) + list(map(lambda x: 'CT' + str(x), range(1, 30))) + list(map(lambda x: 'AG' + str(x), range(1, 30))) + list(map(lambda x: 'TG' + str(x), range(1, 30))) + list(map(lambda x: 'GG' + str(x), range(1, 30))) + list(map(lambda x: 'CG' + str(x), range(1, 30))) + list(map(lambda x: 'AC' + str(x), range(1, 30))) + list(map(lambda x: 'TC' + str(x), range(1, 30))) + list(map(lambda x: 'GC' + str(x), range(1, 30))) + list(map(lambda x: 'CC' + str(x), range(1, 30)))
    out.write('ID' + ',')
    out.write(','.join(header))
    out.write(',' + 'Entropy' + ',' + 'Energy' + ',' + 'GCcount' + ',' + 'Gchigh' + ',' + 'GClow' + ',' + 'MeltingTemperature' + ',' + 'LFC' + ',' + 'A' + ',' + 'T' + ',' + 'G' + ',' + 'C' + ',' + 'AA' + ',' + 'AT' +
              ',' + 'AG' + ',' + 'AC' + ',' + 'CA' + ',' + 'CG' + ',' + 'CC' + ',' + 'CT' + ',' + 'GA' + ',' + 'GC' + ',' + 'GG' + ',' + 'GT' + ',' + 'TA' + ',' + 'TC' + ',' + 'TG' + ',' + 'TT' + '\n')
    sequence = sequence.readlines()
    del sequence[0]
    l = []
    dinuct = []
    ene = {}
    ent = {}
    ext = {}
    dt = {}
    nt = {}
    PosA = {}
    PosT = {}
    PosC = {}
    PosG = {}
    Tm = {}
    PosAA = {}
    PosAT = {}
    PosAG = {}
    PosAC = {}
    PosCA = {}
    PosCC = {}
    PosCG = {}
    PosCT = {}
    PosGA = {}
    PosGC = {}
    PosGG = {}
    PosGT = {}
    PosTA = {}
    PosTC = {}
    PosTG = {}
    PosTT = {}
    gc = {}
    gc_high = {}
    gc_low = {}
    nuc = ['A', 'T', 'G', 'C']
    nas = []
    merged_dict = defaultdict(list)
    sigma_dict = defaultdict(list)
    # ntscount = {'A':0, 'G':0, 'T':0, 'C':0}
    for j in range(0, len(nuc)):
        for t in range(0, len(nuc)):
            p = str(nuc[t]) + str(nuc[j])
            dinuct.append(p)
    for line in sequence:
        seq = line.strip().split(',')
    #	number = seq[2]
        indel = seq[12]
        complete_sequence = seq[5].upper()
        ids = seq[0]
        target_sequence = seq[6]
    #	start = complete_sequence.find(target_sequence)
    #	newstart = start - 21
    #	newend = int(len(target_sequence))  + 21
        Extended_sequence = complete_sequence

    #	print len(Extended_sequence),Extended_sequence
        ext[ids] = indel
        nt[ids] = []
        dt[ids] = []
        ent[ids] = []
        ene[ids] = []
        PosA[ids] = []
        PosT[ids] = []
        PosC[ids] = []
        PosG[ids] = []
        PosAA[ids] = []
        PosAT[ids] = []
        PosAG[ids] = []
        PosAC[ids] = []
        PosGA[ids] = []
        PosGC[ids] = []
        PosGG[ids] = []
        PosGT[ids] = []
        PosCA[ids] = []
        PosCG[ids] = []
        PosCC[ids] = {}
        PosCT[ids] = []
        PosTA[ids] = []
        PosTG[ids] = []
        PosTC[ids] = []
        PosTT[ids] = []
        Tm[ids] = []
        gc[ids] = []
        gc_high[ids] = []
        gc_low[ids] = []
        for y in range(0, len(nuc)):
            for x in range(0, len(Extended_sequence)):
                if nuc[y] in Extended_sequence[x]:
                    Count = str(1)
                    nt[ids].append(Count)
                else:

                    Count = str(0)
                    nt[ids].append(Count)

        for w in range(0, len(dinuct)):
            for z in range(0, len(Extended_sequence)-1):
                if dinuct[w] == Extended_sequence[z:z+2]:
                    Count = str(1)
                    dt[ids].append(Count)
                else:
                    Count = str(0)
                    dt[ids].append(Count)

        entropy = dict()
        Entropycal = []
        entropy_seq = target_sequence
        lentseq = len(entropy_seq)
        ntscount = {'A': 0, 'G': 0, 'T': 0, 'C': 0}
        for ant in nuc:
            ntscount[ant] = (entropy_seq.count(ant))/float((lentseq))
        for ant in nuc:
            if ntscount[ant] != 0:
                entropy[ant] = -(ntscount[ant]*np.log2(ntscount[ant]))
            else:
                entropy[ant] = 0

        entropySum = sum(entropy.values())
        entropySumR = round(entropySum, 1)
        ent[ids] = str(entropySumR)
        Energy = target_sequence
        Energycal = RNA.fold(Energy)[-1]
        Energycal = round(Energycal, 0)
        ene[ids] = str(Energycal)
        PosA[ids] = str(Extended_sequence.count('A'))
        PosC[ids] = str(Extended_sequence.count('C'))
        PosT[ids] = str(Extended_sequence.count('T'))
        PosG[ids] = str(Extended_sequence.count('G'))
        Tm[ids] = str(mt.Tm_NN(target_sequence))
        PosA[ids] = str(Extended_sequence.count('A'))
        PosT[ids] = str(Extended_sequence.count('T'))
        PosG[ids] = str(Extended_sequence.count('G'))
        PosC[ids] = str(Extended_sequence.count('C'))
        PosAA[ids] = str(Extended_sequence.count('AA'))
        PosAT[ids] = str(Extended_sequence.count('AT'))
        PosAG[ids] = str(Extended_sequence.count('AG'))
        PosAC[ids] = str(Extended_sequence.count('AC'))
        PosCA[ids] = str(Extended_sequence.count('CA'))
        PosCC[ids] = str(Extended_sequence.count('CC'))
        PosCG[ids] = str(Extended_sequence.count('CG'))
        PosCT[ids] = str(Extended_sequence.count('CT'))
        PosCC[ids] = str(Extended_sequence.count('CC'))
        PosGA[ids] = str(Extended_sequence.count('GA'))
        PosGC[ids] = str(Extended_sequence.count('GC'))
        PosGG[ids] = str(Extended_sequence.count('GG'))
        PosGT[ids] = str(Extended_sequence.count('GT'))
        PosTA[ids] = str(Extended_sequence.count('TA'))
        PosTC[ids] = str(Extended_sequence.count('TC'))
        PosTG[ids] = str(Extended_sequence.count('TG'))
        PosTT[ids] = str(Extended_sequence.count('TT'))
        gc_content = (target_sequence.count(
            'G') + target_sequence.count('C'))/float(len(target_sequence)) * 100
        gc_content = round(gc_content, 0)
        gc_count = (target_sequence.count('G') + target_sequence.count('C'))
        gc[ids] = str(gc_content)
        if gc_count < int(10):
            alpha = str(1)
        else:
            alpha = str(0)
        if gc_count >= int(10):
            beta = str(1)
        else:
            beta = str(0)

        gc_low[ids] = alpha
        gc_high[ids] = beta

    # for t,v in nt.items():
    #	v = ''.join[v]
    for key, value in nt.items():
        nt[key] = ','.join(value)
    for key, value in dt.items():
        dt[key] = ','.join(value)
#	print nt,dt

    dict_list = [nt, dt, ent, ene, gc, gc_high, gc_low, Tm, ext,PosA, PosT, PosG, PosC, PosAA, PosAT, PosAG,
                 PosAC, PosCA, PosCC, PosCG, PosCT, PosGA, PosGC, PosGG, PosGT, PosTA, PosTC, PosTG, PosTT]

    for dicts in dict_list:
        for k, v in dicts.items():
            merged_dict[k].append(v)
#	print merged_dict
    for key, value in (merged_dict.items()):
        sigma_dict[key] = ','.join((value))

    for key, value in sorted(sigma_dict.items()):
        out.write(str(key) + ',' + str(value) + '\n')
    out.close()

args = sys.argv[1]
result = targetsequence(args)
print (result)

