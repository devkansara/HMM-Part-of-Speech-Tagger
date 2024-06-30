import sys
import os
import argparse


parser = argparse.ArgumentParser(prog="eval", description="Evaluation of HW3")
parser.add_argument("-g", "--gold", help="gold standard file")
parser.add_argument("-p", "--pred", help="prediction file")
args = parser.parse_args()

pred = args.pred
gold = args.gold


total = 0
corr = 0
try:
    with open(gold, 'r') as gf, open(pred, 'r') as pf, open('outputs.txt', 'w') as offile:
        pf_count = 0
        pf_lines = pf.readlines()
        for gline in gf.readlines():
            gline = gline.strip()
            if len(gline) == 0:
            
                    # handles missing blank line issue
                    if len(pf_lines[pf_count].strip()) == 0:
                        pf_count+=1
                    continue
            tokens = gline.split()
            
            # print tokens
            gidx = tokens[0]
            gword = tokens[1]
            glabel = tokens[2]
            
            pline = pf_lines[pf_count]
            pline = pline.strip()        
            tokens = pline.split()
            
            pword = tokens[1] if len(tokens) > 1 else ''    # handles blank line issue if wrong dev dataset is used
            plabel = tokens[2] if len(tokens) > 2 else ''   # handles missing label issue

            #if gidx != pidx:
            #    print('warning: index mismatch: {}, {}'.format(gidx, pidx))

            #if gword != pword:
            #    print('warning: word mismatch: {}, {}'.format(gword, pword))

            total += 1
            if glabel == plabel:
                corr += 1

            pf_count += 1
            
            offile.write(' '.join([gword, pword, glabel, plabel, '\n']))

except:
    print(repr(gline), repr(pline), total)
    
print("total: {}, correct: {}, accuracy: {:.2f}%".format(total, corr, float(corr) * 100 / total))
