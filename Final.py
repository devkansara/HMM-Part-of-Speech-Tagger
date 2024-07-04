from collections import defaultdict
import operator
import copy
import json
import toolz

def tup_to_str(x):
    return str(x)

def greedyDecoding(input_file, output_file):
    # Determine if we are working with 'dev' data (which includes POS tags)
    is_dev_data = 'data/dev' in input_file
    actual_tags = []  # This will only be used for 'dev' data
    predicted_tags = []
    prev_tag = "start"

    # Open the input file for reading; if output_file is specified, open it for writing
    with open(input_file, 'r') as f, open(output_file, 'w') if output_file else DummyContextManager() as out:
        i = 1
        for line in f:
            if line.strip():  # Check if line is not empty
                parts = line.strip().split('\t')
                if len(parts) == 3:  # 'dev' data with POS tags
                    index, word, actual_tag = parts
                    actual_tags.append(actual_tag)
                elif len(parts) == 2:  # 'test' data without POS tags
                    index, word = parts
                else:
                    raise ValueError("Unexpected line format")

                max_pred_tag = [-1, None]
                for state in tags:
                    if word in vocab_list:
                        em_prob = emission[(state, word)]
                    else:
                        em_prob = emission[(state, '<unk>')]
                    trans_prob = transition[(prev_tag, state)]
                    prob = em_prob * trans_prob
                    if prob > max_pred_tag[0]:
                        max_pred_tag = [prob, state]
                prev_tag = max_pred_tag[1]
                predicted_tags.append(max_pred_tag[1])

                if output_file:  # If output file is specified, write predictions
                    out.write(f"{i}\t{word}\t{max_pred_tag[1]}\n")
                    i += 1
            else:  # Reset for new sentence
                prev_tag = "start"
                if output_file:
                    out.write("\n")
                    i = 1

    if is_dev_data:
        return actual_tags, predicted_tags

class DummyContextManager:
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass

def getDevAccuracy(actual_tags, predicted_tags):
    true_pred = 0
    for i in range(len(actual_tags)):
        if actual_tags[i]==predicted_tags[i]:
            true_pred+=1
    return true_pred/len(predicted_tags)

def get_emission_probability(emission, state, word, vocab_list):
    if word in vocab_list:
        em_prob = emission.get((state, word), 0)
    else:
        em_prob = emission.get((state, "<unk>"), 0)
    return em_prob

def viterbiDecoding(data):
    with open(data, 'r') as f:
        i = 1
        for line in f:
            if line.strip():  
                parts = line.strip().split('\t')
    if len(parts) == 3:
        f = open(data, 'r')
        actual_tags = []
        predicted_tags = []
        prev_tag="start" 
        for line in f:
            get_indiv = line.split()
            if len(get_indiv)>0:
                actual_tags.append(get_indiv[2])
                if prev_tag=="start":
                    viterbi=[]
                    first_dict = {}
                    for state in tags:
                        if get_indiv[1] in vocab_list:
                            em_prob = emission[(state,get_indiv[1])]
                        else:
                            em_prob = emission[(state,'<unk>')]
                        trans_prob = transition[(prev_tag,state)]
                        prob = em_prob*trans_prob
                        first_dict[state] = (prob, prev_tag)
                    viterbi.append(copy.deepcopy(first_dict))
                    prev_tag="not start"
                else:
                    curr_dict = {}
                    for state in tags:
                        if get_indiv[1] in vocab_list:
                            em_prob = emission[(state,get_indiv[1])]
                        else:
                            em_prob = emission[(state,'<unk>')]
                        max_state_prob = [-1,None]
                        for prev_state_key, prev_state_val in viterbi[-1].items():
                            trans_prob = transition[(prev_state_key,state)]
                            prev_state_prob_val = prev_state_val[0]
                            final_prob = em_prob*trans_prob*prev_state_prob_val
                            if final_prob>max_state_prob[0]:
                                max_state_prob = [final_prob, prev_state_key]
                        curr_dict[state] = (max_state_prob[0], max_state_prob[1])
                    viterbi.append(copy.deepcopy(curr_dict))
            else:
                preds = []
                max_val = max(viterbi[len(viterbi)-1].values(), key = lambda x: x[0])
                preds.append(next(key for key,val in viterbi[len(viterbi)-1].items() if val==max_val))
                prev_state=max_val[1]
                for i in range(len(viterbi)-2, -1, -1):
                    preds.append(prev_state)
                    prev_state = viterbi[i][prev_state][1]
                preds.reverse()
                predicted_tags.extend(preds)
                prev_tag = "start"
        f.close()
        preds = []
        max_val = max(viterbi[len(viterbi)-1].values(), key = lambda x: x[0])
        preds.append(next(key for key,val in viterbi[len(viterbi)-1].items() if val==max_val))
        prev_state=max_val[1]
        for i in range(len(viterbi)-2, -1, -1):
            preds.append(prev_state)
            prev_state = viterbi[i][prev_state][1]
        preds.reverse()
        predicted_tags.extend(preds)
        
    elif len(parts) == 2:
        f = open(data, 'r')
        predicted_tags = []
        prev_tag="start" 
        for line in f:
            get_indiv = line.split()
            if len(get_indiv)>0:
                if prev_tag=="start":
                    viterbi=[]
                    first_dict = {}
                    for state in tags:
                        if get_indiv[1] in vocab_list:
                            em_prob = emission[(state,get_indiv[1])]
                        else:
                            em_prob = emission[(state,'<unk>')]
                        trans_prob = transition[(prev_tag,state)]
                        prob = em_prob*trans_prob
                        first_dict[state] = (prob, prev_tag)
                    viterbi.append(copy.deepcopy(first_dict))
                    prev_tag="not start"
                else:
                    curr_dict = {}
                    for state in tags:
                        if get_indiv[1] in vocab_list:
                            em_prob = emission[(state,get_indiv[1])]
                        else:
                            em_prob = emission[(state,'<unk>')]
                        max_state_prob = [-1,None]
                        for prev_state_key, prev_state_val in viterbi[-1].items():
                            trans_prob = transition[(prev_state_key,state)]
                            prev_state_prob_val = prev_state_val[0]
                            final_prob = em_prob*trans_prob*prev_state_prob_val
                            if final_prob>max_state_prob[0]:
                                max_state_prob = [final_prob, prev_state_key]
                        curr_dict[state] = (max_state_prob[0], max_state_prob[1])
                    viterbi.append(copy.deepcopy(curr_dict))
            else:
                preds = []
                max_val = max(viterbi[len(viterbi)-1].values(), key = lambda x: x[0])
                preds.append(next(key for key,val in viterbi[len(viterbi)-1].items() if val==max_val))
                prev_state=max_val[1]
                for i in range(len(viterbi)-2, -1, -1):
                    preds.append(prev_state)
                    prev_state = viterbi[i][prev_state][1]
                preds.reverse()
                predicted_tags.extend(preds)
                prev_tag = "start"
        f.close()
        preds = []
        max_val = max(viterbi[len(viterbi)-1].values(), key = lambda x: x[0])
        preds.append(next(key for key,val in viterbi[len(viterbi)-1].items() if val==max_val))
        prev_state=max_val[1]
        for i in range(len(viterbi)-2, -1, -1):
            preds.append(prev_state)
            prev_state = viterbi[i][prev_state][1]
        preds.reverse()
        predicted_tags.extend(preds)
        
    if len(parts) == 3:
        return actual_tags, predicted_tags
    elif len(parts) == 2:
        model_out = open('viterbi.out','w')
        f = open(data,'r')
        i=0
        for line in f:
            get_indiv = line.split()
            if len(get_indiv)>0:
                model_out.write(str(get_indiv[0])+"\t"+get_indiv[1]+"\t"+predicted_tags[i]+"\n")
                i+=1
            else:
                model_out.write("\n")
        f.close()
        model_out.close()

if __name__ == "__main__":
    f = open("data/train","r")
    count_dict = defaultdict(int)
    for line in f:
        get_words = line.split()
        if len(get_words)!=0:
            count_dict[get_words[1]]+=1
    f.close()

    unkw = 0
    for key,val in count_dict.items():
        if val<2:
            unkw += val

    sorted_count_list = sorted(count_dict.items(),key=operator.itemgetter(1), reverse=True)

    f = open("vocab.txt", "w")
    f.write('<unk>\t0\t'+str(unkw)+'\n')
    i=1
    vocab_count=0
    vocab_list = []
    thresh=2
    for word,count in sorted_count_list:
        if count>=thresh: 
            vocab_count += 1
            vocab_list.append(word)
            f.write(word+'\t'+str(i)+'\t'+str(count)+'\n')
            i+=1
    f.close()

    print("Threshold for rare words is: " + str(thresh))
    print("The total size of the vocabulary is "+str(vocab_count)+".")
    print("The total occurrences of the special token '<unk>' after replacement is "+str(unkw)+".")

    s_counts = defaultdict(int)
    e_counts = defaultdict(int)
    t_counts = defaultdict(int)
    prev_s = "start"
    s_counts["start"] += 1
    f = open("data/train", "r")
    for line in f:
        get_indiv = line.split()
        if(len(get_indiv)!=0):
            t_counts[(prev_s,get_indiv[2])]+=1
            if get_indiv[1] in vocab_list:
                e_counts[(get_indiv[2],get_indiv[1])]+=1
            else:
                e_counts[(get_indiv[2],'<unk>')]+=1
            s_counts[get_indiv[2]]+=1
            prev_s = get_indiv[2]
        else:
            prev_s="start"
            s_counts["start"] += 1
    f.close()

    transition = defaultdict(int)
    for key,val in t_counts.items():
        transition[key] = t_counts[key]/s_counts[key[0]]
        
    emission = defaultdict(int)
    for key,val in e_counts.items():
        emission[key] = e_counts[key]/s_counts[key[0]]

    print("The number of transition parameters in HMM:",str(len(transition.keys())))
    print("The number of emission parameters in HMM:",str(len(emission.keys())))

    tags = copy.deepcopy(list(s_counts.keys()))
    tags.remove('start') 
    transition_json = copy.deepcopy(transition)
    emission_json = copy.deepcopy(emission)
    transition_json = toolz.keymap(tup_to_str, transition_json)
    emission_json = toolz.keymap(tup_to_str, emission_json)
    total_dict = {'transition':transition_json, 'emission':emission_json}
    with open("hmm.json","w") as output_file:
        json.dump(total_dict, output_file, indent=4)



    actual_tags, predicted_tags = greedyDecoding('data/dev', None)

    dev_accuracy = getDevAccuracy(actual_tags,predicted_tags)
    print(f"Accuracy of Greedy Decoding on dev data: {(dev_accuracy*100)}%")

    greedyDecoding('data/test', 'greedy.out')


    actual_tags, predicted_tags = viterbiDecoding('data/dev')

    dev_accuracy = getDevAccuracy(actual_tags, predicted_tags)
    print(f"Accuracy of Viterbi Decoding on dev data: {(dev_accuracy*100)}%")

    viterbiDecoding('data/test')
