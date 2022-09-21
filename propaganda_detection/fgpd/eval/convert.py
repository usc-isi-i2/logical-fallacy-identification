import pdb
from pathlib import Path
import pathlib 
import re
import sys
from operator import itemgetter

def read_data(directory):
    ids = []
    texts = []
    for f in directory.glob('*.txt'):
        id = f.name.replace('article', '').replace('.txt','')
        ids.append(id)
        texts.append(f.read_text())
    return ids, texts

def clean_text(articles, ids):
    texts = []
    for article, id in zip(articles, ids):
        sentences = article.split('\n')
        start = 0
        end = -1
        res = []
        for sentence in sentences:
           start = end + 1
           end = start + len(sentence)  # length of sequence
           if sentence != "": # if not empty line
               res.append([id, sentence, start, end])
        texts.append(res)
    return texts

def check_overlap(line_1, line_2):
    if line_1[2] > line_2[3] or line_1[3] < line_2[2]:
        return False
    else: 
        return True


def remove_duplicates(res):
    sorted_res = sorted(res, key=itemgetter(0,1,2,3))
    ans = []
    skip = 0
    for i, line_1 in enumerate(sorted_res):
        assert line_1 == sorted_res[i]
        for j, line_2 in enumerate(sorted_res[i+1:]):
            skip = 0
            if line_1[0] != line_2[0]:
                break 
            elif line_1[1] != line_2[1]:
                continue

            if check_overlap(line_1, line_2):
                if line_1[2] != line_2[2] or line_1[3] != line_2[3]: 
                    sorted_res[i+j+1][2] = min(line_1[2], line_2[2])
                    sorted_res[i+j+1][3] = max(line_1[3], line_2[3])
                skip = 1
                break
        if skip == 0:
            ans.append(line_1)
    return ans


def convert(ids, texts, ind, flat_texts, filename):
    with open(filename, 'r') as f1:
        output = []
        for line in f1:
            if len(line.split()) == 1: # if line is id 
                id = line.strip()
                continue
            if line != '\n': # In the same sentence 
                tmp = [id] + line.strip().split() # add id to line  
                if len(tmp) == 6: # num_task 2
                    tmp += [tmp[-2]]
                else:
                    tmp += [tmp[-(1+ind)]]
                output.append(tmp + [len(tmp[1])]) # add word length to line
            else: 
                output.append('\n')

    res = []
    aid = output[0][0]
    sub_list = [sentence for sentence in flat_texts if sentence[0] == aid]
    sub_dic = {sentence:(start, end) for _, sentence, start, end in sub_list}


    start = 0 
    end = -1 
    sentence = ""
    cur = 0 
    on = 0

    tmp_ans = []
    cur_tag = 'O'
    for line in output:
        if line != '\n':
            aid = line[0]
            sentence += line[1] + " "
            if line[-2] != 'O' and line[-2] != '<PAD>':
                if on == 0:
                    on = 1
                    cur_tag = line[-2]
                    start = cur
                    end = cur + line[-1]
                elif line[-2] == cur_tag:
                    end = cur + line[-1] 
                elif line[-2] != cur_tag:
                    tmp_ans.append([aid, cur_tag, start, end])
                    cur_tag = line[-2]
                    start = cur
                    end = cur + line[-1]
            else:
                if on:
                    tmp_ans.append([aid, cur_tag, start, end])
                    on = 0
            cur += line[-1] + 1
            
        else: 
            if on: 
                tmp_ans.append([aid, cur_tag, start, end])
                on = 0

            cur = 0
            sub_list = [sentence for sentence in flat_texts if sentence[0] == aid]
            sub_dic = {sentence:(start, end) for _, sentence, start, end in sub_list}

            if len(tmp_ans) and sentence[:-1] != "":
                s, e = sub_dic.get(sentence[:-1])
                for ans in tmp_ans:
                    ans[2] += s
                    ans[3] += s 
                    res.append(ans)
            sentence = ""
            tmp_ans = []
    return res 



if __name__ == "__main__":

    directory = pathlib.Path('./data/protechn_corpus_eval/test')
    ids, texts = read_data(directory)
    
    t_texts = clean_text(texts, ids)
    flat_texts = [sentence for article in t_texts for sentence in article]

    id_ind = [sentence[0] for sentence in flat_texts]
    
    fi = []
    if sys.argv[2] == 'bert':
        fi = convert(ids, texts, 0, flat_texts, sys.argv[1])
    elif sys.argv[2] == 'bert-joint' or sys.argv[2] == 'bert-granu' or sys.argv[2] == 'mgn':
        fi = convert(ids, texts, 1, flat_texts, sys.argv[1])

    res = remove_duplicates(fi)

    with open("./eval/official_prediction.txt", 'w') as f3:
        for i in res:
            f3.write("\t".join([i[0], i[1], str(i[2]), str(i[3])])+"\n")


   
