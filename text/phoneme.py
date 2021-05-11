import sys

try:
    sys.path.append('./text')
except:
    pass

list_phones = []
with open('./text/phoneme.txt', 'r', encoding='utf-8') as rf:
    lines = rf.read().split('\n')
    for line in lines:
        if len(line.strip()) > 0:
            list_phones.append(line.strip())