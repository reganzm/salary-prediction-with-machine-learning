#encoding: utf8
"""
Do some data-work
"""


from jieba.analyse import textrank
import cPickle
import sys

def getWordwithWeight(sentence):
    """
    return
    """
    try:
        salary = float(sentence.split(',')[0])
        jd = sentence[sentence.index(',')+1:]
    except Exception,e:
        return None,None,None
    ret = []
    for (w,f) in textrank(jd,topK=30,
                          withWeight=True,
                          allowPOS=['n','eng','v','a','i','ns','vn']):
        ret.append((w,f))
    wordlist = [r[0] for r in ret]
    flist = [r[1] for r in ret]
    return ret,wordlist,flist

def getCountedDict(count_dict,wl,fl,output='CountDict.pkl',maxNum=5*10000):
    """
    word list
    frequence list
    """
    for i in range(len(wl)):
        if count_dict.get(wl[i]) == None:
            count_dict[wl[i]] = fl[i]
        else:
            count_dict[wl[i]] += fl[i]
    return count_dict

if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise Exception("Wrong Argument number!")
    count_dict = {}
    with open(sys.argv[1]) as f:
        for l in f:
            _,wl,fl = getWordwithWeight(l)
            getCountedDict(count_dict,wl,fl)
    with open('CountedDict.pkl','a+') as pf:
        cPickle.dump(count_dict,pf)
