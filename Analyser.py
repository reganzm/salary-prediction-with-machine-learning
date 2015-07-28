#encoding: utf8
"""
Do some data-work
"""


from jieba.analyse import textrank

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
    for (w,f) in textrank(jd,topK=50,
                          withWeight=True,
                          allowPOS=['n','eng','v','a','i','ns','vn']):
        ret.append((w,f))
    wordlist = [r[0] for r in ret]
    flist = [r[1] for r in ret]
    return ret,wordlist,flist
