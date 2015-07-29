#encoding: utf8
"""
Do some data-work
烦的时候写写注释 By H.YC
"""
from __future__ import print_function


import cPickle
import sys
import os
import numpy as np
import sklearn
from sklearn import linear_model, datasets, metrics
from sklearn.cross_validation import train_test_split
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline


def getWordwithWeight(sentence):
    """
    return
    """
    try:
        salary = float(sentence.split(',')[0])
        jd = sentence[sentence.index(',')+1:]
    except Exception,e:
        return None,None,None,None
    ret = []
    for (w,f) in textrank(jd,topK=30,
                          withWeight=True,
                          allowPOS=['n','eng','v','a','i','ns','vn']):
        ret.append((w,f))
    wordlist = [r[0] for r in ret]
    flist = [r[1] for r in ret]
    return ret,wordlist,flist,salary

def getCountedDict(count_dict,wl,fl,output='CountDict.pkl'):
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

def genCountedDict():
    from jieba.analyse import textrank
    count_dict = {}
    with open(sys.argv[1]) as f:
        index = 0
        for l in f:
            print('\r Processed {0} line of jd&salary. \r'.format(index),file=sys.stdout,end=" ")
            index += 1
            _,wl,fl,_ = getWordwithWeight(l)
            if wl == None:
                continue
            getCountedDict(count_dict,wl,fl)
    with open('CountedDict.pkl','a+') as pf:
        cPickle.dump(count_dict,pf)

def genWholeDict():
    with open('CountedDict.pkl') as f:
        cd = cPickle.load(f)
    scd = sorted(cd.items(),key=lambda k:k[1],reverse=True)
    return scd

def genXY(fName,bSave=False,limit=None):
    """
    Ofcause return X,y
    it quite complex
    hard to rewrite
    """
    X = []
    y = []
    whole_word_freq_list = genWholeDict()[:15000]
    print("whole dict load success!")
    whole_word_list = [wi[0] for wi in whole_word_freq_list]
    #whole_freq_list = [wi[1] for wi in whole_word_list]
    with open(fName) as f:
        index = 0
        for l in f:
            print('\r Processed {0} line of jd&salary. \r'.format(index),file=sys.stdout,end=" ")
            index += 1
            _,wl,fl,salary = getWordwithWeight(l)
            if wl == None:
                continue
            if not limit == None:
                if index > limit:
                    break
            Xi = np.zeros(len(whole_word_list))
            yi  = salary
            for i in range(len(wl)):
                try:
                    windex = whole_word_list.index(wl[i])
                    Xi[windex] = fl[i]
                except Exception,e: # dono deal with any exception!
                    pass
            X.append(Xi.tolist())
            y.append(yi)
    if bSave == True:
        with open('X_train.pkl','a+') as Xf:
            cPickle.dump(X,Xf)
        with open('y_train.pkl','a+') as yf:
            cPickle.dump(y,yf)
    return X,y

def Train():
    """
    Train Function
    """
    with open('X_train.pkl') as Xf:
        X = cPickle.load(Xf)
    with open('y_train.pkl') as yf:
        y = cPickle.load(yf)
    X_train, X_test, Y_train, Y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        random_state=0)
    logistic = linear_model.LogisticRegression()
    rbm = BernoulliRBM(random_state=0, verbose=True)
    rbm.learning_rate = 0.03
    rbm.n_iter = 100
    rbm.n_components = 1000
    logistic.C = 6000.0
    clf = Pipeline(steps=[('rbm', rbm), ('logistic', logistic)])
    #clf.fit(X_train,Y_train)
    logistic_classifier = linear_model.LogisticRegression(C=100.0)
    logistic_classifier.fit(X_train, Y_train)
    print("Logistic regression using raw pixel features:\n%s\n" % (
    metrics.classification_report(
        Y_test,
        logistic_classifier.predict(X_test))))
    #print("Logistic regression using RBM features:\n%s\n" % (
    #metrics.classification_report(
    #    Y_test,
    #    clf.predict(X_test))))
    with open('clf.pkl','a+') as clf_f:
        cPickle.dump(clf,clf_f)


if __name__ == '__main__':
    # generate a counted dictionary named 'CountDict.pkl'
    if len(sys.argv) < 2:
        raise Exception("Wrong Argument number!")
    print(sys.argv)
    if os.path.exists('./CountedDict.pkl') == False:
        genCountedDict()
    # generate X,y ====> Hard work here.
    # 1. build an N-length array, generate X with this array
    # 2. paired with y
    if os.path.exists('X_train.pkl') == False:
        genXY(sys.argv[1],bSave=True,limit=20000)
    Train()
