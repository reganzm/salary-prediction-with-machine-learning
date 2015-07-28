#encoding: utf-8
"""
Nothing to say
"""
from __future__ import print_function
import re
import requests
from BeautifulSoup import BeautifulSoup as soup
from sqlalchemy.orm import sessionmaker
import sys

from Tasks import getContent
import DataModel
from DataModel import Salary
import datetime
import time

LAGOU_JOB_URL = 'http://www.lagou.com/jobs/{0}.html'
def getLagou(i,p={'job_description':('dd',{'class':'job_bt'}),
                  'job_request':('dd',{'class':'job_request'})}):
    return getContent(LAGOU_JOB_URL.format(i),patterns=p,ignore_pattern=['div',{'class':'position_del'}])

#DBSession = sessionmaker(DataModel.engine)
get_salary = r'(?P<low>[\d]{,3})k-(?P<high>[\d]{,3})k'
salary_reg = re.compile(get_salary)

SAVE_CSV = './save_file_' + str(datetime.datetime.fromtimestamp(time.time())) + '.csv'

def saveLagou(job):
    res = re.match(salary_reg,job['job_request'])
    try:
        with open(SAVE_CSV,'a+') as f:
            res = res.groupdict()
            salary = (int(res['low']) + int(res['high'])) / 2
            jd = job['job_description']
            f.write('{0},{1}\n'.format(salary,jd.encode('utf8')))
    except Exception,e:
        print(e)

if __name__=='__main__':
    for i in range(0,999999):
        try:
            saveLagou(getLagou(i))
            print('\r {0} id Finished\r'.format(i),file=sys.stdout)
        except Exception,e:
            pass
