#encoding: utf-8
"""
Nothing to say
"""
import re
import requests
from BeautifulSoup import BeautifulSoup as soup
from sqlalchemy.orm import sessionmaker

from Tasks import getContent
import DataModel
from DataModel import Salary

LAGOU_JOB_URL = 'http://www.lagou.com/jobs/{0}.html'
def getLagou(i,p={'job_description':('dd',{'class':'job_bt'}),
                  'job_request':('dd',{'class':'job_request'})}):
    return getContent(LAGOU_JOB_URL.format(i),patterns=p,ignore_pattern=['div',{'class':'position_del'}])

DBSession = sessionmaker(DataModel.engine)
get_salary = r'(?P<low>[\d]{,3})k-(?P<high>[\d]{,3})k'
salary_reg = re.compile(get_salary)

def saveLagou(job):
    res = re.match(salary_reg,job['job_request'])
    try:
        session = DBSession()
        res = res.groupdict()
        salary = (int(res['low']) + int(res['high'])) / 2
        jd = job['job_description']
        s= Salary(salary_num=salary,job_description=jd)
        session.add(s)
        session.commit()
    except Exception,e:
        print(e)

if __name__=='__main__':
    print(getLagou(20000))
