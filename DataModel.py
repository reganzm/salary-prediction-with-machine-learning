#encoding: utf8
"""
Model
"""

from sqlalchemy import create_engine,MetaData,Table
from sqlalchemy import Column, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()
engine = create_engine('mysql://root@localhost/salary')

class Salary(Base):
    __tablename__ = 'salary'
    id = Column(Integer,primary_key=True,autoincrement=True)
    salary_num = Column(String(250))
    job_description = Column(String(1024))

def init_db():
    Base.metadata.create_all(engine)
