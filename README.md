# 互联网工资预测


## 训练模式 输入：
* 数据1：Job Description
* 数据2: 薪水数量

## 预测模式 输入：
* 数据1： Job Description

### 输出：
* 数据1： 薪水数量

### 用法：
* 1.启动redis
```bash
$ redis-server
```
* 2.启动celery
```bash
celery -A Tasks worker --loglevel=info
```
