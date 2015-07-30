# 互联网工资预测


# 先保证安装了必要的软件库

# 首先，先抓取数据，目前我的记录是抓取了10w条，大概有100w~500w条记录
### 前置性能优化：
* 1.启动redis
```bash
$ redis-server
```
* 2.启动celery
```bash
$ celery -A Tasks worker --loglevel=info
```
### 运行
```bash
python DataCrawer.py
```
### 会在目录下面生成一个带时间标识的csv文件


# 接下来，生成数据训练集
### 运行
```bash
python DataAnalyser.py
```
### 上一步会自动处理数据并生成一个预测子名为clf.pkl存放于当前目录，看到clf知道怎么用的都不用说了吧，接下来主要调准精度。


## 训练模式 输入：
* 数据1: Job Description
* 数据2: 薪水数量


# 最后


## 预测模式 输入：
* 数据1: Job Description

### 输出：
* 数据1: 薪水数量
