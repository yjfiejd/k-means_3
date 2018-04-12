# -*- coding:utf8 -*-
# @TIME : 2018/4/12 下午05:33
# @Author : yjfiejd
# @File : K-means_3.py

from numpy import *
import time
import matplotlib.pyplot as plt

# 1）先定义3个函数，第一个函数为导入数据函数
def loadDataSet(filename):
    dataSet = [] #定义一个数组用来存储导入的数据
    fr = open(filename) #打开文件
    for line in fr.readlines(): #便利文件的每一行，返回List
        curline = line.strip().split('\t') #对每一行格式化
        fltLine = map(float, curLine) #把每一行转为float类型
        datMat.append(fltLine) #使用append，把转换好对行，添加进dataSet数组

# 2）定义第二个函数，计算欧氏距离函数, 两个矩阵之间对距离，所以有sum
def distEclud(vector1, vector2):
    return sqrt(sum(power(vector2 - vector1, 2)))


# 3）定义随机对簇点，簇点来自与dataSet，注意随机簇点对对取值范围
def randCent(dataSet, k):
    n = shape(dataSet)[1] #获取矩阵列数,shape[行，列]
    centroids = mat(zeros((k, n))) #初始化簇矩阵，有K个质心, array转换为matrix格式
    for j in range(n): #遍历每一个列（特征—）
        minJ = min(dataSet[:, j]) #取每列中最小值
        maxJ = max(dataSet[:, j]) #取每一列中最大值
        rangeJ = float(maxJ - minJ) #计算簇对取值范围
        centroids[:, j] = mat(minJ + rangeJ*random.rand(k, 1)) #rand随机生成K行1列数组，其中值【0，1】
    return centroids

# 这个函数对另一种做法
#def initCentroids(dataSet, k):
#    numSamples, dim = dataSet.shape   #矩阵的行数、列数
#    centroids = zeros((k, dim))
#   for i in range(k):  #直接初始化簇点
#        index = int(random.uniform(0, numSamples))  #随机产生一个浮点数，然后将其转化为int型
#        centroids[i, :] = dataSet[index, :]
#    return centroids

# 4)定义Kmeans函数
def kMeans(dataSet, k):
    # 1）初始化存储点簇分配结果矩阵
    m = shape(dataSet)[0] #获取数据对行数，也就是样本点对个数
    clusterAssment = mat(zeros((m, 2))) #初始化簇对分类结果矩阵，m行，2列特征
    # 2) 初始化随机簇点
    centroids = randCent(dataSet, k) #初始化随机簇点
    clusterChange = True #定义循环判定条件，当簇点中心位置不再变化时，退出
    while clusterChange:
        clusterChange = False
        for i in range(m): #对每个样本计算最近对中心
            minDist = 1000000.0; #初始化minDist，第一次设置为很大对值
            minIndex = 0 #初始化索引，对应为，分到哪个簇
            for j in range(k): #对每个簇中心遍历，因为有多个，计算每个点到簇中心距离，把最小对进行归类
                distJI = distEclud(centroids[j, :], dataSet[i, :]) #计算距离
                if distJI < minDist:
                    minDist = distJI; #把小对值给minDist
                    minIndex = j #得到最近对中心，以及对应对j，簇号
                # 3) 更新cluster, 若所有对簇质心位置不变，则退出while循环
                # K个簇里第i个样本距离最小对标号和距离保存在clusterAssment
                if clusterAssment[i, :] != minIndex:
                    clusterChange = False
                    clusterAssement[i,:] = minIndex, minDist**2 #两个**表示对数minDist对平方
        #4）更新centroids，遍历簇心位置
        for j in range(k):
            #clusterAssment[:,0].A==j是找出矩阵clusterAssment中第一列元素中等于j的行的下标，返回的是一个以array的列表，第一个array为等于j的下标
            pointsInCluster = dataSet[nonzero(clusterAssment[:, 0].A == j)[0]]
            centroids[j, :] = mean(pointsInCluster, axis = 0)  #计算标注为j的所有样本的平均值

    print('finish')
    return centroids, clusterAssment

# 5)展示2D
def showCluster(dataSet, k , centroids, clusterAssment):
    m = shape(dataSet)[0] #行数
    n = shape(dataSet)[1] #列数
    if n != 2:
        print("sorry, can't draw because your data dimension is 2")
        return 1

    mark = ['or', 'ob', 'og', 'ok','^r', '+r', 'sr', 'dr', '<r', 'pr']
    if k > len(mark):
        print ("sorry, your K is too large")
        return 1
    # 绘制所有点对颜色
    for i in range(m):
        markIndex = int(clusterAssment[i, 0]) #为样本指定颜色
        plt.plot(dataSet[i, 0], dataSet[i, 1], mark[markIndex])

    mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']
    #绘制簇中心点对颜色
    for i in range(k):
        plt.plot(centroids[i, 0], centroids[i, i], mark[i], markersize = 12)

    plt.show()



## step 1: load data
import os
print ("step 1: load data..." )
dataSet = []   #列表，用来表示，列表中的每个元素也是一个二维的列表；这个二维列表就是一个样本，样本中包含有我们的属性值和类别号。
#与我们所熟悉的矩阵类似，最终我们将获得N*2的矩阵，
f = "/Users/a1/Desktop/算法实战/K-means/K-means_3/testSet.txt"
fileIn = open(f)  #是正斜杠
for line in fileIn.readlines():
    temp=[]
    lineArr = line.strip().split('\t')  #line.strip()把末尾的'\n'去掉
    temp.append(float(lineArr[0]))
    temp.append(float(lineArr[1]))
    dataSet.append(temp)
    #dataSet.append([float(lineArr[0]), float(lineArr[1])])#上面的三条语句可以有这条语句代替
fileIn.close()
## step 2: clustering...
print ("step 2: clustering..."  )
dataSet = mat(dataSet)  #mat()函数是Numpy中的库函数，将数组转化为矩阵
k = 4
centroids, clusterAssment = kMeans(dataSet, k)  #调用KMeans文件中定义的kmeans方法。

## step 3: show the result
print ("step 3: show the result..."  )
KMeans.showCluster(dataSet, k, centroids, clusterAssment)

