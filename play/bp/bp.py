"""
说明：本程序为BP神经网络初步模型，网络结构包含输入层、隐层、输出层，目前输入层为31个节点，隐层设计为2层（节点数为12,6，可更改），
输出层为1（NOX）。采用梯度下降算法，误差反向传播，更新权重；根据训练目标或训练次数作为程序结束运行条件。
"""
# 导入相关数据库，实现对excel数据表格的访问，利用Numpy模块实现数据结构矩阵化运算，提高效率。最后对网络输出利用画图模块
# 进行可视化显示。
import numpy as np
import xlrd
import matplotlib.pyplot as plt
# 定义激活函数，完成隐层输入（隐层输入数据为：输入层与隐层的权值矩阵*输入层数据矩阵）到隐层输出的传递任务
def logsig(x):    #定义激活函数为logsig（）
    return 1/(1+np.exp(-x))    #定义激活函数logsig运算公式为：1/(1+e(-x))
#定义函数访问excel表格，提出其中数据并生成矩阵，便于网络对数据的读取和运算
def excel_to_matrix(path):
    table = xlrd.open_workbook(path).sheets()[0]  # 获取第一个sheet表
    row = table.nrows  # 行数，获取该sheet中的有效列数
    col = table.ncols  # 列数，获取该sheet中的有效行数
    datamatrix = np.zeros((row, col))  # 生成一个nrows行ncols列，且元素均为0的初始矩阵
    for x in range(col):  #样本总体训练的次数，把列表中有效行数中的每个元素代入变量x，然后执行缩进块的语句。
        cols = np.matrix(table.col_values(x))  # 把list转换为矩阵进行矩阵操作
        datamatrix[:, x] = cols  # 按列把数据存进矩阵中
    return datamatrix   #循环结束后返回矩阵数据
datafile = u'BP模拟练习.xlsx'  #从给定路径中读取Excel中的数据
excel_to_matrix(datafile)    #给矩阵matrix(datafile)中填充数据
#读入数据 负荷、煤量、风量及风门开度
fuhe=excel_to_matrix(datafile)[:,0]
coal_31=excel_to_matrix(datafile)[:,1]
coal_32=excel_to_matrix(datafile)[:,2]
coal_33=excel_to_matrix(datafile)[:,3]
coal_34=excel_to_matrix(datafile)[:,4]
coal_35=excel_to_matrix(datafile)[:,5]
coal_36=excel_to_matrix(datafile)[:,6]
lutang_1jiao=excel_to_matrix(datafile)[:,7]
sofa_1jiao=excel_to_matrix(datafile)[:,8]
AA_1=excel_to_matrix(datafile)[:,9]
A_1=excel_to_matrix(datafile)[:,10]
AB_1=excel_to_matrix(datafile)[:,11]
B_1=excel_to_matrix(datafile)[:,12]
BC_1=excel_to_matrix(datafile)[:,13]
C_1=excel_to_matrix(datafile)[:,14]
CD_1=excel_to_matrix(datafile)[:,15]
D_1=excel_to_matrix(datafile)[:,16]
CE_1=excel_to_matrix(datafile)[:,17]
E_1=excel_to_matrix(datafile)[:,18]
EE_1=excel_to_matrix(datafile)[:,19]
F_1=excel_to_matrix(datafile)[:,20]
FF_1=excel_to_matrix(datafile)[:,21]
SOFA1_12=excel_to_matrix(datafile)[:,22]
SOFA1_3=excel_to_matrix(datafile)[:,23]
SOFA1_4=excel_to_matrix(datafile)[:,24]
SOFA1_5=excel_to_matrix(datafile)[:,25]
SOFA1_6=excel_to_matrix(datafile)[:,26]
SOFA1_7=excel_to_matrix(datafile)[:,27]
zongfengliang=excel_to_matrix(datafile)[:,28]
chaya=excel_to_matrix(datafile)[:,29]
rukouyangliang=excel_to_matrix(datafile)[:,30]
#输出为NOX值
nox=excel_to_matrix(datafile)[:,31]
#将读取的数据写入，作为网络的输入参数
samplein = np.mat([fuhe,coal_31,coal_32,coal_33,coal_34,coal_35,coal_36,lutang_1jiao,sofa_1jiao,AA_1,A_1,AB_1,B_1,BC_1,C_1,CD_1,D_1,
                  CE_1,E_1,EE_1,F_1,FF_1,SOFA1_12,SOFA1_3,SOFA1_4,SOFA1_5,SOFA1_6,SOFA1_7,zongfengliang,chaya,rukouyangliang]) #31
sampleout = np.mat([nox])      #将nox作为网络输出
#输入、输出参数归一化[-1,1]区间
sampleinminmax = np.array([samplein.min(axis=1).T.tolist()[0],samplein.max(axis=1).T.tolist()[0]]).transpose()#2*31
#从输入矩阵选出最大值和最小值，生成一个2*31的数组
sampleoutminmax = np.array([sampleout.min(axis=1).T.tolist()[0],sampleout.max(axis=1).T.tolist()[0]]).transpose()#2*1
#从输出矩阵选出最大值和最小值，生成一个2*1的数组
sampleinnorm = ((2*(np.array(samplein.T)-sampleinminmax.transpose()[0])/(sampleinminmax.
                transpose()[1]-sampleinminmax.transpose()[0])-1).transpose())
# 2*（输入数据-输入数据中的最小值）/（输入数据中的最大值-输入数据中的最小值）-1得出一组区间[0,1]的30000*31输入数组
sampleoutnorm = ((2*(np.array(sampleout.T).astype(float)-sampleoutminmax.transpose()[0])/
                  (sampleoutminmax.transpose()[1]-sampleoutminmax.transpose()[0])-1).transpose())
# 2*（输出数据-输出数据中的最小值）/（输出数据中的最大值-输出数据中的最小值）-1得出一组区间[0,1]的30000*1输出数组
#给输出样本添加噪音，为了防止网络过度拟合
noise = 0.03 * np.random.rand(sampleoutnorm.shape[0], sampleoutnorm.shape[1])
sampleoutnorm += noise  #给归一化后的输出数组添加噪音
#定义网络模型的参数 迭代次数 学习率 样本数量 输入节点 输出节点 隐含层节点
maxepochs = 40000   #最大迭代次数为40000次
learnrate = 0.0001  #学习率为0.0001
errorfinal = 1.75   #预期误差为1.75
samnum =3000        #样本数量为3000
indim = 31          #输入神经元个数为31
outdim = 1          #输出神经元个数为1
hiddenunitnum1 = 12 #第一个隐藏层神经元个数为12
hiddenunitnum2 = 6  #第二个隐藏层神经元个数为6
#设置网络输入到隐层权值与阈值w1、b1；w2、b2;隐层到输出层权值与阈值w3、b3
w1 = 0.5*np.random.rand(hiddenunitnum1,indim)-0.1  ##初始化输入层与隐含层之间的权值，区间为[0,0.4]的随机12*31浮点数组
b1 = 0.5*np.random.rand(hiddenunitnum1,1)-0.1               #第一个隐含层各个神经元的阀值
w2 = 0.5*np.random.rand(hiddenunitnum2,hiddenunitnum1)-0.1  #初始化第一个隐藏层与第二个隐含层之间的权值
b2 = 0.5*np.random.rand(hiddenunitnum2,1)-0.1               #第二个隐藏层各个神经元的阀值
w3 = 0.5*np.random.rand(outdim,hiddenunitnum2)-0.1          #初始化第二个隐藏层与输出层层之间的权值
b3 = 0.5*np.random.rand(outdim,1)-0.1                       #输出层各个神经元的阀值

errhistory = []                                             #建立误差收录位置

#开始训练模型

for i in range(maxepochs):                                                                #按迭代次数对数据开始训练
    hiddenout1 = logsig((np.dot(w1,sampleinnorm).transpose()+b1.transpose())).transpose() #第一个隐含层输出
    hiddenout2 = logsig((np.dot(w2,hiddenout1).transpose()+b2.transpose())).transpose()   #第二个隐含层输出
    networkout =(np.dot(w3,hiddenout2).transpose()+b3.transpose()).transpose()            #最终输出
    err = sampleoutnorm - networkout                                                      #每次训练误差值
    sse = sum(sum(err ** 2))                                                              #对训练误差值进行函数处理
    errhistory.append(sse)                                                                #将误差值see添加到误差收录位置

    # 更新权重
    if sse < errorfinal:    #如果训练误差小于目标值，跳出循环，终止训练
        break
    #利用训练误差，由后向前逐层对权值、阈值进行更改
    delta3 = err
    delta2 = np.dot(w3.transpose(),delta3)*hiddenout2*(1-hiddenout2)
    delta1 = np.dot(w2.transpose(),delta2)*hiddenout1*(1-hiddenout1)
    dw3 = np.dot(delta3,hiddenout2.transpose())
    db3 = np.dot(delta3,np.ones((samnum,1)))
    dw2 = np.dot(delta2,hiddenout1.transpose())
    db2 = np.dot(delta2,np.ones((samnum,1)))
    dw1 = np.dot(delta1,sampleinnorm.transpose())
    db1 = np.dot(delta1,np.ones((samnum,1)))
    w3 += learnrate*dw3
    b3 += learnrate*db3
    w2 += learnrate*dw2
    b2 += learnrate*db2
    w1 += learnrate*dw1
    b1 += learnrate*db1

#绘制误差曲线图，对网络训练进行可视化处理。
plt.rcParams['font.sans-serif']=['SimHei']    #实现图中显示中文
plt.rcParams['axes.unicode_minus']=False      #实现图中显示负号
errhistory10 = np.log10(errhistory)           #将误差值进行log函数处理，方便绘制曲线图
minerr = min(errhistory10)                    #取误差值里面的最小值
plt.plot(errhistory10)                        #对误差值进行画图
plt.plot(range(0, i+1000, 5000), [minerr] * len(range(0, i+1000, 5000)))
#获取曲线图设置坐标轴标签和刻度
ax = plt.gca()    #设置图像边框
ax.set_yticks([-2, -1, 0, minerr,1, 2, ]) #定义坐标轴刻度
ax.set_yticklabels([u'$10^{-2}$',u'$10^{-1}$',u'$10^{1}$',str(('%.4f' % np.power(10,minerr))),u'$10^{2}$'])
#调整刻度位置
ax.set_xlabel('迭代次数')
ax.set_ylabel('误差log()')
ax.set_title('误差记录')
plt.savefig('误差迭代图.png',dpi=700)
plt.show()
plt.close()

#实现预测输出和实际输出对比图

hiddenout2 = logsig((np.dot(w2,hiddenout1).transpose()+b2.transpose())).transpose()#隐层输出
networkout = (np.dot(w3,hiddenout2).transpose()+b3.transpose()).transpose()
#反归一化
diff = sampleoutminmax[:,1]-sampleoutminmax[:,0]
networkout2 = (networkout+1)/2
networkout2[0] = networkout2[0]*diff[0]+sampleoutminmax[0][0]
sampleout = np.array(sampleout)

fig,axes = plt.subplots(nrows=1,ncols=1,figsize=(12,10))
line1, =axes.plot(networkout2[0][2890:2898],'k',marker = u'$\circ$')
line2, = axes.plot(sampleout[0][2890:2898],'r',markeredgecolor='b',marker = u'$\star$',markersize=9)
#输出预测与实际误差结果（绝对值）

print('网络预测NOX：',networkout2[0][2890:2898])
print('样本数据NOX：',sampleout[0][2890:2898])
print('预测与实际误差:',abs(sampleout[0][2890:2898]-networkout2[0][2890:2898]))

#添加图例说明
axes.legend((line1,line2),('预测值','实际值'),loc = 'upper left')
#设置坐标轴尺寸
yticks = [300,320,340,360,380,400,420,440,460,480,500]
ytickslabel = [u'$3$',u'$3.2$',u'$3.4$',u'$3.6$',u'$3.8$',u'$4$',u'$4.2$',u'$4.4$',u'$4.6$',u'$4.8$',u'$5$']
axes.set_yticks(yticks)
axes.set_yticklabels(ytickslabel)
axes.set_ylabel(u'NOX$(10^2)$')

xticks = range(0,10,1)
xtickslabel = range(0,10,1)
axes.set_xticks(xticks)
axes.set_xticklabels(xtickslabel)
axes.set_xlabel(u'样本')
axes.set_title('NOX_BP神经网络')
fig.savefig('sim仿真预测结果.png',dpi=500,bbox_inches='tight')
plt.show()
plt.close()
NUM=input('请输入当前寻优数据的行数:')
print('预测NOX：',networkout2[0][NUM])

