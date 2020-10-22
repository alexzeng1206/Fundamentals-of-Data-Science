import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures   #只是用来数据预处理
##################################
##线性回归，直接使用矩阵运算求解w
def Linear(X,Y):
    p=np.dot(X.T,X)
    p=np.linalg.pinv(p)
    p=np.dot(p,X.T)
    w=np.dot(p,Y)
    w=w.reshape(-1, 1)
    return w
###########################
##岭回归，直接使用矩阵运算求解w
def L2(X,Y,c):
    p=np.dot(X.T,X)
    a=c*np.identity(5)
    p=p+a
    p=np.linalg.pinv(p)
    p=np.dot(p,X.T)
    w=np.dot(p,Y)
    w=w.reshape(-1, 1)
    return w
#################################

plt.rcParams['figure.dpi'] = 300 #分辨率
plt.ylim(-30,125) #设定画图的y轴范围
x=np.linspace(0,8)
y=-x*x+17*x+3   #目标函数f(x)
test_x=np.array([1,3,5,6]).reshape(4,1)    #生成测试集数据
test_y=np.array([19,45,63,69]).reshape(4,1)
plt.scatter(test_x,test_y,label='test')    #画出测试集数据
plt.plot(x,y,label='$f(x)$')         #画出目标函数曲线
x=np.array([1.7,3,4.2,5])           #生成训练集数据
y=np.array([25,56,52,68])
plt.scatter(x,y,label='train')
x=x.reshape(4,1)
y=y.reshape(4,1)
poly =PolynomialFeatures(4)    #这个函数是为了映射为多项式，即把x映射为{1，x,...,x的n次方}向量
X=poly.fit_transform(x)
w=Linear(X,y)     #线性回归求出w
xx = np.linspace(0.5,5.6)
yy = w[0]+w[1]*xx+w[2]*xx*xx+w[3]*xx*xx*xx+w[4]*xx*xx*xx*xx   #把w映射回来，表达成多项式
plt.plot(xx, yy,label='Linear')
w2=L2(X,y,1)     #岭回归求出w2
xx = np.linspace(0,6)
yy = w2[0]+w2[1]*xx+w2[2]*xx*xx+w2[3]*xx*xx*xx+w2[4]*xx*xx*xx*xx #把w2映射回来，表达成多项式
plt.plot(xx, yy,label='$L_2$')
############################################
x1=poly.fit_transform(test_x)  #映射为高维多项式
y1=np.dot(x1,w)
err=np.linalg.norm(test_y-y1, ord=2, axis=None, keepdims=False)  #先求二范数，然后平分再除以样本个数打印出均方误差
print("线性回归均方误差为：",err*err/len(test_y))
y2=np.dot(x1,w2)
err1=np.linalg.norm(test_y-y2, ord=2, axis=None, keepdims=False)
print("岭回归均方误差为：",err1*err1/len(test_y))
print("线性回归拟合函数从低次到高次的多项式系数为:",w.reshape(5))
print("岭回归（lambda=1）拟合函数从低次到高次的多项式系数为:",w2.reshape(5))
plt.legend()
plt.show()

###############下面是对比不同正则化系数lambda对拟合曲线的影响
print("----------------------------------------------------")
plt.ylim(-30,125)
x=np.linspace(0,8)
y=-x*x+17*x+3
test_x=np.array([1,3,5,6]).reshape(4,1)
test_y=np.array([19,45,63,69]).reshape(4,1)
plt.scatter(test_x,test_y,label='test')
plt.plot(x,y,label='$f(x)$')
x=np.array([1.7,3,4.2,5])
y=np.array([25,56,52,68])
plt.scatter(x,y,label='train')
x=x.reshape(4,1)
y=y.reshape(4,1)
poly =PolynomialFeatures(4)    #这个函数是为了映射为多项式
X=poly.fit_transform(x)
w=Linear(X,y)
xx = np.linspace(0.5,5.6)
yy = w[0]+w[1]*xx+w[2]*xx*xx+w[3]*xx*xx*xx+w[4]*xx*xx*xx*xx
#plt.plot(xx, yy,label='Linear')
w2=L2(X,y,0.01)
xx = np.linspace(0,6)
yy = w2[0]+w2[1]*xx+w2[2]*xx*xx+w2[3]*xx*xx*xx+w2[4]*xx*xx*xx*xx
plt.plot(xx, yy,label='$L_2,\lambda =0.01$')
print("lambda=0.01时从低次到高次的多项式系数为:",w2.reshape(5))
y2=np.dot(x1,w2)
err1=np.linalg.norm(test_y-y2, ord=2, axis=None, keepdims=False)
print("均方误差为：",err1*err1/len(test_y))
w2=L2(X,y,0.1)
xx = np.linspace(0,6)
yy = w2[0]+w2[1]*xx+w2[2]*xx*xx+w2[3]*xx*xx*xx+w2[4]*xx*xx*xx*xx
plt.plot(xx, yy,label='$L_2,\lambda =0.1$')
print("lambda=0.1时从低次到高次的多项式系数为:",w2.reshape(5))
y2=np.dot(x1,w2)
err1=np.linalg.norm(test_y-y2, ord=2, axis=None, keepdims=False)
print("均方误差为：",err1*err1/len(test_y))
w2=L2(X,y,1)
xx = np.linspace(0,7.5)
yy = w2[0]+w2[1]*xx+w2[2]*xx*xx+w2[3]*xx*xx*xx+w2[4]*xx*xx*xx*xx
plt.plot(xx, yy,label='$L_2,\lambda =1$')
print("lambda=1时从低次到高次的多项式系数为:",w2.reshape(5))
y2=np.dot(x1,w2)
err1=np.linalg.norm(test_y-y2, ord=2, axis=None, keepdims=False)
print("均方误差为：",err1*err1/len(test_y))
w2=L2(X,y,10)
xx = np.linspace(0,7.5)
yy = w2[0]+w2[1]*xx+w2[2]*xx*xx+w2[3]*xx*xx*xx+w2[4]*xx*xx*xx*xx
plt.plot(xx, yy,label='$L_2,\lambda =10$')
print("lambda=10时从低次到高次的多项式系数为:",w2.reshape(5))
y2=np.dot(x1,w2)
err1=np.linalg.norm(test_y-y2, ord=2, axis=None, keepdims=False)
print("均方误差为：",err1*err1/len(test_y))
############################################
x1=poly.fit_transform(test_x)
y1=np.dot(x1,w)
err=np.linalg.norm(test_y-y1, ord=2, axis=None, keepdims=False)
#print("线性回归均方误差为：",err*err/len(test_y))
y2=np.dot(x1,w2)
err1=np.linalg.norm(test_y-y2, ord=2, axis=None, keepdims=False)
plt.legend()
plt.show()