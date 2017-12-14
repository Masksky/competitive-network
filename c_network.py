import numpy as np

def sigmoid(x,derivative=False):#激活函数
    return 1/(1+np.exp(-x))

def normalization(M):
    """
    对行向量进行归一化
    :param M:行向量：【dim=len(M)】
    :return: 归一化后的行向量M
    """
    M=M/np.sqrt(np.dot(M,M.T))
    return M

def normalization_all(N):
    """
    对矩阵进行归一化
    :param N: 矩阵：【m,n】
    :return: 归一化后的矩阵M_all:【m,n】
    """
    M_all=[]
    for i in range(len(N)):
        K=normalization(N[i])
        M_all.append(K)
    return M_all


class competitive_network(object):
    def __init__(self,x_dim,c_dim,a):
        W=np.random.rand(c_dim,x_dim)*(-2)+1
        self.W=normalization_all(W)
        self.a=a

    def forward_propagation(self,x):
        x=x.reshape(1,x.shape[0])
        z_layer=np.dot(self.W,x.T)
        a_layer=sigmoid(z_layer)
        argmax=np.where(a_layer==np.amax(a_layer))[0][0]
        return argmax

    def back_propagation(self,argmax,x):
        self.W[argmax] = self.a * (x - self.W[argmax])
        self.W[argmax]=normalization(self.W[argmax])
        self.a-=self.decay

    def train(self,X,num_item):
        X=np.array(X)
        self.decay=self.a/num_item
        for item in range(num_item):
            for i in range(X.shape[0]):
                argmax=self.forward_propagation(X[i])
                self.back_propagation(argmax,X[i])
    def prediction(self,x):
        argmax=self.forward_propagation(x)
        return argmax



