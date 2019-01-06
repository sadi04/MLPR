import numpy as np
import copy

# sigmoid function
def nonlin(x, deriv=False):
    if (deriv == True):
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))

K = []
weights = []
delta_w = []
features_no = -1
total_class_no = 0
N = 0
meu = 0.9

def normalize(dataset):
    #print(dataset)
    arr = np.array(dataset)

    for i in range(len(dataset)):
        for j in range(1, len(dataset[0])):
            min_ = min(arr[:,j])
            #mean = np.mean(arr[:,j])
            max_ = max(arr[:,j])
            #std = np.std(arr[:,j],dtype=np.float64)
            #print(min_,max_)
            dataset[i][j] = (dataset[i][j]-min_)/(max_-min_);
    return dataset

def init():
    global features_no
    global total_class_no
    global N
    dataname = 'trainNN.txt'

    file = open(dataname,'r')

    traindata = file.read().split('\n')
    trainClass = []
    #reading data
    for i in range(len(traindata)):
        traindata[i] = traindata[i].split('\t')
        if '' in traindata[i]:
            traindata[i].remove('')
        #print(traindata[i])
        for j in range(len(traindata[i])-1):
            traindata[i][j] = float(traindata[i][j])
        traindata[i][len(traindata[i])-1] = int(traindata[i][len(traindata[i])-1])
        trainClass.append(traindata[i][len(traindata[i])-1])
        traindata[i].remove(traindata[i][len(traindata[i])-1])
        features_no = len(traindata[i])
        traindata[i].insert(0, 1.0)
        #traindata[i] = np.array(traindata[i])

    K.append(features_no)
    unique_classes = set(trainClass)
    total_class_no = len(unique_classes)
    N = len(traindata)
    #print(total_class_no)
    return traindata, trainClass
np.random.seed(123456)


def initweights():
    neurons_in_layers = [4, 5, 6, 3];
    for k in neurons_in_layers:
        K.append(k)

    #print(total_class_no)
    K.append(total_class_no)
    #print(K)
    for r in range(len(K)-1):
        temp = np.random.random((K[r]+1,K[r+1]+1))
        temp2 = np.zeros((K[r]+1,K[r+1]+1))
        weights.append(temp)
        delta_w.append(temp2)
        #print(r)
    #temp = np.random.random((K[len(K)-1] + 1, total_class_no+1))
    #weights.append(temp)
    #print(weights)

L = 0

y_frame = []


def backward_propagation():
    global L
    trainset, trainclass = init()
    #print(trainset)
    initweights()
    L = len(weights)

    #init delta

    delta = [np.zeros((2))]
    for i in range(L + 1):
        # print(i)
        # print(K[i], i)
        shape = (K[i])
        # print(shape)
        tmp = np.zeros(shape)
        delta.append(tmp)



    trainset = normalize(trainset)
    #print(trainset)


    #forward
    for it in range(100):
        e_i = []
        J = 0
        correct = 0
        for i in range(N):
            y_star = []
            er = []
            E_i = 0
            row = []
            row.append(trainset[i])
            #print(trainset[i])
            #print(i)
            row = np.array(row)
            tmp = row
            y_star.append(tmp)
            #for i in range(1):
            for r in range(L):
                shape = (1, len(weights[r][0]))
                yr = np.zeros(shape)
                np.matmul(tmp, weights[r], yr)
                yr[0][0] = 1
                for j in range(1, len(yr[0])):
                    yr[0][j] = nonlin(yr[0][j])
                tmp = copy.deepcopy(yr)
                y_star.append(tmp)
            y_frame.append(y_star)
            #exit(0)
            yt = trainclass[i]
            c = copy.deepcopy(tmp[0])
            c = np.delete(c, 0)
            #print(c)
            #print(np.argmax(c)+1)
            if yt == (np.argmax(c)+1):
                correct+=1
            for m in range(1, total_class_no+1):
                if m == yt:
                    er.append(tmp[0][m]-1)
                    E_i += 0.5*(tmp[0][m]-1)*(tmp[0][m]-1)
                else:
                    er.append(tmp[0][m])
                    E_i += 0.5*tmp[0][m]*tmp[0][m]
            e_i.append(er)
            J+=E_i


        #print(y_star[L-1][0])
        #print(y_frame[N-1])
        for i in range(N):
            #print(y_frame[i])
            for j in range(1, len(y_frame[i][L][0])):
                #print(e_i[i][j - 1])
                #print(y_frame[i][L][0][j])
                delta[L+1][j-1] = e_i[i][j - 1] * nonlin(y_frame[i][L][0][j], True)
                #delta.append(delta_j)

            #print(delta)
            for r in range(L, 1, -1):
                tomult = np.zeros((K[r]+1))
                #tomult[0] = np.random.uniform(0,1)
                for j in range(1, K[r]+1):
                    tomult[j] = delta[r+1][j-1]
                for j in range(1,K[r-1]+1):
                    error = tomult.dot(weights[r-1][j])
                    delta[r][j-1] = error*nonlin(y_frame[i][r-1][0][j], True)

            #print(delta)
            #exit(0)
            for r in range(1, L+1):
                dummy = np.random.random((1, K[r] + 1))
                tomult = np.zeros((K[r] + 1))
                #tomult[0] = np.random.uniform(0,1)
                for j in range(1, K[r] + 1):
                    tomult[j] = delta[r + 1][j - 1]
                dummy[0] = tomult
                #for j in range(1, K[r]+1):
                temp = copy.deepcopy(delta_w[r-1])
                np.matmul(y_frame[i][r-1].T,dummy,delta_w[r-1])
                #print(temp)
                #print(delta_w[r-1])
                delta_w[r-1] += temp
            for r in range(1, L + 1):
                weights[r - 1] -= meu * delta_w[r - 1]


        #print(weights)
        #print(delta_w)
        #exit(0)
        del y_frame[:]
        del e_i[:]
        print(J)
        print("Accuracy : ",correct)
    #print(weights)



backward_propagation()
'''
testname = 'testNN.txt'
file = open(testname,'r')

testdata = file.read().split('\n')
testClass = []
#reading data
for i in range(len(testdata)):
    testdata[i] = testdata[i].split('\t')
        #print(traindata[i])
    if '' in testdata[i]:
        testdata[i].remove('')
    for j in range(len(testdata[i])-1):
        testdata[i][j] = float(testdata[i][j])

    testdata[i][len(testdata[i])-1] = int(testdata[i][len(testdata[i])-1])
    #print(testdata[i])
    testClass.append(testdata[i][len(testdata[i])-1])
    testdata[i].remove(testdata[i][len(testdata[i])-1])
    features_no = len(testdata[i])
    testdata[i].insert(0, 1.0)

yt_star = []
correct = 0
testdata = normalize(testdata)
for i in range(len(testdata)):
    E_i = 0
    row = []
    row.append(testdata[i])
    # print(trainset[i])
    row = np.array(row)
    tmp = row
    yt_star.append(tmp)
    # for i in range(1):
    for r in range(L):
        shape = (1, len(weights[r][0]))
        yr = np.zeros(shape)
        np.matmul(tmp, weights[r], yr)
        # print("temp",tmp)
        # print(weights[r])
        #  print(yr)
        yr[0][0] = 1
        for j in range(1, len(yr[0])):
            #print(yr[0])
            yr[0][j] = nonlin(yr[0][j])
            print(yr[0])
        tmp = yr
        # print(tmp)
        yt_star.append(tmp)
        print(yr[0])
        c = copy.deepcopy(tmp[0])
        c = np.delete(c, 0)
        #print(c)
        # print(np.argmax(c)+1)
        yt = testClass[i]
        if yt == (np.argmax(c) + 1):
            correct += 1

print("Accuracy in test : ", correct * 100 / len(testdata))'''
