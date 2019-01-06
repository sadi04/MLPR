import numpy as np
import copy

n = 4
l = 2

np.random.seed(2)

infile = 'input.txt'
file = open(infile, 'r+')
params = file.read().split(' ')
_hk1 = float(params[0])
_hk2 = float(params[1])
_nmu = float(params[2])
_ncov = float(params[3])

current_signal = 0
current_state = [0, 0, 0]
state_x = {}
state_mu = {}
state_cov = {}
state_transition = {}
state_transition[0] = [0, 4]
state_transition[1] = [0, 4]
state_transition[2] = [1, 5]
state_transition[3] = [1, 5]
state_transition[4] = [2, 6]
state_transition[5] = [2, 6]
state_transition[6] = [3, 7]
state_transition[7] = [3, 7]
from_state = {}
from_state[0] = [0, 1]
from_state[1] = [2, 3]
from_state[2] = [4, 5]
from_state[3] = [6, 7]
from_state[4] = [0, 1]
from_state[5] = [2, 3]
from_state[6] = [4, 5]
from_state[7] = [6, 7]

omega_2d = []
prior = []

def probGenerate(xvec, muvec, covmat, d):
    diff = xvec - muvec
    power = np.dot(diff.T.dot(np.linalg.inv(covmat)), diff)
    prob = np.exp(-0.5*power.item(0))
    prob /= np.sqrt(pow(2*np.pi,d)*(abs(np.linalg.det(covmat))))
    return prob

def train():
    global current_state
    global current_signal
    global state_x
    global state_mu
    global state_cov
    global prior
    filename = 'train.txt'

    file = open(filename, 'r+')
    traindata = file.read()

    _nk = np.random.normal(_nmu, _ncov, len(traindata)+1)

    for i in range(8):
        x_k = []
        x_k_1 = []
        _xvec = []
        _xvec.append(x_k)
        _xvec.append(x_k_1)
        state_x[i] = _xvec

    for k in range(len(traindata)):
        current_signal = int(traindata[k])
        if not (current_signal == 0 or current_signal == 1): continue
        current_state[-1] = current_signal
        current_state = current_state[-1:]+current_state[:-1]
        current_class = 4*current_state[0] + 2*current_state[1] + current_state[2]

        _xk = _hk1*current_state[1] + _hk2*current_state[0] + _nk[k]
        _xk1 = _hk1 * current_state[2] + _hk2 * current_state[1] + _nk[k-1]

        state_x[current_class][0].append(_xk)
        state_x[current_class][1].append(_xk1)

    for i in range(8):
        _xmuvec = []
        xk_mu = np.mean(state_x[i][0])
        xk1_mu = np.mean(state_x[i][1])
        _xmuvec.append(xk_mu)
        _xmuvec.append(xk1_mu)

        _xcovvec = []
        _xcovvec.append(state_x[i][0])
        _xcovvec.append(state_x[i][1])

        state_mu[i] = _xmuvec
        state_cov[i] = np.cov(np.array(_xcovvec))

    for j in range(8):
        val = len(state_x[j][0])
        val /= len(traindata)
        print(val)
        prior.append(val)



def test():
    global current_state
    filename = 'test.txt'

    file = open(filename, 'r+')
    testdata = file.read()

    _ntk = np.random.normal(_nmu, _ncov, len(testdata)+1)

    for k in range(len(testdata)):
        current_signal = int(testdata[k])
        if not (current_signal == 0 or current_signal == 1): continue
        current_state[-1] = current_signal
        current_state = current_state[-1:] + current_state[:-1]

        _xk = _hk1 * current_state[0] + _hk2 * current_state[1] + _ntk[k]
        _xk1 = _hk1 * current_state[1] + _hk2 * current_state[2] + _ntk[k-1]

        xvec = []
        xvec.append(_xk)
        xvec.append(_xk1)
        xvec = np.asmatrix(xvec)

        omega_i = []
        if k == 0:
            for i in range(8):
                muvec = []
                muvec.append(state_mu[i])
                muvec = np.asmatrix(muvec)
                covvec = copy.deepcopy(state_cov[i])
                covvec = np.asmatrix(covvec)
                prob = probGenerate(xvec.T, muvec.T, covvec, 2)

                distance = np.log(prior[i]) + np.log(prob)

                omega_i.append(distance)
            omega_2d.append(omega_i)
        else:
            for i in range(8):
                muvec = []
                muvec.append(state_mu[i])
                muvec = np.asmatrix(muvec)
                covvec = copy.deepcopy(state_cov[i])
                covvec = np.asmatrix(covvec)
                prob = probGenerate(xvec.T, muvec.T, covvec, 2)

                distance = np.log(prob)

                last_max_distance = -10000

                for l in from_state[i]:
                    if omega_2d[k-1][l] > last_max_distance:
                        last_max_distance = omega_2d[k-1][l]
                distance += last_max_distance
                omega_i.append(distance)
            omega_2d.append(omega_i)

    curr_state = 0
    correct = 0
    generated_bits = []

    for i in range(len(testdata)-1, -1, -1):
        if i == len(testdata)-1:
            _max = - 10000
            for j in range(8):
                if omega_2d[i][j] > _max:
                    _max = omega_2d[i][j]
                    curr_state = j
        else:
            _max = -10000
            for j in state_transition[curr_state]:
                if omega_2d[i][j] > _max:
                    _max = omega_2d[i][j]
                    curr_state = j
        generated_bit = curr_state // 4;
        if generated_bit == int(testdata[i]):
            correct+=1
        generated_bits.append(generated_bit)

    print(correct / len(generated_bits))
    generated_bits.reverse()

train()

current_state = [0, 0, 0]

test()
