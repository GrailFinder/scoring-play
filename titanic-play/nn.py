# nn class
import numpy as np
import scipy.special

class NeuralNetwork():
    
    def __init__(self, inp, hid, out, alpha):
        self.inp = inp
        self.hid = hid
        self.out = out

        # learning rate
        self.alpha = alpha

        # weights
        # input => hidden
        self.wih = np.random.normal(loc=0.0,
         scale=pow(self.inp, -0.5),
         size=(self.hid, self.inp))

        # hidden => output
        self.who = np.random.normal(loc=0.0,
         scale=pow(self.hid, -0.5),
         size=(self.out, self.hid))

        self.act_func = lambda x: scipy.special.expit(x)
    
    def train(self, inputs, targets):
        fin_out, hid_out = self.query(inputs)
        targets = np.array(targets, ndmin=2).T

        out_err = targets - fin_out
        hid_err = np.dot(self.who.T, out_err)

        # update weights
        self.who += self.alpha * np.dot(
            (out_err * fin_out
            *
            (1.0 - fin_out)),
            hid_out.T
        )

        self.wih += self.alpha * np.dot(
            (hid_err * hid_out
            *
            (1.0 - hid_out)),
            np.array(inputs, ndmin=2)
            )
        
        return out_err
    
    def query(self, inputs):
        # calculates the output
        inputs = np.array(inputs, ndmin=2).T
        hidin = np.dot(self.wih, inputs)
        # output from hidden layer
        hidout = self.act_func(hidin)

        # out
        fin_inp = np.dot(self.who, hidout)
        fin_out = self.act_func(fin_inp)
        return fin_out, hidout


if __name__ == "__main__":
    inp = hid = out = 3
    alpha = .3

    nn = NeuralNetwork(inp=inp, hid=hid, out=out, alpha=alpha)
    print(nn.query([1.0, 2, .5])[0])