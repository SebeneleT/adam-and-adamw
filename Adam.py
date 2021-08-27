import math
import torch 
from torch.optim import Optimizer

class Adam(Optimizer):
    """
    implements Adam
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(Adam, self).__init__(params, defaults)
        
    def step(self):
        """
        Performs a single optimization step.
        """
        loss = None
        for group in self.param_groups:
            for p in group['params']:
                grad = p.grad.data
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    #initialize time step
                    state['step'] = 0
                    # 1st moment vector initialized as 0's
                    state['exp_avg'] = torch.zeros_like(p.data)
                    #2nd moment vector initialized as 0's
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                beta1, beta2 = group['betas']
                state['step'] += 1
                
                #functional that perfoms the adam algorithm


                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                # decaying moments
                exp_avg = torch.mul(exp_avg, beta1) + (1 - beta1)*grad
                exp_avg_sq = torch.mul(exp_avg_sq, beta2) + (1-beta2)*(grad*grad)
                
                denom = exp_avg_sq.sqrt() + group['eps']

                bias_correction1 = 1 / (1 - beta1 ** state['step'])
                bias_correction2 = 1 / (1 - beta2 ** state['step'])
                
                adapted_learning_rate = group['lr'] * bias_correction1 / math.sqrt(bias_correction2)

                p.data = p.data - adapted_learning_rate * exp_avg / denom
                
                
        return loss