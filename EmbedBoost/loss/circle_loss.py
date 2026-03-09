import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class CircleLoss(nn.Module):
    def __init__(self, m: float = 0.2, gamma: float = 20.0):
        super(CircleLoss, self).__init__()
        self.m = m
        self.gamma = gamma
        self.soft_plus = nn.Softplus()

    def forward(self, sp, sn, mp=0.2, mn=0.2, gamma_p=20.0, gamma_n=20.0):
        """
        sp: 正样本对的相似度 (similarity of positive pairs)
        sn: 负样本对的相似度 (similarity of negative pairs)
        """
        mp = mp if mp is not None else self.m
        gamma_p = gamma_p if gamma_p is not None else self.gamma

        mn = mn if mn is not None else self.m
        gamma_n = gamma_n if gamma_n is not None else self.gamma

        # 1. 定义最优目标值 (Optimum)
        op = 1 + mp
        on = -mn
        
        alpha_p = torch.relu(op - sp.detach()) 
        alpha_n = torch.relu(sn.detach() - on)

        # 3. 计算间隔 (Margin)
        delta_p = 1 - mp
        delta_n = mn

        # 4. 计算逻辑：alpha * (s - delta)
        logit_p = -gamma_p * alpha_p * (sp - delta_p)
        logit_n = gamma_n * alpha_n * (sn - delta_n)
     
        # 5. 结合成 Circle Loss 公式
        loss = torch.mean(self.soft_plus(torch.logsumexp(logit_n, dim=1) + torch.logsumexp(logit_p, dim=1)))
        
        return loss


def manual_softplus(x):
    # 数值稳定性优化：当 x 很大时，exp(x) 会溢出。
    # 此时 ln(1 + e^x) 几乎等于 x。
    if x > 20:
        return x
    return math.log(1 + math.exp(x))
#
def manual_logsumexp(input_list):
    # 找到最大值以保证数值稳定性
    max_val = max(input_list)
    
    # 计算 sum(exp(x - max))
    sum_exp = sum(math.exp(x - max_val) for x in input_list)
    
    return max_val + math.log(sum_exp)

def observe_circle_loss(sp, sn, m, gamma):
    op = 1 + m
    on = -m

    alpha_p = max(0, op - sp)
    alpha_n = max(0, sn - on)

    delta_p = 1 - m
    delta_n = m

    logit_p = -gamma * alpha_p * (sp - delta_p)
    logit_n = gamma * alpha_n * (sn - delta_n)
    
    loss = manual_softplus(manual_logsumexp([logit_n]) + manual_logsumexp([logit_p]))
    return logit_p, logit_n, loss



if __name__ == '__main__':
    m = 0.05
    gamma=5
    print(f"m={m}, gamma={gamma}")
    for i in range(1, 10):
        sp = i * 0.1
        sn = 1 - sp
        logit_p, logit_n, loss = observe_circle_loss(sp, sn, m, gamma)
        print(f"sp={sp:.2f}, sn={sn:.2f}, logit_p={logit_p:.2f}, logit_n={logit_n:.2f}, loss={loss:.2f}")
        

    # # 假设我们有一组相似度得分
    # sp_sample = torch.randn([4, 1], requires_grad=True)
    # sn_sample = torch.randn([4, 15], requires_grad=True)
    # criterion = CircleLoss(m=0.25, gamma=10)
    # loss = criterion(sp_sample, sn_sample)
    # loss.backward()
    # print(f"Loss: {loss.item()}")
