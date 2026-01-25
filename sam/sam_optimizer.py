"""
业务问题：实现 SAM (Sharpness-Aware Minimization) 优化器。
实现逻辑：
1. SAM 通过在参数空间中寻找一个邻域，使得该邻域内的最大损失最小化，从而找到更平滑的极小值点。
2. 算法分为两步：
   a. 在梯度方向上移动一个步长 rho，计算该处的梯度。
   b. 回到原始位置，利用第二步的梯度执行真实的参数更新。
"""

import torch

class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        """寻找扰动后的位置"""
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                # 存储原始梯度方向的位移
                e_w = p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum
                self.state[p]["e_w"] = e_w

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        """回到原始位置，利用扰动点的梯度进行更新"""
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.sub_(self.state[p]["e_w"])  # get back to the original point

        self.base_optimizer.step()  # do the actual optimization step

        if zero_grad: self.zero_grad()

    def _grad_norm(self):
        """计算全局梯度范数"""
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
                    torch.stack([
                        p.grad.norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def step(self, closure=None):
        raise NotImplementedError("SAM requires two steps (first_step and second_step)")
