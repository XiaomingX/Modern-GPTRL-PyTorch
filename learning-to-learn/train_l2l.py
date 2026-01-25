"""
业务问题：元训练一个能够快速优化简单二次函数的 LSTM 优化器。
实现逻辑：
1. 定义元训练环境。
2. 运行多个 Epochs，不断演化元优化器的参数。
3. 观察元损失的下降。
"""

from l2l_optimizer import MetaOptimizer, QuadraticProblem

def train_l2l():
    print("Start Meta-Training Learning-to-Learn Optimizer...")
    
    # 定义待优化问题的生成工厂
    def problem_factory():
        return QuadraticProblem()
        
    meta_opt = MetaOptimizer(problem_factory)
    
    # 开始元训练
    meta_opt.meta_train(num_epochs=100, unroll_len=20)
    
    print("\nMeta-Training Completed!")

if __name__ == "__main__":
    train_l2l()
