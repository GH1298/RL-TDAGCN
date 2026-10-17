"""
早停机制
监控验证集损失，连续 patience 个 epoch 不下降则停止训练
"""
import torch


class EarlyStopping:
    """
    早停机制
    
    Args:
        mode: 'min' (损失越小越好) 或 'max' (指标越大越好)
        min_delta: 最小改善量
        patience: 耐心值
        verbose: 是否打印信息
    """
    def __init__(self, mode='min', min_delta=1e-4, patience=20, verbose=True):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_epoch = 0
        self.should_stop = False
        self.best_model_state = None

    def __call__(self, current_score, epoch, model=None):
        if self.best_score is None:
            self.best_score = current_score
            self.best_epoch = epoch
            if model is not None:
                self.best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            return False
        
        if self.mode == 'min':
            improved = current_score < self.best_score - self.min_delta
        else:
            improved = current_score > self.best_score + self.min_delta
        
        if improved:
            self.best_score = current_score
            self.best_epoch = epoch
            self.counter = 0
            if model is not None:
                self.best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            if self.verbose:
                print(f"  [EarlyStop] 改善: {current_score:.6f} (epoch {epoch})")
        else:
            self.counter += 1
            if self.verbose:
                print(f"  [EarlyStop] 无改善 ({self.counter}/{self.patience}), 最佳: {self.best_score:.6f} @ epoch {self.best_epoch}")
            if self.counter >= self.patience:
                self.should_stop = True
                if self.verbose:
                    print(f"  [EarlyStop] 触发早停! 最佳 epoch: {self.best_epoch}")
        
        return self.should_stop