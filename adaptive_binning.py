import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict
from collections import defaultdict

class AdaptiveHeightBinning:
    def __init__(self, 
                 original_bins: np.ndarray,
                 min_samples: int = 500,
                 max_bin_width: float = 200,
                 split_entropy_threshold: float = 0.7,
                 merge_entropy_threshold: float = 0.3,
                 low_confidence_threshold: float = 0.45,
                 high_confidence_threshold: float = 0.7,
                 min_bin_width: float = 20,
                 max_width_diff: float = 50):
        """
        初始化自适应分箱器
        
        Args:
            original_bins: 原始分箱边界数组
            min_samples: 每个新分箱的最小样本数
            max_bin_width: 最大分箱宽度
            split_entropy_threshold: 细分熵阈值
            merge_entropy_threshold: 合并熵阈值
            low_confidence_threshold: 低置信度阈值
            high_confidence_threshold: 高置信度阈值
            min_bin_width: 最小分箱宽度
            max_width_diff: 相邻分箱最大宽度差
        """
        self.original_bins = original_bins
        self.min_samples = min_samples
        self.max_bin_width = max_bin_width
        self.split_entropy_threshold = split_entropy_threshold
        self.merge_entropy_threshold = merge_entropy_threshold
        self.low_confidence_threshold = low_confidence_threshold
        self.high_confidence_threshold = high_confidence_threshold
        self.min_bin_width = min_bin_width
        self.max_width_diff = max_width_diff
        
        # 存储每个分箱的统计信息
        self.bin_stats = {}
        
    def compute_entropy(self, probs: np.ndarray) -> float:
        """计算概率分布的熵"""
        return -np.sum(probs * np.log(probs + 1e-8))
    
    def compute_bin_statistics(self, 
                             predictions: np.ndarray, 
                             true_labels: np.ndarray) -> Dict:
        """
        计算每个分箱的统计信息
        
        Args:
            predictions: 预测概率矩阵 (N_samples, N_bins)
            true_labels: 真实标签 (N_samples,)
        
        Returns:
            每个分箱的统计信息字典
        """
        bin_stats = defaultdict(dict)
        
        for i in range(len(self.original_bins) - 1):
            # 获取当前分箱的样本索引
            mask = (true_labels >= self.original_bins[i]) & (true_labels < self.original_bins[i+1])
            bin_predictions = predictions[mask]
            
            if len(bin_predictions) == 0:
                continue
                
            # 计算平均熵
            entropies = np.array([self.compute_entropy(p) for p in bin_predictions])
            avg_entropy = np.mean(entropies)
            
            # 计算平均最大置信度
            max_confidences = np.max(bin_predictions, axis=1)
            avg_confidence = np.mean(max_confidences)
            
            # 计算相邻混淆率
            pred_labels = np.argmax(bin_predictions, axis=1)
            confusion_rate = np.mean((pred_labels == i-1) | (pred_labels == i+1))
            
            bin_stats[i] = {
                'avg_entropy': avg_entropy,
                'avg_confidence': avg_confidence,
                'confusion_rate': confusion_rate,
                'sample_count': len(bin_predictions)
            }
            
        return dict(bin_stats)
    
    def should_split_bin(self, stats: Dict) -> bool:
        """判断是否应该细分当前分箱"""
        return (stats['avg_entropy'] >= self.split_entropy_threshold or 
                stats['avg_confidence'] <= self.low_confidence_threshold) and \
               stats['sample_count'] >= self.min_samples
    
    def should_merge_bins(self, stats1: Dict, stats2: Dict) -> bool:
        """判断是否应该合并两个相邻分箱"""
        return (stats1['avg_entropy'] <= self.merge_entropy_threshold and 
                stats2['avg_entropy'] <= self.merge_entropy_threshold and
                stats1['avg_confidence'] >= self.high_confidence_threshold and
                stats2['avg_confidence'] >= self.high_confidence_threshold)
    
    def smooth_boundaries(self, boundaries: np.ndarray) -> np.ndarray:
        """平滑分箱边界"""
        smoothed = boundaries.copy()
        for i in range(1, len(smoothed)-1):
            width = smoothed[i+1] - smoothed[i]
            prev_width = smoothed[i] - smoothed[i-1]
            
            if abs(width - prev_width) > self.max_width_diff:
                # 调整边界以使相邻分箱宽度差异不超过阈值
                target_width = (width + prev_width) / 2
                smoothed[i] = smoothed[i-1] + target_width
                
        return smoothed
    
    def generate_adaptive_bins(self, 
                             predictions: np.ndarray, 
                             true_labels: np.ndarray) -> np.ndarray:
        """
        生成自适应分箱边界
        
        Args:
            predictions: 预测概率矩阵
            true_labels: 真实标签
        
        Returns:
            新的分箱边界数组
        """
        # 计算原始分箱的统计信息
        self.bin_stats = self.compute_bin_statistics(predictions, true_labels)
        
        # 初始化新分箱边界
        new_bins = [self.original_bins[0]]
        current_pos = self.original_bins[0]
        
        i = 0
        while i < len(self.original_bins) - 1:
            current_stats = self.bin_stats.get(i, None)
            if current_stats is None:
                i += 1
                continue
                
            # 检查是否需要细分
            if self.should_split_bin(current_stats):
                # 细分当前分箱
                bin_width = self.original_bins[i+1] - self.original_bins[i]
                n_splits = min(3, max(2, int(bin_width / self.min_bin_width)))
                
                split_width = bin_width / n_splits
                for j in range(1, n_splits):
                    new_boundary = current_pos + j * split_width
                    new_bins.append(new_boundary)
                    
            # 检查是否需要与下一个分箱合并
            elif i < len(self.original_bins) - 2:
                next_stats = self.bin_stats.get(i+1, None)
                if next_stats and self.should_merge_bins(current_stats, next_stats):
                    # 跳过下一个分箱，实现合并
                    i += 1
            
            # 添加当前分箱的右边界
            new_bins.append(self.original_bins[i+1])
            current_pos = self.original_bins[i+1]
            i += 1
            
        # 确保边界严格递增
        new_bins = sorted(list(set(new_bins)))
        
        # 应用边界平滑
        new_bins = self.smooth_boundaries(np.array(new_bins))
        
        # 确保覆盖完整范围
        new_bins[0] = self.original_bins[0]
        new_bins[-1] = self.original_bins[-1]
        
        return new_bins

    def update_labels(self, labels: np.ndarray) -> np.ndarray:
        """
        将原始标签映射到新的分箱
        
        Args:
            labels: 原始高度标签
        
        Returns:
            新分箱的标签
        """
        new_labels = np.zeros_like(labels)
        for i in range(len(self.new_bins)-1):
            mask = (labels >= self.new_bins[i]) & (labels < self.new_bins[i+1])
            new_labels[mask] = i
        return new_labels
    
    def get_bin_centers(self) -> np.ndarray:
        """获取新分箱的中心值"""
        return (self.new_bins[:-1] + self.new_bins[1:]) / 2

# 使用示例
if __name__ == "__main__":
    # 示例数据
    original_bins = np.arange(0, 1000, 50)  # 50m间隔的原始分箱
    n_samples = 10000
    n_bins = len(original_bins) - 1
    
    # 生成模拟预测概率和真实标签
    predictions = np.random.dirichlet(np.ones(n_bins), size=n_samples)
    true_labels = np.random.uniform(0, 1000, size=n_samples)
    
    # 创建自适应分箱器
    binning = AdaptiveHeightBinning(
        original_bins=original_bins,
        min_samples=500,
        max_bin_width=200,
        split_entropy_threshold=0.7,
        merge_entropy_threshold=0.3
    )
    
    # 生成新的分箱边界
    new_bins = binning.generate_adaptive_bins(predictions, true_labels)
    
    print("原始分箱数量:", len(original_bins)-1)
    print("新分箱数量:", len(new_bins)-1)
    print("新分箱边界:", new_bins)
