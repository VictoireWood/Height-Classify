# from pytorch_metric_learning import losses, miners
# from pytorch_metric_learning.distances import CosineSimilarity, DotProductSimilarity
import torch.nn as nn
import numpy as np

class CoLoss(nn.Module):
    def __init__(self, lambda1=0.3, lambda2=0.2) -> None:
        super().__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.single_loss = nn.SmoothL1Loss()
        self.coherence_loss = nn.MSELoss()
    
    def forward(self, pred_group, gt_group):
        pred_main = pred_group[:,0]
        pred_trans = pred_group[:,1]
        gt_main = gt_group[:,0]
        gt_trans = gt_group[:,1]
        
        # coherence_loss = self.coherence_loss(pred_main, pred_trans)

        main_loss = self.single_loss(pred_main, gt_main)
        trans_loss = self.single_loss(pred_trans, gt_trans)
        pred_trans_reverse = pred_trans/(gt_trans/gt_main)
        co_loss = self.coherence_loss(pred_trans_reverse, pred_main)
        loss = main_loss * (1 - self.lambda1 - self.lambda2) + trans_loss * self.lambda1 + co_loss * self.lambda2
        return loss

class huber(nn.Module):
    def __init__(self, delta=10):
        super().__init__()
        self.delta = delta
    def foward(self, pred, gt):    # gt是ground truth，pred是prediction
        loss = np.where(np.abs(gt-pred) < self.delta , 0.5*((gt-pred)**2), self.delta *np.abs(gt - pred) - 0.5*(self.delta**2))
        return np.sum(loss)

class logcosh(nn.Module):
    def __init__(self):
        super().__init__()
    def foward(self, pred, gt):
        loss = np.log(np.cosh(pred - gt))
        return np.sum(loss)
    
def get_loss(loss_name):
    if loss_name == 'MAE': return nn.L1Loss()
    elif loss_name == 'MSE': return nn.MSELoss()
    elif loss_name == 'Huber': return huber(delta=50)
    elif loss_name == 'LogCosh': return logcosh()
    raise NotImplementedError(f'Sorry, <{loss_name}> loss function is not implemented!')


# def get_loss(loss_name):
#     if loss_name == 'SupConLoss': return losses.SupConLoss(temperature=0.07)
#     if loss_name == 'CircleLoss': return losses.CircleLoss(m=0.4, gamma=80) #these are params for image retrieval
#     if loss_name == 'MultiSimilarityLoss': return losses.MultiSimilarityLoss(alpha=1.0, beta=50, base=0.0, distance=DotProductSimilarity())
#     if loss_name == 'ContrastiveLoss': return losses.ContrastiveLoss(pos_margin=0, neg_margin=1)
#     if loss_name == 'Lifted': return losses.GeneralizedLiftedStructureLoss(neg_margin=0, pos_margin=1, distance=DotProductSimilarity())
#     if loss_name == 'FastAPLoss': return losses.FastAPLoss(num_bins=30)
#     if loss_name == 'NTXentLoss': return losses.NTXentLoss(temperature=0.07) #The MoCo paper uses 0.07, while SimCLR uses 0.5.
#     if loss_name == 'TripletMarginLoss': return losses.TripletMarginLoss(margin=0.1, swap=False, smooth_loss=False, triplets_per_anchor='all') #or an int, for example 100
#     if loss_name == 'CentroidTripletLoss': return losses.CentroidTripletLoss(margin=0.05,
#                                                                             swap=False,
#                                                                             smooth_loss=False,
#                                                                             triplets_per_anchor="all",)
#     raise NotImplementedError(f'Sorry, <{loss_name}> loss function is not implemented!')

# def get_miner(miner_name, margin=0.1):
#     if miner_name == 'TripletMarginMiner' : return miners.TripletMarginMiner(margin=margin, type_of_triplets="semihard") # all, hard, semihard, easy
#     if miner_name == 'MultiSimilarityMiner' : return miners.MultiSimilarityMiner(epsilon=margin, distance=CosineSimilarity())
#     if miner_name == 'PairMarginMiner' : return miners.PairMarginMiner(pos_margin=0.7, neg_margin=0.3, distance=DotProductSimilarity())
#     return None
