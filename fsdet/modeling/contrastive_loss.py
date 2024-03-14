import torch
import torch.nn as nn
import torch.nn.functional as F
import fvcore.nn.weight_init as weight_init

from .utils import soft_max, soft_min, min_sets


class ContrastiveHead(nn.Module):
    """MLP head for contrastive representation learning, https://arxiv.org/abs/2003.04297
    Args:
        dim_in (int): dimension of the feature intended to be contrastively learned
        feat_dim (int): dim of the feature to calculated contrastive loss

    Return:
        feat_normalized (tensor): L-2 normalized encoded feature,
            so the cross-feature dot-product is cosine similarity (https://arxiv.org/abs/2004.11362)
    """
    def __init__(self, dim_in, feat_dim):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(dim_in, dim_in),
            nn.ReLU(inplace=True),
            nn.Linear(dim_in, feat_dim),
        )
        for layer in self.head:
            if isinstance(layer, nn.Linear):
                weight_init.c2_xavier_fill(layer)

    def forward(self, x):
        feat = self.head(x)
        feat_normalized = F.normalize(feat, dim=1)
        return feat_normalized


class SupConLoss(nn.Module):
    """Supervised Contrastive LOSS as defined in https://arxiv.org/pdf/2004.11362.pdf."""

    def __init__(self, temperature=0.2, iou_threshold=0.5, reweight_func='none'):
        '''Args:
            tempearture: a constant to be divided by consine similarity to enlarge the magnitude
            iou_threshold: consider proposals with higher credibility to increase consistency.
        '''
        super().__init__()
        self.temperature = temperature
        self.iou_threshold = iou_threshold
        self.reweight_func = reweight_func

    def forward(self, features, labels, ious):
        """
        Args:
            features (tensor): shape of [M, K] where M is the number of features to be compared,
                and K is the feature_dim.   e.g., [8192, 128]
            labels (tensor): shape of [M].  e.g., [8192]
        """
        assert features.shape[0] == labels.shape[0] == ious.shape[0]

        if len(labels.shape) == 1:
            labels = labels.reshape(-1, 1)

        # mask of shape [None, None], mask_{i, j}=1 if sample i and sample j have the same label
        label_mask = torch.eq(labels, labels.T).float().cuda()

        similarity = torch.div(
            torch.matmul(features, features.T), self.temperature)
        # for numerical stability
        sim_row_max, _ = torch.max(similarity, dim=1, keepdim=True)
        similarity = similarity - sim_row_max.detach()

        # mask out self-contrastive
        logits_mask = torch.ones_like(similarity)
        logits_mask.fill_diagonal_(0)

        # sim_kernel - 256 x 256
        exp_sim = torch.exp(similarity) * logits_mask
        log_prob = similarity - torch.log(exp_sim.sum(dim=1, keepdim=True))

        # per_label_loss - 1 x 256
        per_label_log_prob = (log_prob * logits_mask * label_mask).sum(1) / label_mask.sum(1)

        keep = ious >= self.iou_threshold
        per_label_log_prob = per_label_log_prob[keep]
        loss = -per_label_log_prob

        coef = self._get_reweight_func(self.reweight_func)(ious)
        coef = coef[keep]

        loss = loss * coef
        return loss.mean()

    @staticmethod
    def _get_reweight_func(option):
        def trivial(iou):
            return torch.ones_like(iou)
        def exp_decay(iou):
            return torch.exp(iou) - 1
        def linear(iou):
            return iou

        if option == 'none':
            return trivial
        elif option == 'linear':
            return linear
        elif option == 'exp':
            return exp_decay

class GraphCut(nn.Module):
    
    def __init__(self, temperature=0.2, 
                       iou_threshold=0.5, 
                       reweight_func='none',
                       lamda=1.0, is_cf=True):
        '''Args:
            tempearture: a constant to be divided by consine similarity to enlarge the magnitude
            iou_threshold: consider proposals with higher credibility to increase consistency.
        '''
        super().__init__()
        self.temperature = temperature
        self.iou_threshold = iou_threshold
        self.reweight_func = reweight_func
        self.lamda = lamda
        self.base_temperature = 0.07
        self.is_cf = is_cf

    def forward(self, features, labels, ious):
        """
        Args:
            features (tensor): shape of [M, K] where M is the number of features to be compared,
                and K is the feature_dim.   e.g., [8192, 128]
            labels (tensor): shape of [M].  e.g., [8192]
        """
        assert features.shape[0] == labels.shape[0] == ious.shape[0]

        if len(labels.shape) == 1:
            labels = labels.reshape(-1, 1)

        # mask of shape [None, None], mask_{i, j}=1 if sample i and sample j have the same label
        mask_pos = torch.eq(labels, labels.T).float().cuda()
        mask_neg = 1.0 - mask_pos

        similarity = torch.div(
            torch.matmul(features, features.T), self.temperature)
        # for numerical stability
        sim_row_max, _ = torch.max(similarity, dim=1, keepdim=True)
        similarity = similarity - sim_row_max.detach()

        # mask out self-contrastive
        logits_mask = torch.ones_like(similarity)
        logits_mask.fill_diagonal_(0)
        mask_pos.fill_diagonal_(0)
        mask_neg.fill_diagonal_(0)

        sim_kernel = torch.exp(similarity) * logits_mask
        # log_prob = self.lamda * (exp_sim * mask_pos).sum(1, keepdim=True) - (exp_sim * mask_neg).sum(1, keepdim=True)

        if self.is_cf:
            log_prob = torch.div(
                -self.lamda * (sim_kernel * mask_neg).sum(1),
                mask.sum(1)
            )
        else:
            # Min the similarity between negative set
            log_prob = torch.log(
                (self.lamda * (sim_kernel * mask)).sum(1) / (sim_kernel * mask_neg).sum(1)
            )
        
        per_label_log_prob = log_prob
        keep = ious >= self.iou_threshold
        per_label_log_prob = per_label_log_prob[keep]
        loss = -per_label_log_prob

        coef = self._get_reweight_func(self.reweight_func)(ious)
        coef = coef[keep]

        loss = loss * coef
        return loss.mean()

    @staticmethod
    def _get_reweight_func(option):
        def trivial(iou):
            return torch.ones_like(iou)
        def exp_decay(iou):
            return torch.exp(iou) - 1
        def linear(iou):
            return iou

        if option == 'none':
            return trivial
        elif option == 'linear':
            return linear
        elif option == 'exp':
            return exp_decay

class FacilityLocation(nn.Module):
    
    def __init__(self, temperature=0.2, 
                       iou_threshold=0.5, 
                       reweight_func='none',
                       lamda=1.0):
        '''Args:
            tempearture: a constant to be divided by consine similarity to enlarge the magnitude
            iou_threshold: consider proposals with higher credibility to increase consistency.
        '''
        super().__init__()
        self.temperature = temperature
        self.iou_threshold = iou_threshold
        self.reweight_func = reweight_func
        self.lamda = lamda
        self.base_temperature = 0.07

    def forward(self, features, labels, ious):
        """
        Args:
            features (tensor): shape of [M, K] where M is the number of features to be compared,
                and K is the feature_dim.   e.g., [8192, 128]
            labels (tensor): shape of [M].  e.g., [8192]
        """
        assert features.shape[0] == labels.shape[0] == ious.shape[0]

        if len(labels.shape) == 1:
            labels = labels.reshape(-1, 1)

        # mask of shape [None, None], mask_{i, j}=1 if sample i and sample j have the same label
        mask_pos = torch.eq(labels, labels.T).float().cuda()
        mask_neg = 1.0 - mask_pos

        similarity = torch.div(
            torch.matmul(features, features.T), self.temperature)
        # for numerical stability
        sim_row_max, _ = torch.max(similarity, dim=1, keepdim=True)
        similarity = similarity - sim_row_max.detach()

        # mask out self-contrastive
        logits_mask = torch.ones_like(similarity)
        logits_mask.fill_diagonal_(0)
        mask_pos.fill_diagonal_(0)
        mask_neg.fill_diagonal_(0)

        exp_sim = torch.exp(similarity) * logits_mask
        log_prob = torch.log(exp_sim.sum(1, keepdim=True))
        log_prob = torch.log(
            (exp_sim * mask_neg).sum(1)
        )
        
        per_label_log_prob = log_prob

        keep = ious >= self.iou_threshold
        per_label_log_prob = per_label_log_prob[keep]
        loss = per_label_log_prob

        coef = self._get_reweight_func(self.reweight_func)(ious)
        coef = coef[keep]

        loss = loss * coef
        return loss.mean()

    @staticmethod
    def _get_reweight_func(option):
        def trivial(iou):
            return torch.ones_like(iou)
        def exp_decay(iou):
            return torch.exp(iou) - 1
        def linear(iou):
            return iou

        if option == 'none':
            return trivial
        elif option == 'linear':
            return linear
        elif option == 'exp':
            return exp_decay

class LogDet(nn.Module):
    
    def __init__(self, temperature=0.2, 
                       iou_threshold=0.5, 
                       reweight_func='none',
                       lamda=1.0):
        '''Args:
            tempearture: a constant to be divided by consine similarity to enlarge the magnitude
            iou_threshold: consider proposals with higher credibility to increase consistency.
        '''
        super().__init__()
        self.temperature = temperature
        self.iou_threshold = iou_threshold
        self.reweight_func = reweight_func
        self.lamda = lamda
        self.base_temperature = 0.07

    def forward(self, features, labels, ious):
        """
        Args:
            features (tensor): shape of [M, K] where M is the number of features to be compared,
                and K is the feature_dim.   e.g., [8192, 128]
            labels (tensor): shape of [M].  e.g., [8192]
        """
        assert features.shape[0] == labels.shape[0] == ious.shape[0]

        keep = ious >= self.iou_threshold
        coef = self._get_reweight_func(self.reweight_func)(ious)
        coef = coef[keep]

        features = features[keep]
        labels = labels[keep]

        assert features.shape[0] == labels.shape[0]

        if len(labels.shape) == 1:
            labels = labels.reshape(-1, 1)
        
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))
        
        # mask of shape [None, None], mask_{i, j}=1 if sample i and sample j have the same label
        mask_pos = torch.eq(labels, labels.T).float().cuda()
        mask_neg = 1.0 - mask_pos
        mask_ground = torch.ones_like(mask_pos) 

        
        similarity = torch.matmul(features, features.T)
        similarity = similarity * coef

        # get unique vectors 
        labelSet = torch.unique(labels, sorted=True)

        log_prob = 0.0
        ground_set_det = torch.logdet((similarity * mask_ground) + (0.5 * torch.eye(mask_ground.shape[0]).to(device)))

        for label_index in labelSet:
            mask_set = torch.where(labels == label_index, 1.0, 0.0)
            indices = torch.nonzero(mask_set.squeeze(1))
            S_label = torch.index_select(similarity, 0, indices.squeeze(1))
            S_label = torch.index_select(S_label, 1, indices.squeeze(1))

            log_prob +=  torch.logdet(S_label + (0.5 * torch.eye(S_label.shape[0]).to(device))) 
        log_prob -= ground_set_det

        loss = log_prob

        return loss.mean()

    @staticmethod
    def _get_reweight_func(option):
        def trivial(iou):
            return torch.ones_like(iou)
        def exp_decay(iou):
            return torch.exp(iou) - 1
        def linear(iou):
            return iou

        if option == 'none':
            return trivial
        elif option == 'linear':
            return linear
        elif option == 'exp':
            return exp_decay

class FLVMI(nn.Module):
    
    def __init__(self, temperature=0.2, 
                       iou_threshold=0.5, 
                       reweight_func='none',
                       lamda=1.0):
        '''Args:
            tempearture: a constant to be divided by consine similarity to enlarge the magnitude
            iou_threshold: consider proposals with higher credibility to increase consistency.
        '''
        super().__init__()
        self.temperature = temperature
        self.iou_threshold = iou_threshold
        self.reweight_func = reweight_func
        self.lamda = lamda
        self.base_temperature = 0.07

    def forward(self, features, labels, ious):
        """
        Args:
            features (tensor): shape of [M, K] where M is the number of features to be compared,
                and K is the feature_dim.   e.g., [8192, 128]
            labels (tensor): shape of [M].  e.g., [8192]
        """
        assert features.shape[0] == labels.shape[0] == ious.shape[0]

        if len(labels.shape) == 1:
            labels = labels.reshape(-1, 1)

        # mask of shape [None, None], mask_{i, j}=1 if sample i and sample j have the same label
        mask_pos = torch.eq(labels, labels.T).float().cuda()
        mask_neg = 1.0 - mask_pos

        similarity = torch.div(
            torch.matmul(features, features.T), self.temperature)
        # for numerical stability
        sim_row_max, _ = torch.max(similarity, dim=1, keepdim=True)
        similarity = similarity - sim_row_max.detach()

        # mask out self-contrastive
        logits_mask = torch.ones_like(similarity)
        logits_mask.fill_diagonal_(0)
        mask_pos.fill_diagonal_(0)
        mask_neg.fill_diagonal_(0)

        log_prob = min_sets(self.lamda * soft_max(mask_neg * similarity, axis=0),
                                     soft_max(mask_pos * similarity, axis=0))
        
        per_label_log_prob = log_prob

        keep = ious >= self.iou_threshold
        per_label_log_prob = per_label_log_prob[keep]
        loss = per_label_log_prob

        coef = self._get_reweight_func(self.reweight_func)(ious)
        coef = coef[keep]

        loss = loss * coef
        return loss.mean()

    @staticmethod
    def _get_reweight_func(option):
        def trivial(iou):
            return torch.ones_like(iou)
        def exp_decay(iou):
            return torch.exp(iou) - 1
        def linear(iou):
            return iou

        if option == 'none':
            return trivial
        elif option == 'linear':
            return linear
        elif option == 'exp':
            return exp_decay

class FLQMI(nn.Module):
    
    def __init__(self, temperature=0.2, 
                       iou_threshold=0.5, 
                       reweight_func='none',
                       lamda=1.0, 
                       n_classes = 20, n_novel = 5):
        '''Args:
            tempearture: a constant to be divided by consine similarity to enlarge the magnitude
            iou_threshold: consider proposals with higher credibility to increase consistency.
        '''
        super().__init__()
        self.temperature = temperature
        self.iou_threshold = iou_threshold
        self.reweight_func = reweight_func
        self.lamda = lamda
        self.base_temperature = 0.07

        self.n_classes = n_classes
        self.n_novel = n_novel

    def forward(self, features, labels, ious):
        """
        Args:
            features (tensor): shape of [M, K] where M is the number of features to be compared,
                and K is the feature_dim.   e.g., [8192, 128]
            labels (tensor): shape of [M].  e.g., [8192]
        """
        assert features.shape[0] == labels.shape[0] == ious.shape[0]

        if len(labels.shape) == 1:
            labels = labels.reshape(-1, 1)

        # mask of shape [None, None], mask_{i, j}=1 if sample i and sample j have the same label
        mask_pos = torch.eq(labels, labels.T).float().cuda()
        mask_neg = 1.0 - mask_pos

        # find novel classes
        keep_novel = labels >= (self.n_classes - self.n_novel)
        features_novel = features[keep_novel.squeeze(1)]
        
        # Calc sim kernel -> All x novel
        similarity = torch.div(
            torch.matmul(features, features_novel.T), self.temperature)

        log_prob = self.lamda * soft_max(similarity, axis=0).sum(0) + \
                                soft_max(similarity, axis=1).sum(0)
        # log_prob = self.lamda * torch.max(similarity, dim=0)[0].sum(0) + \
        #                         torch.max(similarity, dim=1)[0].sum(0)
        
        # Normalize with the size of the whole set
        log_prob = (1 / features.shape[0]) * log_prob
        loss = log_prob

        return loss.mean()

    @staticmethod
    def _get_reweight_func(option):
        def trivial(iou):
            return torch.ones_like(iou)
        def exp_decay(iou):
            return torch.exp(iou) - 1
        def linear(iou):
            return iou

        if option == 'none':
            return trivial
        elif option == 'linear':
            return linear
        elif option == 'exp':
            return exp_decay


class GCMI(nn.Module):
    
    def __init__(self, temperature=0.2, 
                       iou_threshold=0.5, 
                       reweight_func='none',
                       lamda=1.0):
        '''Args:
            tempearture: a constant to be divided by consine similarity to enlarge the magnitude
            iou_threshold: consider proposals with higher credibility to increase consistency.
        '''
        super().__init__()
        self.temperature = temperature
        self.iou_threshold = iou_threshold
        self.reweight_func = reweight_func
        self.lamda = 0.5
        self.base_temperature = 0.07

    def forward(self, features, labels, ious):
        """
        Args:
            features (tensor): shape of [M, K] where M is the number of features to be compared,
                and K is the feature_dim.   e.g., [8192, 128]
            labels (tensor): shape of [M].  e.g., [8192]
        """
        assert features.shape[0] == labels.shape[0] == ious.shape[0]

        if len(labels.shape) == 1:
            labels = labels.reshape(-1, 1)

        # mask of shape [None, None], mask_{i, j}=1 if sample i and sample j have the same label
        mask_pos = torch.eq(labels, labels.T).float().cuda()
        mask_neg = 1.0 - mask_pos

        similarity = torch.div(
            torch.matmul(features, features.T), self.temperature)
        # for numerical stability
        sim_row_max, _ = torch.max(similarity, dim=1, keepdim=True)
        similarity = similarity - sim_row_max.detach()

        # mask out self-contrastive
        logits_mask = torch.ones_like(similarity)
        logits_mask.fill_diagonal_(0)
        mask_pos.fill_diagonal_(0)
        mask_neg.fill_diagonal_(0)

        log_prob = min_sets(self.lamda * soft_max(mask_neg * similarity, axis=0),
                                     soft_max(mask_pos * similarity, axis=0))
        
        log_prob = 2.0 * self.lamda * (similarity * mask_neg).sum(1, keepdim=True)

        keep = ious >= self.iou_threshold
        per_label_log_prob = log_prob[keep]
        loss = -per_label_log_prob

        coef = self._get_reweight_func(self.reweight_func)(ious)
        coef = coef[keep]

        loss = loss * coef
        return loss.mean()

    @staticmethod
    def _get_reweight_func(option):
        def trivial(iou):
            return torch.ones_like(iou)
        def exp_decay(iou):
            return torch.exp(iou) - 1
        def linear(iou):
            return iou

        if option == 'none':
            return trivial
        elif option == 'linear':
            return linear
        elif option == 'exp':
            return exp_decay

class JointContrastiveLoss(nn.Module):
    
    def __init__(self, sim_func="FL",
                       smi_func="FLVMI",
                       temperature=0.2, 
                       iou_threshold=0.5, 
                       reweight_func='none',
                       lamda=1.0):
        '''Args:
            tempearture: a constant to be divided by consine similarity to enlarge the magnitude
            iou_threshold: consider proposals with higher credibility to increase consistency.
        '''
        super().__init__()
        self.temperature = temperature
        self.iou_threshold = iou_threshold
        self.reweight_func = reweight_func
        self.lamda = 0.5
        self.base_temperature = 0.07

        self.eta = 0.5

        # Choice of Information function
        if sim_func == "FL":
            self.sim = FacilityLocation(self.temperature, self.iou_threshold, self.reweight_func)
        elif sim_func == "GC":
            self.sim = GraphCut(self.temperature, self.iou_threshold, self.reweight_func)
        elif sim_func == "LogDet":
            self.sim = FacilityLocation(self.temperature, self.iou_threshold, self.reweight_func)

        # Choice of Mutual Information function
        if smi_func == "FLVMI":
            self.smi = FLVMI(self.temperature, self.iou_threshold, self.reweight_func)
        elif smi_func == "FLQMI":
            self.smi = FLQMI(self.temperature, self.iou_threshold, self.reweight_func)
        elif smi_func == "GCMI":
            self.smi = GCMI(self.temperature, self.iou_threshold, self.reweight_func)


    def forward(self, features, labels, ious):
        """
        Args:
            features (tensor): shape of [M, K] where M is the number of features to be compared,
                and K is the feature_dim.   e.g., [8192, 128]
            labels (tensor): shape of [M].  e.g., [8192]
        """

        sim_loss = self.sim(features, labels, ious)

        smi_loss = self.smi(features, labels, ious)

        loss = (1 - self.eta) * sim_loss + self.eta * smi_loss

        return loss

    @staticmethod
    def _get_reweight_func(option):
        def trivial(iou):
            return torch.ones_like(iou)
        def exp_decay(iou):
            return torch.exp(iou) - 1
        def linear(iou):
            return iou

        if option == 'none':
            return trivial
        elif option == 'linear':
            return linear
        elif option == 'exp':
            return exp_decay