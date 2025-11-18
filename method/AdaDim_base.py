# Copyright (c) Winci.
# Licensed under the Apache License, Version 2.0 (the "License");

import torch
import copy
from itertools import chain
from method.base import BaseMethod
import torch.nn.functional as F
import torch.distributed as dist
import wandb
class AdaDim_Base(BaseMethod):
    # ReSA using the momentum network, better performance

    def __init__(self, args):
        super().__init__(args)

        #self.momentum_encoder = copy.deepcopy(self.encoder)
        #self.momentum_projector = copy.deepcopy(self.projector)
        '''
        for param in chain(self.momentum_encoder.parameters(), 
                           self.momentum_projector.parameters()):
            param.requires_grad = False
        '''
        self.temp = args.temperature
        self.alpha = args.alpha
        self.gamma = args.gamma
        self.args = args
        self.rank_list = []
    # https://github.com/facebookresearch/VICRegL/blob/main/main_vicregl.py
    # 
    def ForwardWrapper(self, samples, encoder, projector):

        # do not concate different views if BN is in the model 
        # As it will disrupt the zero-mean, unit-variance distribution
        h = [encoder(x) for x in samples]
        z = [projector(x) for x in h]
        emb = [F.normalize(projector(x)) for x in h]

        if not torch.is_grad_enabled():
            emb = [concat_all_gather(x) for x in emb]
            z = [concat_all_gather(x) for x in z]
            h = [concat_all_gather(x) for x in h]

        return h, emb, z
    def vicreg_loss(self,z1, z2, sim_coeff=25.0, var_coeff=25.0, cov_coeff=1.0, eps=1e-4):
        # Invariance Loss
        inv_loss = F.mse_loss(z1, z2)

        # Std Loss
        eps = 1e-4
        std_z1 = torch.sqrt(z1.var(dim=0) + eps)
        std_z2 = torch.sqrt(z2.var(dim=0) + eps)
        std_loss = torch.mean(F.relu(1 - std_z1)) + torch.mean(F.relu(1 - std_z2))

        # Covariance Loss
        N, D = z1.size()

        z1_c = z1 - z1.mean(dim=0)
        z2_c = z2 - z2.mean(dim=0)
        cov_z1 = (z1_c.T @ z1_c) / (N - 1)
        cov_z2 = (z2_c.T @ z2_c) / (N - 1)

        diag = torch.eye(D, device=z1.device)
        cov_loss = cov_z1[~diag.bool()].pow_(2).sum() / D + cov_z2[~diag.bool()].pow_(2).sum() / D
        loss = sim_coeff * inv_loss + var_coeff * std_loss + cov_coeff * cov_loss
        return loss
    
    def get_negative_mask(self,batch_size):
        negative_mask = torch.ones((batch_size, 2 * batch_size), dtype=bool)
        for i in range(batch_size):
            negative_mask[i, i] = 0
            negative_mask[i, i + batch_size] = 0

        negative_mask = torch.cat((negative_mask, negative_mask), 0)
        return negative_mask


    def simclr_loss(self, z1, z2, batch_size=256, temp=0.1):
        out = torch.cat([z1, z2], dim=0)
        neg = torch.exp(torch.mm(out, out.t().contiguous()) / temp)
        old_neg = neg.clone()
        mask = self.get_negative_mask(batch_size).to('cuda')
        neg = neg.masked_select(mask).view(2 * batch_size, -1)

        # pos score
        pos = torch.exp(torch.sum(z1 * z2, dim=-1) / temp)
        pos = torch.cat([pos, pos], dim=0)
        
        # negative samples similarity scoring
        
        Ng = neg.sum(dim=-1)
    
            
        # contrastive loss
        loss = (- torch.log(pos / (pos + Ng) )).mean()

        return loss

    def renyi_entropy(self, matrix, alpha=2):
        assert matrix.shape[0] == matrix.shape[1]
        n = matrix.shape[0]
        
        # Calculate the Renyi entropy
        if alpha == 1:
            entropy = -torch.trace((matrix/n) @ self.mat_log(matrix/n))
        else:
            entropy = 1 / (1-alpha) * torch.log(torch.trace(self.mat_pow(matrix/n, alpha)))
        return entropy
    def mat_pow(self,matrix, alpha):
        n = matrix.shape[0]
        res = torch.eye(n).cuda()
        for i in range(alpha):
            res = res @ matrix
        return res

    def mat_log(self,matrix):
        matrix = matrix.to(torch.float32)
        n = matrix.shape[0]
        U, S, V = torch.linalg.svd(matrix, full_matrices=False)
        T = torch.zeros(n).cuda()
        for i in range(n):
            T[i] = torch.log(S[i])
        res = U @ torch.diag(T) @ V
        return res
    def mutual_info_loss(self, z1, z2, alpha=2):
        Z1 = F.normalize(z1, dim=-1)
        Z2 = F.normalize(z2, dim=-1)
        matrix1 = Z1 @ Z1.T
        matrix2 = Z2 @ Z2.T
        
        # Calculate individual entropies
        h1 = self.renyi_entropy(matrix1, alpha)
        h2 = self.renyi_entropy(matrix2, alpha)
        
        # Calculate the Hadamard product of the two matrices
        hadamard_product = torch.mul(matrix1, matrix2)
        
        # Calculate the joint Renyi entropy
        joint_entropy = self.renyi_entropy(hadamard_product, alpha)

        # Calculate mutual information
        mutual_info = h1 + h2 - joint_entropy
        return mutual_info
    def get_rank(self, z1, z2):
        z = torch.cat((z1,z2))

        svd = torch.linalg.svdvals(z)

        rankme_score = 0
        svd_norm = torch.norm(svd, p=1, dim=-1)
        eppsilon = 1e-7
        for i in range(0, len(svd)):
            p_k = svd[i] / svd_norm + eppsilon
            rankme_score = rankme_score + p_k * torch.log(p_k)
        return torch.exp(-1 * rankme_score)
    def forward(self, samples, it, epoch):

        samples = [x.cuda(non_blocking=True) for x in samples]

        h, emb, z = self.ForwardWrapper(samples, self.encoder, self.projector)
        
        #with torch.no_grad():
            #self.update_momentum_params(self.momentum)
            #h_m, emb_m, z_m = self.ForwardWrapper(samples[:2], self.momentum_encoder, self.momentum_projector)
            #assign = self.sinkhorn_knopp(h @ h_m.T)

        
        total_loss = 0
        n_loss_terms = 0
        
        if(epoch % self.args.step_epoch == 0 and epoch != 0):
            if (it >= 0 and it <= 10):

                self.rank_list.append(self.get_rank(z[0],z[1]))
            elif(it == 11):
                mean_rank = torch.mean(torch.stack(self.rank_list))
                self.rank_list = []
                self.alpha = (mean_rank/(self.args.batch_size*2)).detach().cpu().numpy()
                self.alpha = torch.tensor(self.alpha).float()
                wandb.log({"alpha": self.alpha})
                self.rank = mean_rank
        else:
            self.alpha = self.alpha


        beta = self.gamma * self.alpha
        # Compute SimCLR loss
        
        
        for q in range(len(emb)):
            for v in range(len(emb)):
                if v == q:
                    continue
                emb_sim = (self.alpha) * self.simclr_loss(emb[q],emb[v],self.args.batch_size,self.temp)
                total_loss += emb_sim
                n_loss_terms += 1
        # Compute VICReg loss
        for q in range(len(z)):
            for v in range(len(z)):
                if v == q:
                    continue
                emb_sim = (1-self.alpha) * self.vicreg_loss(z[q],z[v])
                total_loss += emb_sim
                n_loss_terms += 1
        
        ssl_loss = total_loss / n_loss_terms
        
        total_matrix_loss = 0
        total_matrix_terms = 0

        # Compute other losses
        for q in range(len(h)):
            for v in range(len(z)):
                if v == q:
                    continue
                emb_sim = self.mutual_info_loss(h[q],z[v])
                total_matrix_loss += emb_sim
                total_matrix_terms += 1

        matrix_loss = total_matrix_loss / total_matrix_terms
        AdaDim_loss = (1-beta) * ssl_loss - beta * matrix_loss
        

        return AdaDim_loss


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=0)
    return output
