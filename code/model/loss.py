from xml.parsers.expat import model
import torch
import torch.nn.functional as F

from model.utils import gradient_


def onsurface(pred_sdf, gt_sdf):
   sdf_constraint = torch.where(gt_sdf != -1, pred_sdf, torch.zeros_like(pred_sdf))
   return torch.abs(sdf_constraint).mean()

def offsurface(pred_sdf,gt_sdf):
   inter_constraint = torch.where(gt_sdf != -1, torch.zeros_like(pred_sdf), torch.exp(-1e2 * torch.abs(pred_sdf)))
   return inter_constraint.mean() 

def normal(gradient, gt_normals, gt_sdf):
   normal_constraint = torch.where(gt_sdf != -1, 1 - F.cosine_similarity(gradient, gt_normals, dim=-1)[..., None],
                                    torch.zeros_like(gradient[..., :1]))
   return normal_constraint.mean() 

def Eikonal(gradient):
    grad_constraint = torch.abs(gradient.norm(dim=-1) - 1)
    return grad_constraint.mean() 

def consign(pred_sdf1,pred_sdf2,gt_sdf):
   consign_constrint =torch.where(gt_sdf != -1, F.relu(pred_sdf1 * pred_sdf2), torch.zeros_like(pred_sdf1))
   max,indx = torch.max(consign_constrint,dim=0)
   p1 = pred_sdf1[indx]
   p2 = pred_sdf2[indx]
   return consign_constrint.mean()


def sdf(coords, model_output, gt):
   '''
      x: batch of input coordinates
      y: usually the output of the trial_soln function
      '''
   gt_sdf = gt['sdf']
   gt_normals = gt['normals']
   pred1 = model_output['base']
   pred2 = model_output['fine']

   #  coords = model_output['model_in']
   #  pred_sdf = model_output['model_out']
   pred_sdf = model_output['mix']

   gradient = gradient_(pred_sdf, coords)

   # Wherever boundary_values is not equal to zero, we interpret it as a boundary constraint.
   sdf_constraint = torch.where(gt_sdf != -1, pred_sdf, torch.zeros_like(pred_sdf))
   inter_constraint = torch.where(gt_sdf != -1, torch.zeros_like(pred_sdf), torch.exp(-1e2 * torch.abs(pred_sdf)))
   normal_constraint = torch.where(gt_sdf != -1, 1 - F.cosine_similarity(gradient, gt_normals, dim=-1)[..., None],
                                 torch.zeros_like(gradient[..., :1]))
   grad_constraint = torch.abs(gradient.norm(dim=-1) - 1)

   # sum_constraint = torch.abs(pred1+pred2).mean()
   consign_constraint = consign(pred1,pred2,gt_sdf) 
   # Exp      # Lapl
   # -----------------
   return {'sdf': torch.abs(sdf_constraint).mean() * 3e3,  # 1e4      # 3e3
         'inter': inter_constraint.mean() * 1e2,  # 1e2                   # 1e3
         'normal_constraint': normal_constraint.mean() * 1e2,  # 1e2
         'grad_constraint': grad_constraint.mean() * 5e1,
         'consign': consign_constraint * 1e2}  # 1e1      # 5e1
         #'sum': sum_constraint * 1e2 } 

def sdf_base(coords, pred_sdf, gt):
   '''
      x: batch of input coordinates
      y: usually the output of the trial_soln function
      '''
   gt_sdf = gt['sdf']
   gt_normals = gt['normals']

   #  coords = model_output['model_in']
   #  pred_sdf = model_output['model_out']

   gradient = gradient_(pred_sdf, coords)

   # Wherever boundary_values is not equal to zero, we interpret it as a boundary constraint.
   sdf_constraint = torch.where(gt_sdf != -1, pred_sdf, torch.zeros_like(pred_sdf))
   inter_constraint = torch.where(gt_sdf != -1, torch.zeros_like(pred_sdf), torch.exp(-1e2 * torch.abs(pred_sdf)))
   normal_constraint = torch.where(gt_sdf != -1, 1 - F.cosine_similarity(gradient, gt_normals, dim=-1)[..., None],
                                 torch.zeros_like(gradient[..., :1]))
   grad_constraint = torch.abs(gradient.norm(dim=-1) - 1)

   # Exp      # Lapl
   # -----------------
   return {'base_sdf': torch.abs(sdf_constraint).mean() * 3e3,  # 1e4      # 3e3
         'base_inter': inter_constraint.mean() * 1e2,  # 1e2                   # 1e3
         'base_normal_constraint': normal_constraint.mean() * 1e2,  # 1e2
         'base_grad_constraint': grad_constraint.mean() * 5e1}  # 1e1      # 5e1