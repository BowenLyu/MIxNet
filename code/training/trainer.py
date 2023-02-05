from multiprocessing import current_process
import os
import torch
import utils.general as utils
import numpy as np
from datetime import datetime
import GPUtil
import omegaconf
from tqdm import tqdm
import time
from torch.utils.tensorboard import SummaryWriter

from dataset.Recondataset import ReconDataset
from model.Mixnet import Mixnet
from model.loss import sdf, sdf_base,test_loss
from utils.plot import plot_surface
from model.MLP import MLPNet
from training.lr_adjust import lr_adjust


class Basetrainner():
    def __init__(self, args):
        #load
        # with open(args.conf_path) as f:
        #     conf = yaml.safe_load(f)
        conf = omegaconf.OmegaConf.load(args.conf_path)

        #arguments
        self.io = conf.dataio
        self.train = conf['train']
        self.base = conf.MLP
        self.fine = conf.Siren
        self.inter = conf.PNet
        self.plot = conf.plot

        # # assign GPU
        # if args.gpu == "auto":
        #     deviceIDs = GPUtil.getAvailable(order='memory', limit=2, maxLoad=0.5, maxMemory=0.5, includeNan=False, excludeID=[],
        #                             excludeUUID=[])
        #     gpu = deviceIDs[0]
        # else:
        #     gpu = args.gpu
        # self.GPU_INDEX = gpu
        # os.environ["CUDA_VISIBLE_DEVICES"] = '{0}'.format(self.GPU_INDEX)
        

        #create export folder
        utils.mkdir_ifnotexists(os.path.join('../',self.io.exps_folder_name))
        self.expdir = os.path.join('../', self.io.exps_folder_name, self.io.expname)
        utils.mkdir_ifnotexists(self.expdir)

        #create timestamp
        self.timestamp = '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.now())
        utils.mkdir_ifnotexists(os.path.join(self.expdir, self.timestamp))

        # #debug folder
        # log_dir = os.path.join(self.expdir, self.timestamp, 'log')
        # self.log_dir = log_dir
        # utils.mkdir_ifnotexists(log_dir)
        # utils.configure_logging(True,False,os.path.join(self.log_dir,'log.txt'))

        #create plot folder
        self.plots_dir = os.path.join(self.expdir, self.timestamp, 'plots')
        utils.mkdir_ifnotexists(self.plots_dir)

        #create parameter folder
        self.model_params_subdir = "ModelParameters"
        self.optimizer_params_subdir = "OptimizerParameters"

        self.checkpoints_path = os.path.join(self.expdir, self.timestamp, 'checkpoints')
        self.model_params_path = os.path.join(self.checkpoints_path,self.model_params_subdir)
        self.optimizer_params_path = os.path.join(self.checkpoints_path, self.optimizer_params_subdir)
        utils.mkdir_ifnotexists(self.checkpoints_path)
        utils.mkdir_ifnotexists(self.model_params_path)
        utils.mkdir_ifnotexists(self.optimizer_params_path)

        #tensorboard
        self.summaries_dir = os.path.join(self.expdir, self.timestamp, 'summaries')
        self.writer = SummaryWriter(self.summaries_dir)

        #network
        self.ds= ReconDataset(data_path=self.io['data_path'],point_batch=self.train['points_batch'])
        self.dataloader = torch.utils.data.DataLoader(self.ds, batch_size=self.train['batch_size'], drop_last=True)
        
        self.network = Mixnet(self.base, self.fine, self.inter).cuda()
        # for name, param in self.network.named_parameters():
        #     if param.requires_grad:
        #         print(name, param.data)

        self.loss = sdf
        self.loss_base = sdf_base

        #optimizer
        self.optimizer = torch.optim.Adam(
            [
                {
                    "params": self.network.base.parameters(),
                    "lr": self.train.lr_base
                   # "betas": (0.9, 0.999),
                   # "eps": 1e-08,
                   # "weight_decay": self.weight_decay
                },
                {
                    "params": self.network.fine.parameters(),
                    "lr": self.train.lr_fine
                },
                {
                    "params": self.network.inter.parameters(),
                    "lr": self.train.lr_inter
                }
            ])
        
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, self.get_lr())


        print("Initialization has finished!")

    def get_lr(self):
        lr_func = []
        netlist = ['base', 'fine', 'inter']
        for key in netlist:
            def lr_adjust_(epoch, progress_list=self.train.progress[key], lr_list=self.train.lr_factor[key]):
                current_progress = epoch / self.train['nepoch']
                l = len(progress_list)
                for i in range(l-1):
                    if current_progress >= progress_list[i] and current_progress <= progress_list[i+1]:
                        relative_progress = current_progress-progress_list[i]
                        interval = progress_list[i+1]-progress_list[i]
                        return lr_list[i] + (1 - np.cos(np.pi * relative_progress/interval))*(lr_list[i+1]-lr_list[i])/2                        
            lr_func.append(lr_adjust_)
        return lr_func

    def get_weight(self, epoch):
        current_progress = epoch / self.train['nepoch']
        weight_progress = self.train.progress['base']
        weight_value_list = self.train['weight']
        l = len(weight_progress)
        for i in range(l-1):
            if current_progress >= weight_progress[i] and current_progress <= weight_progress[i+1]:
                relative_progress = current_progress-weight_progress[i]
                interval = weight_progress[i+1]-weight_progress[i]
                return weight_value_list[i] + (1 - np.cos(np.pi * relative_progress/interval))*(weight_value_list[i+1]-weight_value_list[i])/2 
        return 0

    def run(self):
        print("Begin to train!")

        # Initialization
        total_steps = 0

        netclass = ['mix'] 
        #netclass = ['mlp'，'mix'] #for test

        best_loss = 1e5

        torch.cuda.empty_cache()

        weight_list = []
        

        for epoch in tqdm(range(self.train['nepoch'])):
            if (epoch + 1) % self.train.save_frequency == 0 and epoch > 0:
                print("save checkpoints!")
                torch.save(self.network.state_dict(),
                            os.path.join(self.model_params_path, 'model_epoch_%04d.pth' % epoch))
                
                with torch.no_grad():
                    self.network.eval()

                    for key in netclass:
                        plot_surface(decoder=self.network,
                                    path=self.plots_dir,
                                    epoch=epoch,
                                    shapename=self.io.expname,
                                    **self.plot,
                                    cls=key)
                    

                self.network.train()
            
            weight = self.get_weight(epoch)

            weight_list.append(weight)
            

            for step, (model_input, gt) in enumerate(self.dataloader):
                start_time = time.time()
            
                model_input = {key: value.cuda() for key, value in model_input.items()}
                gt = {key: value.cuda() for key, value in gt.items()}

                model_input = model_input['pnts'].requires_grad_()
                model_output = self.network(model_input)

                losses = self.loss(model_input, model_output, gt)

                train_loss = 0
                for loss_name, loss in losses.items():
                    single_loss = loss.mean()
                    self.writer.add_scalar(loss_name, single_loss, total_steps)
                    train_loss += single_loss

                # #============progressive===========
                if weight > 0:
                    losses_base = self.loss_base(model_input, model_output['base'], gt)

                    base_loss = 0
                    for loss_name, loss in losses_base.items():
                        single_loss = loss.mean()
                        self.writer.add_scalar(loss_name, single_loss, total_steps)
                        base_loss += single_loss

                    train_loss = (1 - weight) * train_loss  + weight * base_loss
                

                #=========record best loss=============
                if train_loss <  best_loss:
                    best_loss = train_loss.item()
                    torch.save(self.network.state_dict(),
                            os.path.join(self.model_params_path, 'model_best.pth'))
                #=======================================

                self.writer.add_scalar("total_train_loss", train_loss, total_steps)

                self.optimizer.zero_grad()
                train_loss.backward()

                # if self.train['clip_grad']:
                #     torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.)

                self.optimizer.step()

                total_steps += 1

            
            self.scheduler.step()
        


        print("Training finished!")

