import time, itertools
from dataset import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from networks import *
from utils import *
from glob import glob
from collections import OrderedDict

import torch.distributed as dist
from torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook import PowerSGDState
from torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook import powerSGD_hook

local_rank = int(os.environ["LOCAL_RANK"])
world_size = int(os.environ["WORLD_SIZE"])
rank = int(os.environ["RANK"])

print("rank: %d, local_rank: %d" % (rank, local_rank))

LOW_RANK = 1

USING_SYNTHETIC_DATA = True

class SyntheticDataLoader:
    def __init__(self):
        self.data = torch.randn([1, 3, 256, 256])
        self.trash = torch.randn([1])
        
    def __iter__(self):
        return self

    def __next__(self):
        return self.data, self.trash
    
    def next(self):
        return self.__next__()

def ddp_wrapper(model):
    return torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

class UGATITModule(nn.Module):
    def __init__(self, genA2B, genB2A, disGA, disGB, disLA, disLB):
        super().__init__()
        self.genA2B = genA2B
        self.genB2A = genB2A
        self.disGA = disGA
        self.disGB = disGB
        self.disLA = disLA
        self.disLB = disLB

    def forward(self, real_A, real_B, update):
        # update D
        if update == 'D':
            fake_A2B, _, _ = self.genA2B(real_A)
            fake_B2A, _, _ = self.genB2A(real_B)

            real_GA_logit, real_GA_cam_logit, _ = self.disGA(real_A)
            real_LA_logit, real_LA_cam_logit, _ = self.disLA(real_A)
            real_GB_logit, real_GB_cam_logit, _ = self.disGB(real_B)
            real_LB_logit, real_LB_cam_logit, _ = self.disLB(real_B)

            fake_GA_logit, fake_GA_cam_logit, _ = self.disGA(fake_B2A)
            fake_LA_logit, fake_LA_cam_logit, _ = self.disLA(fake_B2A)
            fake_GB_logit, fake_GB_cam_logit, _ = self.disGB(fake_A2B)
            fake_LB_logit, fake_LB_cam_logit, _ = self.disLB(fake_A2B)

            return real_GA_logit, real_GA_cam_logit, \
                real_LA_logit, real_LA_cam_logit, \
                real_GB_logit, real_GB_cam_logit, \
                real_LB_logit, real_LB_cam_logit, \
                fake_GA_logit, fake_GA_cam_logit, \
                fake_LA_logit, fake_LA_cam_logit, \
                fake_GB_logit, fake_GB_cam_logit, \
                fake_LB_logit, fake_LB_cam_logit
        # update G
        if update == 'G':
            fake_A2B, fake_A2B_cam_logit, _ = self.genA2B(real_A)
            fake_B2A, fake_B2A_cam_logit, _ = self.genB2A(real_B)

            fake_A2B2A, _, _ = self.genB2A(fake_A2B)
            fake_B2A2B, _, _ = self.genA2B(fake_B2A)

            fake_A2A, fake_A2A_cam_logit, _ = self.genB2A(real_A)
            fake_B2B, fake_B2B_cam_logit, _ = self.genA2B(real_B)

            fake_GA_logit, fake_GA_cam_logit, _ = self.disGA(fake_B2A)
            fake_LA_logit, fake_LA_cam_logit, _ = self.disLA(fake_B2A)
            fake_GB_logit, fake_GB_cam_logit, _ = self.disGB(fake_A2B)
            fake_LB_logit, fake_LB_cam_logit, _ = self.disLB(fake_A2B)

            return fake_A2B_cam_logit, fake_B2A_cam_logit, \
                fake_A2B2A, fake_B2A2B, \
                fake_A2A, fake_A2A_cam_logit, \
                fake_B2B, fake_B2B_cam_logit, \
                fake_GA_logit, fake_GA_cam_logit, \
                fake_LA_logit, fake_LA_cam_logit, \
                fake_GB_logit, fake_GB_cam_logit, \
                fake_LB_logit, fake_LB_cam_logit



class UGATIT(object) :
    def __init__(self, args):
        self.light = args.light

        if self.light :
            self.model_name = 'UGATIT_light'
        else :
            self.model_name = 'UGATIT'

        self.result_dir = args.result_dir
        self.dataset_dir = args.dataset_dir
        self.dataset = args.dataset

        self.iteration = args.iteration

        self.decay_flag = args.decay_flag

        self.batch_size = args.batch_size
        self.print_freq = args.print_freq
        self.save_freq = args.save_freq

        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.ch = args.ch

        """ Weight """
        self.adv_weight = args.adv_weight
        self.cycle_weight = args.cycle_weight
        self.identity_weight = args.identity_weight
        self.cam_weight = args.cam_weight

        """ Generator """
        self.n_res = args.n_res

        """ Discriminator """
        self.n_dis = args.n_dis

        self.img_size = args.img_size
        self.img_ch = args.img_ch

        self.device = args.device
        self.benchmark_flag = args.benchmark_flag
        self.resume = args.resume
        self.fix_aug = args.fix_aug
        self.list_mode = args.list_mode

        self.powersgd = args.powersgd

        if torch.backends.cudnn.enabled and self.benchmark_flag:
            print('set benchmark !')
            torch.backends.cudnn.benchmark = True

        print()

        print("##### Information #####")
        print("# light : ", self.light)
        print("# dataset : ", self.dataset)
        print("# batch_size : ", self.batch_size)
        print("# iteration per epoch : ", self.iteration)

        print()

        print("##### Generator #####")
        print("# residual blocks : ", self.n_res)

        print()

        print("##### Discriminator #####")
        print("# discriminator layer : ", self.n_dis)

        print()

        print("##### Weight #####")
        print("# adv_weight : ", self.adv_weight)
        print("# cycle_weight : ", self.cycle_weight)
        print("# identity_weight : ", self.identity_weight)
        print("# cam_weight : ", self.cam_weight)

    ##################################################################################
    # Model
    ##################################################################################

    def build_model(self):
        """ DataLoader """

        if self.fix_aug:
            print("FIX AUG ON")
            train_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
            ])
        else:
            train_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.Resize((self.img_size + 30, self.img_size+30)),
                transforms.RandomCrop(self.img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
            ])

        test_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        if USING_SYNTHETIC_DATA:
            print("========== Using Synthetic Data! ==========")
            self.trainA_loader = SyntheticDataLoader()
            self.trainB_loader = SyntheticDataLoader()
            self.testA_loader = SyntheticDataLoader()
            self.testB_loader = SyntheticDataLoader()
        else:
            self.trainA = ImageFolder(os.path.join(self.dataset_dir, self.dataset, 'trainA'), train_transform, list_mode=self.list_mode)
            self.trainB = ImageFolder(os.path.join(self.dataset_dir, self.dataset, 'trainB'), train_transform, list_mode=self.list_mode)
            self.testA = ImageFolder(os.path.join(self.dataset_dir, self.dataset, 'testA'), test_transform, list_mode=self.list_mode)
            self.testB = ImageFolder(os.path.join(self.dataset_dir, self.dataset, 'testB'), test_transform, list_mode=self.list_mode)
            
            trainA_sampler = torch.utils.data.distributed.DistributedSampler(
                self.trainA, num_replicas=world_size, rank=rank)
            trainB_sampler = torch.utils.data.distributed.DistributedSampler(
                self.trainB, num_replicas=world_size, rank=rank)
            testA_sampler = torch.utils.data.distributed.DistributedSampler(
                self.testA, num_replicas=world_size, rank=rank)
            testB_sampler = torch.utils.data.distributed.DistributedSampler(
                self.testB, num_replicas=world_size, rank=rank)

            self.trainA_loader = DataLoader(self.trainA, batch_size=self.batch_size, sampler=trainA_sampler, num_workers=1)
            self.trainB_loader = DataLoader(self.trainB, batch_size=self.batch_size, sampler=trainB_sampler, num_workers=1)
            self.testA_loader = DataLoader(self.testA, batch_size=1, sampler=testA_sampler)
            self.testB_loader = DataLoader(self.testB, batch_size=1, sampler=testB_sampler)

        """ Define Generator, Discriminator """
        low_rank = LOW_RANK
        self.genA2B = ResnetGenerator(input_nc=3, output_nc=3, ngf=self.ch, n_blocks=self.n_res, img_size=self.img_size, light=self.light).to(self.device)
        self.genB2A = ResnetGenerator(input_nc=3, output_nc=3, ngf=self.ch, n_blocks=self.n_res, img_size=self.img_size, light=self.light).to(self.device)
        self.disGA = Discriminator(input_nc=3, ndf=self.ch, n_layers=7).to(self.device)
        self.disGB = Discriminator(input_nc=3, ndf=self.ch, n_layers=7).to(self.device)
        self.disLA = Discriminator(input_nc=3, ndf=self.ch, n_layers=5).to(self.device)
        self.disLB = Discriminator(input_nc=3, ndf=self.ch, n_layers=5).to(self.device)

        self.ugatit_model = UGATITModule(self.genA2B, self.genB2A, self.disGA, self.disGB, self.disLA, self.disLB)
        self.ugatit_model = ddp_wrapper(self.ugatit_model)

        powersgd_state = PowerSGDState(
            process_group=dist.distributed_c10d.group.WORLD,
            matrix_approximation_rank=low_rank,
            use_error_feedback=True,
            warm_start=False,
            start_powerSGD_iter=2)
        if self.powersgd:
            self.ugatit_model.register_comm_hook(powersgd_state, powerSGD_hook)



        """ Define Loss """
        self.L1_loss = nn.L1Loss().to(self.device)
        self.MSE_loss = nn.MSELoss().to(self.device)
        self.BCE_loss = nn.BCEWithLogitsLoss().to(self.device)

        """ Trainer """
        self.G_optim = torch.optim.Adam(itertools.chain(self.genA2B.parameters(), self.genB2A.parameters()), 
                                        lr=self.lr,
                                        betas=(0.5, 0.999),
                                        weight_decay=self.weight_decay)
        self.D_optim = torch.optim.Adam(itertools.chain(self.disGA.parameters(), self.disGB.parameters(), self.disLA.parameters(), self.disLB.parameters()), 
                                        lr=self.lr,
                                        betas=(0.5, 0.999),
                                        weight_decay=self.weight_decay)

        """ Define Rho clipper to constraint the value of rho in AdaILN and ILN"""
        self.Rho_clipper = RhoClipper(0, 1)

    def train(self):
        # self.genA2B.train(), self.genB2A.train(), self.disGA.train(), self.disGB.train(), self.disLA.train(), self.disLB.train()
        self.ugatit_model.train()

        start_iter = 1

        # training loop
        print('training start !')
        start_time = time.time()
        last_time = start_time
        for step in range(start_iter, self.iteration + 1):
            if self.decay_flag and step > (self.iteration // 2):
                self.G_optim.param_groups[0]['lr'] -= (self.lr / (self.iteration // 2))
                self.D_optim.param_groups[0]['lr'] -= (self.lr / (self.iteration // 2))

            try:
                real_A, _ = trainA_iter.next()
            except:
                trainA_iter = iter(self.trainA_loader)
                real_A, _ = trainA_iter.next()

            try:
                real_B, _ = trainB_iter.next()
            except:
                trainB_iter = iter(self.trainB_loader)
                real_B, _ = trainB_iter.next()

            real_A, real_B = real_A.to(self.device), real_B.to(self.device)

            # Update D
            self.D_optim.zero_grad()

            real_GA_logit, real_GA_cam_logit, \
                real_LA_logit, real_LA_cam_logit, \
                real_GB_logit, real_GB_cam_logit, \
                real_LB_logit, real_LB_cam_logit, \
                fake_GA_logit, fake_GA_cam_logit, \
                fake_LA_logit, fake_LA_cam_logit, \
                fake_GB_logit, fake_GB_cam_logit, \
                fake_LB_logit, fake_LB_cam_logit = self.ugatit_model(real_A, real_B, 'D')
            # fake_A2B, _, _ = self.genA2B(real_A)
            # fake_B2A, _, _ = self.genB2A(real_B)

            # real_GA_logit, real_GA_cam_logit, _ = self.disGA(real_A)
            # real_LA_logit, real_LA_cam_logit, _ = self.disLA(real_A)
            # real_GB_logit, real_GB_cam_logit, _ = self.disGB(real_B)
            # real_LB_logit, real_LB_cam_logit, _ = self.disLB(real_B)

            # fake_GA_logit, fake_GA_cam_logit, _ = self.disGA(fake_B2A)
            # fake_LA_logit, fake_LA_cam_logit, _ = self.disLA(fake_B2A)
            # fake_GB_logit, fake_GB_cam_logit, _ = self.disGB(fake_A2B)
            # fake_LB_logit, fake_LB_cam_logit, _ = self.disLB(fake_A2B)

            D_ad_loss_GA = self.MSE_loss(real_GA_logit, torch.ones_like(real_GA_logit).to(self.device)) + self.MSE_loss(fake_GA_logit, torch.zeros_like(fake_GA_logit).to(self.device))
            D_ad_cam_loss_GA = self.MSE_loss(real_GA_cam_logit, torch.ones_like(real_GA_cam_logit).to(self.device)) + self.MSE_loss(fake_GA_cam_logit, torch.zeros_like(fake_GA_cam_logit).to(self.device))
            D_ad_loss_LA = self.MSE_loss(real_LA_logit, torch.ones_like(real_LA_logit).to(self.device)) + self.MSE_loss(fake_LA_logit, torch.zeros_like(fake_LA_logit).to(self.device))
            D_ad_cam_loss_LA = self.MSE_loss(real_LA_cam_logit, torch.ones_like(real_LA_cam_logit).to(self.device)) + self.MSE_loss(fake_LA_cam_logit, torch.zeros_like(fake_LA_cam_logit).to(self.device))
            D_ad_loss_GB = self.MSE_loss(real_GB_logit, torch.ones_like(real_GB_logit).to(self.device)) + self.MSE_loss(fake_GB_logit, torch.zeros_like(fake_GB_logit).to(self.device))
            D_ad_cam_loss_GB = self.MSE_loss(real_GB_cam_logit, torch.ones_like(real_GB_cam_logit).to(self.device)) + self.MSE_loss(fake_GB_cam_logit, torch.zeros_like(fake_GB_cam_logit).to(self.device))
            D_ad_loss_LB = self.MSE_loss(real_LB_logit, torch.ones_like(real_LB_logit).to(self.device)) + self.MSE_loss(fake_LB_logit, torch.zeros_like(fake_LB_logit).to(self.device))
            D_ad_cam_loss_LB = self.MSE_loss(real_LB_cam_logit, torch.ones_like(real_LB_cam_logit).to(self.device)) + self.MSE_loss(fake_LB_cam_logit, torch.zeros_like(fake_LB_cam_logit).to(self.device))

            D_loss_A = self.adv_weight * (D_ad_loss_GA + D_ad_cam_loss_GA + D_ad_loss_LA + D_ad_cam_loss_LA)
            D_loss_B = self.adv_weight * (D_ad_loss_GB + D_ad_cam_loss_GB + D_ad_loss_LB + D_ad_cam_loss_LB)

            Discriminator_loss = D_loss_A + D_loss_B
            Discriminator_loss.backward()
            self.D_optim.step()

            # Update G
            self.G_optim.zero_grad()

            # fake_A2B, fake_A2B_cam_logit, _ = self.genA2B(real_A)
            # fake_B2A, fake_B2A_cam_logit, _ = self.genB2A(real_B)

            # fake_A2B2A, _, _ = self.genB2A(fake_A2B)
            # fake_B2A2B, _, _ = self.genA2B(fake_B2A)

            # fake_A2A, fake_A2A_cam_logit, _ = self.genB2A(real_A)
            # fake_B2B, fake_B2B_cam_logit, _ = self.genA2B(real_B)

            # fake_GA_logit, fake_GA_cam_logit, _ = self.disGA(fake_B2A)
            # fake_LA_logit, fake_LA_cam_logit, _ = self.disLA(fake_B2A)
            # fake_GB_logit, fake_GB_cam_logit, _ = self.disGB(fake_A2B)
            # fake_LB_logit, fake_LB_cam_logit, _ = self.disLB(fake_A2B)

            fake_A2B_cam_logit, fake_B2A_cam_logit, \
            fake_A2B2A, fake_B2A2B, \
            fake_A2A, fake_A2A_cam_logit, \
            fake_B2B, fake_B2B_cam_logit, \
            fake_GA_logit, fake_GA_cam_logit, \
            fake_LA_logit, fake_LA_cam_logit, \
            fake_GB_logit, fake_GB_cam_logit, \
            fake_LB_logit, fake_LB_cam_logit = self.ugatit_model(real_A, real_B, 'G')

            G_ad_loss_GA = self.MSE_loss(fake_GA_logit, torch.ones_like(fake_GA_logit).to(self.device))
            G_ad_cam_loss_GA = self.MSE_loss(fake_GA_cam_logit, torch.ones_like(fake_GA_cam_logit).to(self.device))
            G_ad_loss_LA = self.MSE_loss(fake_LA_logit, torch.ones_like(fake_LA_logit).to(self.device))
            G_ad_cam_loss_LA = self.MSE_loss(fake_LA_cam_logit, torch.ones_like(fake_LA_cam_logit).to(self.device))
            G_ad_loss_GB = self.MSE_loss(fake_GB_logit, torch.ones_like(fake_GB_logit).to(self.device))
            G_ad_cam_loss_GB = self.MSE_loss(fake_GB_cam_logit, torch.ones_like(fake_GB_cam_logit).to(self.device))
            G_ad_loss_LB = self.MSE_loss(fake_LB_logit, torch.ones_like(fake_LB_logit).to(self.device))
            G_ad_cam_loss_LB = self.MSE_loss(fake_LB_cam_logit, torch.ones_like(fake_LB_cam_logit).to(self.device))

            G_recon_loss_A = self.L1_loss(fake_A2B2A, real_A)
            G_recon_loss_B = self.L1_loss(fake_B2A2B, real_B)

            G_identity_loss_A = self.L1_loss(fake_A2A, real_A)
            G_identity_loss_B = self.L1_loss(fake_B2B, real_B)

            G_cam_loss_A = self.BCE_loss(fake_B2A_cam_logit, torch.ones_like(fake_B2A_cam_logit).to(self.device)) + self.BCE_loss(fake_A2A_cam_logit, torch.zeros_like(fake_A2A_cam_logit).to(self.device))
            G_cam_loss_B = self.BCE_loss(fake_A2B_cam_logit, torch.ones_like(fake_A2B_cam_logit).to(self.device)) + self.BCE_loss(fake_B2B_cam_logit, torch.zeros_like(fake_B2B_cam_logit).to(self.device))

            G_loss_A =  self.adv_weight * (G_ad_loss_GA + G_ad_cam_loss_GA + G_ad_loss_LA + G_ad_cam_loss_LA) + self.cycle_weight * G_recon_loss_A + self.identity_weight * G_identity_loss_A + self.cam_weight * G_cam_loss_A
            G_loss_B = self.adv_weight * (G_ad_loss_GB + G_ad_cam_loss_GB + G_ad_loss_LB + G_ad_cam_loss_LB) + self.cycle_weight * G_recon_loss_B + self.identity_weight * G_identity_loss_B + self.cam_weight * G_cam_loss_B

            Generator_loss = G_loss_A + G_loss_B
            Generator_loss.backward()
            self.G_optim.step()

            # clip parameter of AdaILN and ILN, applied after optimizer step
            self.genA2B.apply(self.Rho_clipper)
            self.genB2A.apply(self.Rho_clipper)

            if rank == 0:
                this_time = time.time()
                speed = world_size * self.batch_size / (this_time - last_time)
                print("Batch[%5d/%5d] time: %4.4f Speed: %4.4f samples/sec, d_loss: %.8f, g_loss: %.8f"  % (step, self.iteration, 
                            this_time - start_time, speed, Discriminator_loss, Generator_loss))
                # last_time = this_time
                last_time = time.time()

    def save(self, dir, step):
        if rank != 0:
            return
        params = {}
        params['genA2B'] = self.genA2B.state_dict()
        params['genB2A'] = self.genB2A.state_dict()
        params['disGA'] = self.disGA.state_dict()
        params['disGB'] = self.disGB.state_dict()
        params['disLA'] = self.disLA.state_dict()
        params['disLB'] = self.disLB.state_dict()
        torch.save(params, os.path.join(dir, self.dataset + '_params_%07d.pt' % step))

    def load(self, dir, step):
        # add map_location to avoid oom
        params = torch.load(os.path.join(dir, self.dataset + '_params_%07d.pt' % step), map_location='cpu')
        self.genA2B.load_state_dict(params['genA2B'])
        self.genB2A.load_state_dict(params['genB2A'])
        self.disGA.load_state_dict(params['disGA'])
        self.disGB.load_state_dict(params['disGB'])
        self.disLA.load_state_dict(params['disLA'])
        self.disLB.load_state_dict(params['disLB'])

    def test(self):
        model_list = glob(os.path.join(self.result_dir, self.dataset, 'model', '*.pt'))
        if not len(model_list) == 0:
            model_list.sort()
            iter = int(model_list[-1].split('_')[-1].split('.')[0])
            self.load(os.path.join(self.result_dir, self.dataset, 'model'), iter)
            print(" [*] Load SUCCESS")
        else:
            print(" [*] Load FAILURE")
            return

        self.genA2B.eval(), self.genB2A.eval()
        for n, (real_A, _) in enumerate(self.testA_loader):
            real_A = real_A.to(self.device)

            fake_A2B, _, fake_A2B_heatmap = self.genA2B(real_A)

            fake_A2B2A, _, fake_A2B2A_heatmap = self.genB2A(fake_A2B)

            fake_A2A, _, fake_A2A_heatmap = self.genB2A(real_A)

            A2B = np.concatenate((RGB2BGR(tensor2numpy(denorm(real_A[0]))),
                                  cam(tensor2numpy(fake_A2A_heatmap[0]), self.img_size),
                                  RGB2BGR(tensor2numpy(denorm(fake_A2A[0]))),
                                  cam(tensor2numpy(fake_A2B_heatmap[0]), self.img_size),
                                  RGB2BGR(tensor2numpy(denorm(fake_A2B[0]))),
                                  cam(tensor2numpy(fake_A2B2A_heatmap[0]), self.img_size),
                                  RGB2BGR(tensor2numpy(denorm(fake_A2B2A[0])))), 0)

            cv2.imwrite(os.path.join(self.result_dir, self.dataset, 'test', 'A2B_%d.png' % (n + 1)), A2B * 255.0)

        for n, (real_B, _) in enumerate(self.testB_loader):
            real_B = real_B.to(self.device)

            fake_B2A, _, fake_B2A_heatmap = self.genB2A(real_B)

            fake_B2A2B, _, fake_B2A2B_heatmap = self.genA2B(fake_B2A)

            fake_B2B, _, fake_B2B_heatmap = self.genA2B(real_B)

            B2A = np.concatenate((RGB2BGR(tensor2numpy(denorm(real_B[0]))),
                                  cam(tensor2numpy(fake_B2B_heatmap[0]), self.img_size),
                                  RGB2BGR(tensor2numpy(denorm(fake_B2B[0]))),
                                  cam(tensor2numpy(fake_B2A_heatmap[0]), self.img_size),
                                  RGB2BGR(tensor2numpy(denorm(fake_B2A[0]))),
                                  cam(tensor2numpy(fake_B2A2B_heatmap[0]), self.img_size),
                                  RGB2BGR(tensor2numpy(denorm(fake_B2A2B[0])))), 0)

            cv2.imwrite(os.path.join(self.result_dir, self.dataset, 'test', 'B2A_%d.png' % (n + 1)), B2A * 255.0)
