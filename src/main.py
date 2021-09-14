import os
import torch
from tqdm import tqdm
from networks import define_G
from utils import make_define_dir, render, make_imageloader
import argparse


parser = argparse.ArgumentParser(description='a new parser')
parser.add_argument('--taskname', type=str, required=True, help='subdirectory name')
parser.add_argument('--gid', type=int, default=0, help='gpu id')
opt = parser.parse_args()

class arguments:
    def __init__(self, args):
        self.reslt_dir = 'results'
        self.taskname = args.taskname
        self.task_dir = os.path.join(self.reslt_dir, self.taskname)
        if not os.path.exists(self.task_dir):
            os.makedirs(self.task_dir)
            print('created ', self.task_dir)
        else:
            print('will overwrite:', self.task_dir)
            input("enter anything to confirm")

        infos = args.taskname.split('_')
        self.thresh_sc = 75

        self.modelpath = 'ckpt/blond.pth'  ##
        self.dataroot = 'spdt/blond'

        if infos[0] == 'm':
            self.method = 'MAIN'
        elif infos[0] == 'b':
            self.method = 'RGF'

        self.h = 0.001
        self.lr = 1
        self.epochs = 100
        self.set_q = 1000
        self.norm_est_num = 30
        self.gpu_id = args.gid 
        self.eps = 0.1
        
        msg = ['Performing LASGSA attack:', 
        '\n    one model:', self.modelpath, ', with dataset', self.dataroot, 
        '\n    on gpu:', self.gpu_id]
        print(*msg)

def main():
    args = arguments(opt)
    G = define_G(3, 3, 64, 'resnet_9blocks', norm='instance', use_dropout=False, init_type='normal', init_gain=0.02)
    G.load_state_dict(torch.load(args.modelpath))
    G.cuda(args.gpu_id)

    ptq_accumu = 0
    pass_count = 0
    score_accumu = 0

    all_dir = make_define_dir(os.path.join(args.task_dir, 'all'))

    dataloader = make_imageloader(args.dataroot)
    for index, x0 in enumerate(dataloader):
        TRACK_STAT1 = []
        TRACK_STAT2 = []

        image_dir = make_define_dir(os.path.join(args.task_dir, str(index)))
        x0 = x0.cuda(args.gpu_id)
        y0 = G(x0)
        render(x0, os.path.join(all_dir, str(index)+'-x0.jpg'))
        render(y0, os.path.join(all_dir, str(index)+'-y0.jpg'))

        X_var = x0.clone()

        score = lambda x: 1-((G(x) - x0)**2).sum()/((y0-x0)**2).sum()
        valt = lambda x: -((G(x) - x0)**2).sum()
        
        def clipping(X_var):
            X_var = torch.where(X_var < x0 - args.eps, x0 - args.eps, X_var)
            X_var = torch.where(X_var > x0 + args.eps, x0 + args.eps, X_var)
            X_var = X_var.clamp(-1, 1)
            return X_var

        total_query_counter = 1
        firstpass = True
        ptq = 0
        
        D = x0.numel()
        pbar = tqdm(range(args.epochs), ncols=70)

        for i in pbar:
            with torch.no_grad():
        
                standard = valt(X_var)
                total_query_counter += 1
                partial_F_partial_vec = lambda vec: (valt(X_var+vec*args.h)- standard)/args.h
        
                a = -1*(G(X_var) - x0)
                total_query_counter += 1
                ahat = a/a.norm()
                partial_G_partial_dir = lambda drct: (G(X_var + drct*args.h) - G(X_var))/args.h

                # Self-Guided
                vector = partial_G_partial_dir(ahat)
                total_query_counter += 1
                vector = vector/vector.norm()

                ## estimate grad norm
                est_norm_sqare_accumu = 0
                for j in range(args.norm_est_num):
                    r = torch.randn_like(X_var)
                    r /= r.norm() 
                    est_norm_sqare_accumu += partial_F_partial_vec(r) ** 2 * D
                    total_query_counter += 1
            
                est_norm_sqare = est_norm_sqare_accumu/args.norm_est_num
                est_norm = torch.sqrt(est_norm_sqare).cpu().item()
                est_alpha = partial_F_partial_vec(vector)/est_norm
                total_query_counter += 1

                D_2q_2 = (D + 2*args.set_q - 2)
                if est_alpha**2 < 1/D_2q_2:
                    lambda_star = torch.Tensor([0]).cuda(args.gpu_id)
                elif est_alpha**2 > (2*args.set_q - 1)/D_2q_2:
                    lambda_star = 1
                else:
                    denominator = (2*est_alpha**2*D*args.set_q - est_alpha**4*D*D_2q_2 - 1)
                    lambda_star = (1-est_alpha**2)*(est_alpha**2*D_2q_2 - 1)/denominator

                if lambda_star == 1:
                    g = vector
                else:

                    ## Limit-Aware
                    bound_clamp = lambda delta: clipping(X_var + delta) - X_var
                    g = torch.zeros_like(X_var)
                    for j in range(args.set_q):
                        upper_region = bound_clamp(torch.ones_like(X_var))
                        lower_region = bound_clamp(torch.ones_like(X_var)*(-1)) * (-1)
                        bound_region = torch.where(upper_region < lower_region, upper_region, lower_region)
                        bound_region += 1e-5

                        r = torch.randn_like(X_var)
                        r = r/r.norm()
                        r *= bound_region
                        last = r - vector.view(-1).dot(r.view(-1))*vector
                        last = last/last.norm()
                        u = torch.sqrt(lambda_star) * vector + torch.sqrt(1 - lambda_star)*last
                        g += partial_F_partial_vec(u) * u
                        total_query_counter += 1
                    g = g/args.set_q

                prev = X_var.clone()
                X_var = X_var + args.lr*g
                X_var = clipping(X_var)

                length = (args.lr*g).norm()

                ## Gradient-Sliding
                for __ in range(100):
                    nX_var = X_var + (X_var - prev) * (1- ((X_var - prev).norm()/length+1e-6))
                    nX_var = clipping(nX_var)
                    length -= (nX_var - X_var).norm()
                    if length <= 0:
                        break
                    prev = X_var.clone()
                    X_var = nX_var.clone()

                nscore = int(score(X_var)* 100)
                pbar.set_postfix(sc=nscore)

                render(X_var, os.path.join(image_dir, 'x-%d.jpg'%i))
                render(G(X_var), os.path.join(image_dir, 'y-%d-%d-%d.jpg'%(i, nscore, total_query_counter)))
                render(G(X_var), os.path.join(all_dir, str(index)+'-finaly.jpg'))
                render(X_var, os.path.join(all_dir, str(index)+'-finalx.jpg'))
                TRACK_STAT1.append(nscore)
                TRACK_STAT2.append(total_query_counter)
                
                ## statistics
                if nscore >= args.thresh_sc and firstpass:
                    ptq = total_query_counter
                    firstpass = False
                    pass_count += 1

        render(G(X_var), os.path.join(all_dir, str(index)+'-finaly.jpg'))
        render(X_var, os.path.join(all_dir, str(index)+'-finalx.jpg'))
        with open(os.path.join(args.task_dir, 'log.txt'), 'a') as f:
            f.write(str([index, ptq, nscore])+'\n')
            f.write(str(TRACK_STAT1)+'\n')
            f.write(str(TRACK_STAT2)+'\n')

        ptq_accumu += ptq
        score_accumu += nscore

        asr = pass_count / (index + 1)
        avgptq = ptq_accumu / pass_count  if pass_count > 0 else 'sad'
        avgscore = score_accumu / (index + 1)

    asr = pass_count / (index + 1)
    avgptq = ptq_accumu / pass_count  if pass_count > 0 else 'sad'
    avgscore = score_accumu / (index + 1)

    print('FINALL results, ASR', asr, 'AVG, Q', avgptq, 'scbar', avgscore)
    with open(os.path.join(args.task_dir, 'log.txt'), 'a') as f:
        f.write('FINALL results, ASR'+ str(asr)+ 'AVG, Q' +str(avgptq) + 'scbar'+ str(avgscore))

if __name__ == '__main__':
    main()

