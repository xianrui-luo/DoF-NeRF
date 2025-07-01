from cmath import nan
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import numpy as np
import imageio
import json
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange

import matplotlib.pyplot as plt

import nerf_model
from nerf_utils import *
from options import config_parser

from dataloader.load_llff import load_llff_data
from dataloader.load_bokeh import load_bokeh_data_train

from bokeh_utils import render_bokeh, render_bokeh_no_01_disp,render_path_bokeh

from log import LOG

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = False
N_pixel = 48
sample = 16

def runner_bokeh():
    parser = config_parser()
    args = parser.parse_args()
    if args.render_only:
        logger = LOG(name='RENDER',
                     dir=os.path.join(args.basedir, args.expname),
                     file='RenderLog.txt',
                     level='INFO')
    else:
        logger = LOG(name='TRAIN',
                     dir=os.path.join(args.basedir, args.expname),
                     file='TrainLog.txt',
                     level='INFO')
        logger_para = LOG(name='PARAM',
                          dir=os.path.join(args.basedir, args.expname),
                          file='ParaLog.txt',
                          level='INFO')

    # Load Data
    K = None
    if args.dataset_type == 'llff' or args.dataset_type == 'bokeh_llff':
        images, poses, bds, render_poses, i_test = load_llff_data(args.datadir, args.factor,
                                                                  recenter=True, bd_factor=.75,
                                                                  spherify=args.spherify)
        hwf = poses[0,:3,-1]
        poses = poses[:,:3,:4]
        # print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)
        logger.output_info(' '.join(['Loaded llff', str(images.shape), str(render_poses.shape), str(hwf), str(args.datadir)]))
        if not isinstance(i_test, list):
            i_test = [i_test]

        if args.llffhold > 0:
            # print('Auto LLFF holdout,', args.llffhold)
            logger.output_info(' '.join(['Auto LLFF holdout,', str(args.llffhold)]))
            # i_test = np.arange(images.shape[0])[::args.llffhold] + 1
            i_test = np.arange(images.shape[0])[::args.llffhold]

        i_val = i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                        (i not in i_test and i not in i_val)])

        print('DEFINING BOUNDS')
        if args.no_ndc:
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.
            
        else:
            near = 0.
            far = 1.
        # print('NEAR FAR', near, far)
        logger.output_info(' '.join(['NEAR FAR', str(near), str(far)]))

    elif args.dataset_type == 'bokeh':
        
        images_bak, images_obj, images_aif, poses, bds, render_poses = load_bokeh_data_train(args.datadir, args.factor,
                                                                  recenter=True, bd_factor=.75,
                                                                  spherify=args.spherify)
        
        hwf = poses[0,:3,-1]
        poses = poses[:,:3,:4]
        # print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)
        logger.output_info(' '.join(['Loaded bokeh data', str(images_aif.shape), str(render_poses.shape), str(hwf), str(args.datadir)]))

        logger.output_info(' '.join(['Auto bokeh holdout,', str(args.llffhold)]))
        i_test = np.arange(images_aif.shape[0])[::args.llffhold] + 1

        print('Separating Train Images')
        i_val = i_test
        i_train = np.array([i for i in np.arange(int(images_bak.shape[0])) if
                        (i not in i_test and i not in i_val)])
        i_bak = np.array([i for i in i_train if i % 2])
        i_obj = np.array([i for i in i_train if i not in i_bak])
        images = images_aif
        images[i_bak, ...] = images_bak[i_bak, ...]
        images[i_obj, ...] = images_obj[i_obj, ...]
        logger.output_info(' '.join(['OBJ focus views are', str(i_obj)]))
        logger.output_info(' '.join(['BAK focus views are', str(i_bak)]))

        print('DEFINING BOUNDS')
        if args.no_ndc:
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.
            
        else:
            near = 0.
            far = 1.
        # print('NEAR FAR', near, far)
        logger.output_info(' '.join(['NEAR FAR', str(near), str(far)]))

    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    if K is None:
        K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ])

    if args.render_test:
        render_poses = np.array(poses[i_test])
        # render_poses = np.array(poses[i_train])

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    # Create nerf model
    # if args.dataset_type == 'bokeh' or args.dataset_type == 'llff':
    N_image = images.shape[0]
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer, bokeh_param = create_nerf(args, N_image)
    # elif args.dataset_type == 'llff':
    #     render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)
    # else:
    #     print('Unknown dataset type', args.dataset_type, 'exiting')
    #     return
    global_step = start

    bds_dict = {
        'near' : near,
        'far' : far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # Move testing data to GPU
    render_poses = torch.Tensor(render_poses).to(device)

    if args.dataset_type == 'bokeh' or args.dataset_type == 'llff':
        K_bokeh, disp_focus = bokeh_param()
        print('K_bokeh')
        print(K_bokeh)
        print('disp_focus')
        print(disp_focus)

    # Short circuit if only rendering out from trained model
    # args.render_only = True
    if args.render_only:
        # print('RENDER ONLY')
        logger.output_info('RENDER ONLY')
        with torch.no_grad():
            if args.render_test:
                # render_test switches to test poses
                images = images[i_test]
                # images = images[i_train]
            else:
                # Default is smoother render_poses path
                images = None

            testsavedir = os.path.join(basedir, expname, 'renderonly_{}_{:06d}'.format('test' if args.render_test else 'path', start))
            os.makedirs(testsavedir, exist_ok=True)
            # print('test poses shape', render_poses.shape)
            logger.output_info(' '.join(['test poses shape', str(render_poses.shape)]))

            # rgbs, disps = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test, gt_imgs=images, savedir=testsavedir, render_factor=args.render_factor)
            rgbs, disps = render_path_bokeh(render_poses, 
                                            hwf, 
                                            K, 
                                            args.chunk, 
                                            render_kwargs_test,
                                            K_bokeh=8., 
                                            gamma=4, 
                                            disp_focus=90/255, 
                                            defocus_scale=1, 
                                            gt_imgs=None, 
                                            savedir=None, 
                                            render_factor=args.render_factor)
            # print('Done rendering', testsavedir)
            logger.output_info(' '.join(['Done rendering', str(testsavedir)]))
            imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'), nerf_model.to8b(rgbs), fps=30, quality=8)

            return

    # Prepare Batching Rays
    N_rand = args.N_rand
    print('get rays')
    rays = np.stack([nerf_model.get_rays_np(H, W, K, p) for p in poses[:,:3,:4]], 0) # [N, ro+rd, H, W, 3]
    print('done, concats')
    rays_rgb = np.concatenate([rays, images[:,None]], 1) # [N, ro+rd+rgb, H, W, 3]
    rays_rgb = np.transpose(rays_rgb, [0,2,3,1,4]) # [N, H, W, ro+rd+rgb, 3]
    rays_rgb = np.stack([rays_rgb[i] for i in i_train], 0) # train images only
    rays_rgb = np.reshape(rays_rgb, [-1,3,3]) # [(N-1)*H*W, ro+rd+rgb, 3]
    rays_rgb = rays_rgb.astype(np.float32)
    print('shuffle rays')
    np.random.shuffle(rays_rgb)
    print('done')
    i_batch = 0

    # Prepare Patch Rays
    N_patch_sample = 15   
    print('Preparing Patching Rays')
    rays_rgb_patch = np.concatenate([rays, images[:,None]], 1) # [N, ro+rd+rgb, H, W, 3]
    rays_rgb_patch = np.transpose(rays_rgb_patch, [0,2,3,1,4]) # [N, H, W, ro+rd+rgb, 3]
    # rays_rgb_patch = np.stack([rays_rgb_patch[i] for i in i_train], 0) # train images only [N-?, H, W, ro+rd+rgb, 3]
    H_patch = int((H-N_pixel)/N_patch_sample)
    W_patch = int((W-N_pixel)/N_patch_sample)
    patch_rays = []
    image_list = []
    for i in i_train:
        for h in range(H_patch+2):
            for w in range(W_patch+2):
                h0, w0 = h * N_patch_sample, w * N_patch_sample
                if h0 + N_pixel >= H:
                    h0 = H - N_pixel
                if w0 + N_pixel >= W:
                    w0 = W - N_pixel
                patch_rays.append(rays_rgb_patch[i, h0:h0+N_pixel, w0:w0+N_pixel, ...])
                image_list.append(i)
    print(len(patch_rays))
    patch_rays = np.stack(patch_rays)
    patch_rays = patch_rays.astype(np.float32)
    train_index = np.stack(list(range(len(patch_rays))))
    print('shuffle index') 
    np.random.shuffle(train_index)
    print('done')          
    i_patch = 0

    images = torch.Tensor(images).to(device)
    poses = torch.Tensor(poses).to(device)
    rays_rgb = torch.Tensor(rays_rgb).to(device)
    patch_rays = torch.Tensor(patch_rays).to(device)

    N_iters = args.N_iters+1
    logger.output_info('Begin')
    logger.output_info(' '.join(['TRAIN views are', str(i_train)]))
    logger.output_info(' '.join(['TEST views are', str(i_test)]))
    logger.output_info(' '.join(['VAL views are', str(i_val)]))

    loss_total = 0.
    psnr_total = 0.
    start = start + 1
    for i in trange(start, N_iters + args.bokeh_iters):
        time0 = time.time()
        max_disp = 1.0
        # if args.dataset_type == 'bokeh' or args.dataset_type == 'llff':
        para_K_bokeh, para_disp_focus = bokeh_param()
        # print('1')

        if i < N_iters:
            # Random over all images
            batch = rays_rgb[i_batch:i_batch+N_rand] # [B, 2+1, 3*?]
            batch = torch.transpose(batch, 0, 1)
            batch_rays, target_s = batch[:2], batch[2]

            i_batch += N_rand
            if i_batch >= rays_rgb.shape[0]:
                # print("Shuffle data after an epoch!")
                logger.output_info("Shuffle data after an epoch!")
                # rand_idx = torch.randperm(rays_rgb.shape[0])
                # rays_rgb = rays_rgb[rand_idx]
                rand_idx = np.arange(0,rays_rgb.shape[0],1,dtype=np.int64)
                np.random.shuffle(rand_idx)
                rays_rgb = rays_rgb[torch.as_tensor(rand_idx)]
                i_batch = 0

        else:
            
            index = train_index[i_patch]
            patch = patch_rays[index]
            img_i = image_list[index]
            # print('2')
            patch = torch.reshape(patch, [-1, patch.shape[-2], patch.shape[-1]])
            patch = torch.transpose(patch, 0, 1)
            batch_rays, target_s = patch[:2], patch[2]
            i_patch += 1
            if i_patch >= patch_rays.shape[0]:
                logger.output_info("Shuffle data after an epoch!")
                i_patch = 0
                np.random.shuffle(train_index)
                print('Done')



        rgb_ori, disp, acc, extras = render(H, 
                                            W, 
                                            K, 
                                            chunk=args.chunk, 
                                            rays=batch_rays,     
                                            verbose=i < 10, retraw=True,
                                            **render_kwargs_train)

        if i < N_iters:
            rgb = rgb_ori
        else:
            # print('3')
            rgb = torch.reshape(rgb_ori, [N_pixel, -1, 3])
            disp = torch.reshape(disp, [N_pixel, -1])
            rgb = render_bokeh_no_01_disp(rgb, 
                                          disp / 5., # disp / max_disp, 
                                          K_bokeh=para_K_bokeh[img_i] * 10,
                                          gamma=2,
                                          disp_focus=para_disp_focus[img_i],
                                          defocus_scale=1)
            print('4')
            rgb = torch.reshape(rgb, [-1, 3])

        optimizer.zero_grad()
        img_loss = nerf_model.img2mse(rgb, target_s)
        print('5')
        trans = extras['raw'][...,-1]
        loss = img_loss
        psnr = nerf_model.mse2psnr(img_loss)
        print('5')

        if 'rgb0' in extras:
            if i > N_iters:
                rgb_0 = torch.reshape(extras['rgb0'], [N_pixel, -1, 3])
                rgb_0 = render_bokeh_no_01_disp(rgb_0, 
                                                disp / 5. , # disp / max_disp, 
                                                K_bokeh=para_K_bokeh[img_i] * 10,
                                                gamma=2,
                                                disp_focus=para_disp_focus[img_i],
                                                defocus_scale=1)
                print('6')
                rgb_0 = torch.reshape(rgb_0, [-1, 3])
                img_loss0 = nerf_model.img2mse(rgb_0, target_s)
            else:
                img_loss0 = nerf_model.img2mse(extras['rgb0'], target_s)
            loss = loss + img_loss0
            psnr0 = nerf_model.mse2psnr(img_loss0)
        loss.backward()
        optimizer.step()

        loss_total += loss.item()
        psnr_total += psnr.item()

        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        if i < N_iters:
            new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        else:
            new_lrate = args.lrate_bokeh * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate

        if i%args.i_weights==0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            torch.save({
                'global_step': global_step,
                'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)
            # print('Saved checkpoints at', path)
            if args.dataset_type == 'bokeh' or args.dataset_type == 'llff':
                os.makedirs(os.path.join(basedir, expname, 'param'), exist_ok=True)
                path_param = os.path.join(basedir, expname, 'param', '{:06d}_param.tar'.format(i))
                torch.save(bokeh_param.state_dict(), path_param)
                logger.output_info(' '.join(['Saved checkpoints at', str(path)]))
                logger.output_info(' '.join(['Saved param checkpoints at', str(path_param)]))

        if i%args.i_testset==0 and i > 0:
            testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(i))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', poses[i_test].shape)
            with torch.no_grad():
                render_path(torch.Tensor(poses[i_test]).to(device), hwf, K, args.chunk, render_kwargs_test, gt_imgs=images[i_test], savedir=testsavedir)
            # print('Saved test set')
            logger.log_info('Saved test set')

        if i%args.i_video==0 and i > 0:
            # Turn on testing mode
            with torch.no_grad():
            #     rgbs, disps = render_path_bokeh(render_poses, 
            #                                     hwf, 
            #                                     K, 
            #                                     args.chunk, 
            #                                     render_kwargs_test,
            #                                     K_bokeh=0.1, 
            #                                     gamma=4, 
            #                                     disp_focus=90/255, 
            #                                     defocus_scale=1, 
            #                                     gt_imgs=None, 
            #                                     savedir=None, 
            #                                     render_factor=args.render_factor)
                rgbs, disps = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test)
            # print('Done, saving', rgbs.shape, disps.shape)
            logger.output_info(' '.join(['Done, saving', str(rgbs.shape), str(disps.shape)]))
            moviebase = os.path.join(basedir, expname, '{}_spiral_{:06d}_'.format(expname, i))
            imageio.mimwrite(moviebase + 'rgb.mp4', nerf_model.to8b(rgbs), fps=30, quality=8)
            imageio.mimwrite(moviebase + 'disp.mp4', nerf_model.to8b(disps / np.max(disps)), fps=30, quality=8)
            max_disp = np.max(disps)


        if i%args.i_print==0:

            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}")
            logger.log_info(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}")
            tqdm.write(f"[TOTAL] Iter: {i} Loss: {loss_total / args.i_print}  PSNR: {psnr_total / args.i_print}")
            logger.log_info(f"[TOTAL] Iter: {i} Loss: {loss_total / args.i_print}  PSNR: {psnr_total / args.i_print}")
            if (args.dataset_type == 'bokeh' or args.dataset_type == 'llff') and i > N_iters:
                logger_para.log_info(' '.join(['Logged at Iters: ', str(i)]))
                logger_para.log_info(' '.join(['Train param focus: ', str(para_disp_focus)]))
                logger_para.log_info(' '.join(['Train param aperture: ', str(para_K_bokeh)]))

            loss_total = 0.
            psnr_total = 0.

        global_step += 1


if __name__ == "__main__":
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    runner_bokeh()