import torch
import torch.nn.functional as F
import time
import os
import re
import numpy as np
import neurokit2 as nk
import pickle

from utils.path_utils import ensure_exists

def simplify_label(label):
    label = label.lower()
    label = re.sub(r"[^\w\s]", "", label)
    label = re.sub(r"\s+", " ", label)
    label = label.strip()
    core = [
    "sinus rhythm",
    "sinus bradycardia",
    "sinus tachycardia",
    "atrial fibrillation",
    "atrial flutter",
    "supraventricular tachycardia",
    "ventricular tachycardia",
    "premature atrial contractions",
    "premature ventricular contractions",
    "left bundle branch block",
    "right bundle branch block",
    "left ventricular hypertrophy",
    "right ventricular hypertrophy",
    "wpw pattern",
    "prolonged qt",
    "st elevation",
    "st depression",
    "t wave inversion",
    "ischemia",
    "infarct (old/acute/anterolateral/etc.)"
]

    simp = 'other'
    for c in core:
        if c in label:
            simp = c
            break
    return simp

def extract_first_cycle(ecg_pred, sampling_rate=102.4, cycle_len=60):
    ecg_cycles = []
    B, n_lead, L = ecg_pred.shape
    ecg_pred_np = ecg_pred.detach().cpu().numpy()
    window_b = int(0.2 * sampling_rate)
    window_a = int(0.4 * sampling_rate)
    for i in range(B):
        leads_cycles = []
        for lead in range(n_lead):
            signal = ecg_pred_np[i, lead]
            try:
                _, r = nk.ecg_peaks(signal, sampling_rate=sampling_rate)
                r_idx = r['ECG_R_Peaks']
            except Exception:
                r_idx = []
            if len(r_idx) < 2:
                cycle = np.zeros(cycle_len, dtype=np.float32)
            else:
                sec = r_idx[1]
                st = max(sec - window_b, 0)
                ed = min(sec + window_a, signal.shape[0])
                cycle = signal[st:ed]
                if len(cycle) < cycle_len:
                    cycle = np.pad(cycle, (0, cycle_len - len(cycle)))
                else:
                    cycle = cycle[:cycle_len]
            leads_cycles.append(torch.tensor(cycle, dtype=torch.float32))
        leads_cycles = torch.stack(leads_cycles)
        ecg_cycles.append(leads_cycles)
    return torch.stack(ecg_cycles)


LEAD_IDX = {"I":0,"II":1,"III":2,"aVR":3,"aVL":4,"aVF":5,"V1":6,"V2":7,"V3":8,"V4":9,"V5":10,"V6":11}


def interlead_constraint_loss(gen_ecg_one, sim_ecg_one,
                              cycle_len=61,
                              use_delta_t=True,
                              sampling_rate=102.4,
                              lead_idx=LEAD_IDX):
    if not torch.is_tensor(gen_ecg_one):
        gen_ecg_one = torch.as_tensor(gen_ecg_one)
    if not torch.is_tensor(sim_ecg_one):
        sim_ecg_one = torch.as_tensor(sim_ecg_one, dtype=gen_ecg_one.dtype, device=gen_ecg_one.device)
    else:
        sim_ecg_one = sim_ecg_one.to(device=gen_ecg_one.device, dtype=gen_ecg_one.dtype)

    g = gen_ecg_one
    s = sim_ecg_one

    dg = g[..., 1:] - g[..., :-1]
    ds = s[..., 1:] - s[..., :-1]
    if use_delta_t:
        dt = 1.0 / float(sampling_rate)
        dg = dg / dt
        ds = ds / dt

    I, II, III = lead_idx["I"], lead_idx["II"], lead_idx["III"]
    aVR, aVL, aVF = lead_idx["aVR"], lead_idx["aVL"], lead_idx["aVF"]

    r1 = dg[I]   - (ds[II] - ds[III])
    r2 = dg[II]  - (ds[I]  + ds[III])
    r3 = dg[III] - (ds[II] - ds[I])
    r4 = dg[aVR] - (-(ds[I]  + ds[II]) * 0.5)
    r5 = dg[aVL] - ((ds[I]  - ds[III]) * 0.5)
    r6 = dg[aVF] - ((ds[II] + ds[III]) * 0.5)

    res = torch.stack([r1, r2, r3, r4, r5, r6], dim=0)
    return (res ** 2).mean()


def get_simulator_loss(diffused_model, decoder, xt, noise_estim, label, t, fz_val_dict, logger):
    start = time.time()
    device = xt.device

    with torch.no_grad():
        alphas_cumprod = diffused_model.alphas_cumprod.to(device)
        alpha_t = alphas_cumprod[t].view(-1, 1, 1)
        x0_pred = ((xt - (1 - alpha_t).sqrt() * noise_estim) / alpha_t.sqrt()).float()

        ecg_pred = decoder(x0_pred)
        ecg_pred = ecg_pred.permute(0, 2, 1)

    sampling_rate = 102.4
    cycle_len = 61
    delta_t = 1.0 / sampling_rate

    total_loss = torch.tensor(0.0, device=device)
    total_count = 0
    total_inter_loss = 0

    B = ecg_pred.shape[0]
    n_lead = 12

    for b in range(B):
        lbl = label[b]
        simp_lbl = simplify_label(lbl)
        if simp_lbl == 'other':
            continue

        if simp_lbl not in fz_val_dict:
            continue

        for lead in range(n_lead):
            h = ecg_pred[b, lead]
            fz_vals = fz_val_dict[simp_lbl][lead]
            diffs = (h[1:cycle_len] - h[:cycle_len-1]) / delta_t
            fz_vals = torch.as_tensor(fz_vals, device=device, dtype=diffs.dtype)

            fz_t = (fz_vals[1:cycle_len] - fz_vals[:cycle_len-1]) / delta_t
            lead_loss = torch.mean((diffs - fz_t) ** 2)

            total_loss = total_loss + lead_loss
        
        total_count += 1

        sim_ecg_one = torch.as_tensor(fz_val_dict[simp_lbl], device=device, dtype=ecg_pred.dtype)

        inter_loss = interlead_constraint_loss(
            gen_ecg_one = ecg_pred[b, :, :],
            sim_ecg_one = sim_ecg_one,
            cycle_len   = 61,
            use_delta_t = True,
            sampling_rate = 102.4)

        total_inter_loss = total_inter_loss + inter_loss

    if total_count == 0:
        return torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)

    return total_loss / total_count, total_inter_loss / total_count


def train_epoch_channels(dataloader, 
                         unet, 
                         decoder,
                         diffused_model, 
                         condition, 
                         optimizer, 
                         scheduler,
                         device,
                         fz_val_dict,
                         logger,
                         number_of_repetition=1):
    loss_list = []
    mse_list, sim_list = [], []

    unet.train()
    for _ in range(number_of_repetition):
        for data, label in dataloader:
            text_embed = label['text_embed']

            embed_lb = torch.stack(label['text_embed'], dim=0)
            embed_bl = embed_lb.t()
            text_embed = embed_bl.unsqueeze(1)
            text_embed = text_embed.float().to(device)

            latent = data.to(device)

            t = torch.randint(1, diffused_model.config.num_train_timesteps - 1, (latent.shape[0],))

            noise = torch.randn(latent.shape, device=latent.device)
            xt = diffused_model.add_noise(latent, noise, t)

            xt = xt.to(device)
            t = t.to(device)
            noise = noise.to(device)

            if condition:
                gender = []
                age = label['age']
                hr = label['hr']
                condition_dict = {}
            
                for ch in label['gender']:
                    if ch == 'M':
                        gender.append(1)
                    else:
                        gender.append(0)
                
                gender = np.array(gender)
                gender = np.repeat(gender[:, np.newaxis], 1, axis=1)
                gender = np.repeat(gender[:, :, np.newaxis], 1, axis=2)
                gender = torch.Tensor(gender)
                gender = gender.to(device)
                condition_dict.update({'gender': gender})

                age = np.array(age)
                age = np.repeat(age[:, np.newaxis], 1, axis=1)
                age = np.repeat(age[:, :, np.newaxis], 1, axis=2)
                age = torch.Tensor(age)
                age = age.to(device)
                condition_dict.update({'age': age})

                hr = np.array(hr)
                hr = np.repeat(hr[:, np.newaxis], 1, axis=1)
                hr = np.repeat(hr[:, :, np.newaxis], 1, axis=2)
                hr = torch.Tensor(hr)
                hr = hr.to(device)
                condition_dict.update({'heart rate': hr})

                for key in condition_dict:
                    condition_dict[key] = condition_dict[key].to(device)

                noise_estim = unet(xt, t, text_embed, condition_dict)
            else: 
                noise_estim = unet(xt, t, text_embed)

            simulator_loss, inter_loss = get_simulator_loss(diffused_model, decoder, xt, noise_estim, label['text'], t, fz_val_dict, logger)
            mse_loss = F.mse_loss(noise_estim, noise, reduction='sum').div(noise.size(0))

            W_SIM, W_INTER = 3e-3, 5e-2
            simulator_loss = W_SIM * simulator_loss  
            inter_loss = W_INTER * inter_loss

            loss = mse_loss + simulator_loss*0.0001 + inter_loss*0

            loss_list.append(loss.item())
            mse_list.append(mse_loss.item())
            sim_list.append(simulator_loss.item())

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

    total_mean = sum(loss_list) / len(loss_list)
    mse_mean   = sum(mse_list) / len(mse_list)
    sim_mean   = sum(sim_list) / len(sim_list)
    return total_mean, mse_mean, sim_mean

def train_model(meta, 
                save_weights_path, 
                dataloader,  
                diffused_model, 
                unet, 
                decoder,
                simulator_prior_path,
                h_, 
                logger):
 
    device = torch.device(meta['device'] if torch.cuda.is_available() else "cpu")
    unet = unet.to(device)
    optimizer = torch.optim.AdamW(params=unet.parameters(), lr=h_['lr'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=h_['epochs']*len(dataloader), eta_min=0.1*h_['lr'])


    simulator_prior_path = ensure_exists(simulator_prior_path, "simulator prior pickle")
    with open(simulator_prior_path, 'rb') as f:
        fz_val_dict = pickle.load(f)

    min_loss = 50
    start_time = time.time()
    for i in range(1, h_['epochs'] + 1):
        s_t = time.time()
        mean_loss, mse_loss, sim_loss = train_epoch_channels(dataloader=dataloader, 
                                         unet=unet, 
                                         decoder=decoder,
                                         diffused_model=diffused_model, 
                                         optimizer=optimizer, 
                                         scheduler=scheduler,
                                         device=device, 
                                         condition=meta['condition'],
                                         fz_val_dict=fz_val_dict,
                                         number_of_repetition=1, 
                                         logger=logger)
        logger.info(f'Epoch: {i}, mean loss: {mean_loss:.4f}, mse loss:{mse_loss:.4f}, simulator loss:{sim_loss:.4f}, lr: {scheduler.get_last_lr()[0]:.6f}')
        if (mean_loss < min_loss):
            min_loss = mean_loss
            torch.save(unet.state_dict(), os.path.join(save_weights_path, 'best_model.pth'))
            logger.info(f'epoch {i} best_model.pth has been saved.')
        if (i % 20 == 0):
            torch.save(unet.state_dict(), os.path.join(save_weights_path, f'model_epoch_{i}.pth'))

        e_t = time.time()
        logger.info(f"Epoch Time Used: {e_t - s_t}s; Total Time Used: {e_t - start_time}s")
