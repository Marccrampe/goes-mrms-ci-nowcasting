#!/usr/bin/env python3
"""Portable training with archived model/loss/sampler and pre-crop targets."""
import argparse
import csv
import hashlib
import json
import random
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ci_nowcasting.paths import activate_sources, read_config, configured


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--smoke', action='store_true', help='One epoch, 3 train and 2 validation batches')
    parser.add_argument('--check-only', action='store_true', help='Validate real dataset/cohorts without training')
    args = parser.parse_args()
    cfg, base = read_config(args.config)
    for name in ('dataset_root', 'mrms_input_root', 'normalization_json', 'run_dir'):
        cfg[name] = str(configured(cfg[name], base))
    activate_sources()
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    import train_ci_a_radar_features as original
    from preprocessing.dataset.dataset import TargetDataset
    from models.resunet import build_final_model, build_goes_only_model
    from types import SimpleNamespace

    # Preserve the original cohort checks, including period isolation.
    options = SimpleNamespace(**cfg, max_val_batches=2 if args.smoke else 0)
    cohort = original.cohort_check(options)
    common = dict(dataset_root=cfg['dataset_root'], mrms_input_root=cfg['mrms_input_root'],
                  normalization=cfg['normalization_json'], include_mrms=cfg['include_mrms'], seed=cfg['seed'])
    target = dict(training_kind=cfg['training_target_kind'], extra_pixels=cfg['inflation_pixels'],
                  dy_km=cfg['dy_km'], dx_km=cfg['dx_km'])
    train = TargetDataset(split='train', **common, **target)
    validation = TargetDataset(split='validation', **common, **target)
    if args.check_only:
        for ds in (train, validation):
            item = ds[0]
            assert torch.isfinite(item['inputs']).all()
            print('DATASET CHECK', ds.split, len(ds), tuple(item['inputs'].shape), flush=True)
        print(json.dumps(cohort, indent=2))
        return
    if not torch.cuda.is_available():
        raise RuntimeError('Training requires a CUDA GPU. Use --check-only for CPU data checks.')
    output = Path(cfg['run_dir'])
    if output.exists() and any(output.iterdir()):
        raise RuntimeError('Choose an empty run_dir; existing training is preserved.')
    output.mkdir(parents=True, exist_ok=True)
    random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
    device = torch.device('cuda')
    model = (build_final_model() if cfg['include_mrms'] else build_goes_only_model()).to(device)
    epochs = 1 if args.smoke else cfg['epochs']
    sampler = original.FractionBatchSampler(train.issuances.any_usable_positive.to_numpy(bool),
        cfg['batch_size'], cfg['epoch_samples'], cfg['seed'], str(cfg['train_positive_fraction']))
    loader = DataLoader(train, batch_sampler=sampler, num_workers=cfg['num_workers'], pin_memory=True)
    val_loader = DataLoader(validation, batch_size=cfg['batch_size'], shuffle=False,
                            num_workers=cfg['num_workers'], pin_memory=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    cfg.update(cohort=cohort, initialization='random_from_scratch', precision='float32',
               validation_target_kind='unchanged_original_soft', smoke=bool(args.smoke),
               selection='epoch20 last.pt; best.pt uses original soft validation loss',
               trainer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (output/'config.json').write_text(json.dumps(cfg, indent=2)+'\n')
    best = float('inf'); steps = 0
    for epoch in range(epochs):
        train.set_epoch(epoch); sampler.epoch=epoch; model.train(); losses=[]
        for bi, batch in enumerate(loader):
            if args.smoke and bi >= 3: break
            x=original.make_inputs(batch, device, cfg['include_mrms'])
            optimizer.zero_grad(set_to_none=True)
            loss, _, _=original.ci_loss(model(x), batch['targets'].to(device),
                batch['loss_masks'].to(device), cfg['positive_weight'], cfg['dice_weight'], cfg['negative_weight'])
            if not torch.isfinite(loss): raise RuntimeError('Non-finite loss')
            loss.backward()
            if not torch.isfinite(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)):
                raise RuntimeError('Non-finite gradient')
            optimizer.step(); steps+=1; losses.append(float(loss.detach()))
            if bi % 50 == 0: print('TRAIN',epoch+1,bi,float(loss.detach()),flush=True)
        if not losses: raise RuntimeError('Empty training epoch')
        metrics=original.evaluate(model,val_loader,device,options)
        row=dict(epoch=epoch+1,train_loss=float(np.mean(losses)),optimizer_steps=steps,**metrics)
        with (output/'metrics.csv').open('a',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(row))
            if epoch==0:writer.writeheader()
            writer.writerow(row)
        scheduler.step()
        improved=metrics['validation_loss']<best;best=min(best,metrics['validation_loss'])
        payload=dict(model_state=model.state_dict(),optimizer_state=optimizer.state_dict(),
            scheduler_state=scheduler.state_dict(),config=cfg,epoch=epoch+1,metrics=row)
        original.atomic_save(output/'last.pt',payload)
        if improved:original.atomic_save(output/'best.pt',payload)
        print('EPOCH COMPLETE',json.dumps(row),flush=True)
    (output/'completed.json').write_text(json.dumps(dict(status='smoke_completed' if args.smoke else 'completed',
        epochs=epochs,optimizer_steps=steps),indent=2)+'\n')


if __name__=='__main__':main()
