import os
import gc
import sys
import torch
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from IPython.display import display
from torch.cuda.amp import autocast, GradScaler
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.AverageMeter import AverageMeter


def train_fn(train_loader, model, criterion, optimizer, config, device):
    assert hasattr(config, "apex"), "Please create apex(bool) attribute"
    assert hasattr(config, "gradient_accumulation_steps"), "Please create gradient_accumulation_steps(int default=1) attribute"

    model.train()
    if config.apex:
        scaler = GradScaler()
    losses = AverageMeter()
    for step, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)
        if config.apex:
            with autocast():
                y_preds = model(images)
                loss = criterion(y_preds.view(-1), labels)
        else:
            y_preds = model(images)
            loss = criterion(y_preds.view(-1), labels)
        # record loss
        losses.update(loss.item(), batch_size)
        if config.gradient_accumulation_steps > 1:
            loss = loss / config.gradient_accumulation_steps
        if config.apex:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % config.gradient_accumulation_steps == 0:
            if config.apex:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
    return losses.avg


def valid_fn(model, criterion, valid_dataloader, config, device):
    assert hasattr(config, "gradient_accumulation_steps"), "Please create gradient_accumulation_steps(int default=1) attribute"
    model.eval()
    losses = AverageMeter()
    preds = []
    all_targets = []

    for step, (images, targets) in enumerate(valid_dataloader):
        images = images.to(device)
        targets = targets.to(device)
        batch_size = targets.size(0)
        with torch.no_grad():
            y_preds = model(images)
        loss = criterion(y_preds.view(-1), targets)
        losses.update(loss.item(), batch_size)
        preds.append(y_preds.to('cpu').numpy())
        all_targets.append(targets.detach().cpu().numpy())
        if config.gradient_accumulation_steps > 1:
            loss = loss / config.gradient_accumulation_steps
        del loss
        gc.collect()
    predictions = np.concatenate(preds)
    all_labels = np.concatenate(all_targets)
    return losses.avg, predictions, all_labels


def inference_fn(test_loader, model, device, config):
    assert hasattr(config, "n_fold"), "Please create n_fold(int usually 5) attribute"
    assert hasattr(config, "trn_fold"), "Please create trn_fold(list[int] '[0, 1, 2, 3, 4]') attribute"
    assert hasattr(config, "model_dir"), "Please create model_dir(string './') attribute"
    assert hasattr(config, "model_name"), "Please create model_name(string, 'resnet34d') attribute"
    assert hasattr(config, "input_dir"), "Please create input_dir(string './') attribute"
    assert hasattr(config, "target_col"), "Please create target_col(string 'target') attribute"

    def inference(test_loader, model, device):
        model.eval()
        preds = []
        tk0 = tqdm(enumerate(test_loader), total=len(test_loader))
        for step, (images) in tk0:
            images = images.to(device)
            with torch.no_grad():
                pred = model(images)
            preds.append(pred.view(-1).cpu().detach().numpy())
        preds = np.concatenate(preds)
        return preds

    final_pred = []
    for fold in range(config.n_fold):
        if fold in config.trn_fold:
            state = torch.load(config.model_dir + f"/{config.model_name}_fold{fold + 1}_best.pth", map_location=torch.device('cpu'))['model']
            model.load_state_dict(state)
            model.to(device)
            preds = inference(test_loader, model, device)
            final_pred.append(preds)
            del state
            gc.collect()
            torch.cuda.empty_cache()

    final_pred = np.mean(np.column_stack(final_pred), axis=1)
    sub_df = pd.read_csv(config.input_dir + "sample_submission.csv")
    sub_df[config.target_col] = final_pred
    sub_df.to_csv(config.model_dir + "submission.csv", index=False)
    display(sub_df.head())