import gc
import os
import sys
import torch
import numpy as np
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
    for step, (input_ids, attention_mask, target) in enumerate(train_loader):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        target = target.to(device)
        batch_size = target.size(0)
        if config.apex:
            with autocast():
                y_preds = model(ids=input_ids, mask=attention_mask)
                loss = criterion(y_preds.view(-1), target)
        else:
            y_preds = model(ids=input_ids, mask=attention_mask)
            loss = criterion(y_preds.view(-1), target)
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

    for step, (input_ids, attention_mask, targets) in enumerate(valid_dataloader):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        targets = targets.to(device)
        batch_size = targets.size(0)
        with torch.no_grad():
            y_preds = model(ids=input_ids, mask=attention_mask)
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