import gc
import os
import sys
import torch
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from IPython.display import display
from torch.cuda.amp import autocast, GradScaler
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.AverageMeter import AverageMeter


def train_fn(train_loader, model, criterion, optimizer, scheduler, config, device, empty_cache=False):
    assert hasattr(config, "gradient_accumulation_steps"), "Please create gradient_accumulation_steps(int default=1) attribute"
    assert hasattr(config, "apex"), "Please create apex(bool default=False) attribute"
    assert hasattr(config, "batch_scheduler"), "Please create batch_scheduler(bool default=False) attribute"

    model.train()
    scaler = GradScaler(enabled=config.apex)
    losses = AverageMeter()
    tk0 = tqdm(train_loader, total=len(train_loader))
    for step, batch in enumerate(tk0):
        inputs = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["target"].to(device)
        batch_size = labels.size(0)
        with torch.cuda.amp.autocast(enabled=config.apex):
            y_preds = model(inputs, mask)
        loss = criterion(y_preds, labels)
        if config.gradient_accumulation_steps > 1:
            loss = loss / config.gradient_accumulation_steps
        losses.update(loss.item(), batch_size)
        if config.apex:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        if (step + 1) % config.gradient_accumulation_steps == 0:
            if config.apex:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
            if config.batch_scheduler:
                scheduler.step()
        del loss
        if empty_cache:
            torch.cuda.empty_cache()
        gc.collect()
    return losses.avg


def valid_fn(model, criterion, valid_loader, config, device):
    assert hasattr(config, "gradient_accumulation_steps"), "Please create gradient_accumulation_steps(int default=1) attribute"
    losses = AverageMeter()
    model.eval()
    preds = []
    tk0 = tqdm(valid_loader, total=len(valid_loader))
    for step, batch in enumerate(tk0):
        inputs = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["target"].to(device)
        batch_size = labels.size(0)
        with torch.no_grad():
            y_preds = model(inputs, mask)
        loss = criterion(y_preds, labels)
        if config.gradient_accumulation_steps > 1:
            loss = loss / config.gradient_accumulation_steps
        losses.update(loss.item(), batch_size)
        preds.append(y_preds.to('cpu').numpy())
    predictions = np.concatenate(preds)
    predictions = np.concatenate(predictions)
    return losses.avg, predictions


def inference_fn(test_loader, model, device, config):
    assert hasattr(config, "n_fold"), "Please create n_fold(int usually 5) attribute"
    assert hasattr(config, "trn_fold"), "Please create trn_fold(list[int] '[0, 1, 2, 3, 4]') attribute"
    assert hasattr(config, "model_name"), "Please create model_name(string, 'roberta-base') attribute"
    assert hasattr(config, "input_dir"), "Please create input_dir(string './') attribute"
    assert hasattr(config, "model_dir"), "Please create model_dir(string './') attribute"
    assert hasattr(config, "target_col"), "Please create target_col(string 'target') attribute"

    def inference(test_loader, model, device):
        preds = []
        model.eval()
        model.to(device)
        tk0 = tqdm(test_loader, total=len(test_loader))
        for (inputs, mask) in tk0:
            inputs = inputs.to(device)
            mask = mask.to(device)
            with torch.no_grad():
                y_preds = model(inputs, mask)
            preds.append(y_preds.sigmoid().to('cpu').numpy())
        predictions = np.concatenate(preds)
        return predictions

    final_pred = []
    for fold in range(config.n_fold):
        if fold in config.trn_fold:
            if config.IN_KAGGLE:
                state = torch.load(config.exp_path + f"{config.model_name}_fold{fold + 1}_best.pth", map_location=torch.device('cpu'))['model']
            else:
                state = torch.load(config.model_dir + f"{config.model_name}_fold{fold + 1}_best.pth", map_location=torch.device('cpu'))['model']
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