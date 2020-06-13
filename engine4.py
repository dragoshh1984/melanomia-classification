
import torch
from tqdm import tqdm
from average_meter import AverageMeter

class Engine:
    @staticmethod
    def train(
        data_loader,
        model,
        optimizer,
        device,
        scheduler=None,
        accumulation_steps=1,
    ):
        losses = AverageMeter()
        predictions = []
        model.train()
        if accumulation_steps > 1:
            optimizer.zero_grad()
        tk0 = tqdm(data_loader, total=len(data_loader), disable=False)
        # import pdb; pdb.set_trace()
        for b_idx, data in enumerate(tk0):
            for key, value in data.items():
                data[key] = value.to(device)
            
            if accumulation_steps == 1 and b_idx == 0:
                optimizer.zero_grad()
            
            _, loss = model(**data)

            with torch.set_grad_enabled(True):
                loss.backward()
                optimizer.step()

                if scheduler is not None:
                    scheduler.step()
                if b_idx > 0:
                    optimizer.zero_grad()
            
            losses.update(loss.item(), data_loader.batch_size)
            tk0.set_postfix(loss=losses.avg)
        
        return losses.avg

    @staticmethod
    def evaluate(data_loader, model, device):
        losses = AverageMeter()
        final_predictions = []
        model.eval()

        with torch.no_grad():
            tk0 = tqdm(data_loader, total=len(data_loader), disable=False)
            for b_idx, data in enumerate(tk0):
                for key, value in data.items():
                    data[key] = value.to(device)
                
                predictions, loss = model(**data)
                # import pdb; pdb.set_trace()
                predictions = predictions.cpu()
                losses.update(loss.item(), data_loader.batch_size)
                final_predictions.append(predictions)
                tk0.set_postfix(loss=losses.avg)
        
        return final_predictions, losses.avg

    @staticmethod
    def predict(data_loader, model, device):
        model.eval()
        final_predictions = []

        with torch.no_grad():
            tk0 = tqdm(data_loader, total=len(data_loader), disable=False)
            for b_idx, data in enumerate(tk0):
                for key, value in data.items():
                    data[key] = value.to(device)
                
                predictions, _ = model(**data)
                predictions = predictions.cpu()
                final_predictions.append(predictions)
        
        return final_predictions
