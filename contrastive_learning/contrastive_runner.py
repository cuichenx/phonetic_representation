from tqdm import tqdm
import torch
from torch import nn
import numpy as np

class ContrastiveRunner:
    def __init__(self, model, criterion, optim, train_loader, val_loader, evaluator, n_epochs, accum_iter, eval_every=5):
        self.model = model
        self.criterion = criterion
        self.optim = optim
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.evaluator = evaluator
        self.n_epochs = n_epochs
        # gradient accumulation to simulate larger batch size
        self.accum_iter = accum_iter
        self.eval_every = eval_every
        

        #TODO: add in wandb code

    def train_step(self):
        self.model.train()
        self.model.proj_head_on()
        losses = []
        for batch_idx, (ipa, feats) in enumerate(tqdm(self.train_loader)):
            with torch.set_grad_enabled(True):
                # rely on dropout as a form of data augmentation
                z1 = self.model(feats)
                z2 = self.model(feats)

                # compute ntxent loss
                loss = self.criterion(z1, z2)
                loss /= self.accum_iter
                loss.backward()

                # gradient accumulation
                if ((batch_idx + 1) % self.accum_iter == 0) or (batch_idx + 1 == len(self.train_loader)):
                        self.optim.step()
                        self.optim.zero_grad()

                losses.append(self.accum_iter * loss.cpu().detach())
        
        return losses

    def val_step(self):
        self.model.eval()
        self.model.proj_head_on()
        losses = []
        for (ipa, feats) in tqdm(self.val_loader):
            z1 = self.model(feats)
            z2 = self.model(feats)

            loss = self.criterion(z1, z2)
            losses.append(loss.cpu().detach())
    
        return losses


    def __call__(self):
        for epoch in range(self.n_epochs):
            train_losses = self.train_step()
            val_losses = self.val_step()

            print(f"Epoch {epoch+1}")
            print(
                f"Train loss {np.average(train_losses):8.5f}",
                f"Dev loss {np.average(val_losses):8.5f}",
            )

            if (epoch+1) % self.eval_every == 0:
                self.model.proj_head_off()
                dev_pearson, dev_pearson = self.evaluator(self.model, self.val_loader, key="dev")

                print(
                    f"Dev pearson    {dev_pearson:6.2%}",
                    f"Dev spearman   {dev_pearson:6.2%}",
                )
            
        