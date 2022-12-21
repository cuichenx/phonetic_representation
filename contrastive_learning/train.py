import argparse
from tqdm import tqdm

import panphon2
import numpy as np

import torch
from torchlars import LARS
from torch.utils.data import DataLoader

from models import LSTM_Encoder
from ntxent_loss import NTXent_Loss
from contrastive_runner import ContrastiveRunner
from evaluators import IntrinsicEvaluator

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

"""
python3 train.py --data-file ipa_tokens_es.txt \
                 --n-epochs 5
"""
def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("--data-file",
                      type=str,
                      help="path to training file of ipa strings")
    args.add_argument("--n-thousand",
                      type=int,
                      default=99,
                      help="number of lines (in thousands) to use during training")
    args.add_argument("--model-outfile",
                      type=str,
                      default="",
                      help="file to save model to")
    args.add_argument("--embs-outfile",
                      type=str,
                      default="",
                      help="file to save embeddings to")
    args.add_argument("--n-epochs",
                      type=int,
                      default=20,
                      help="number of epochs to train for")
    args.add_argument("--train-batch-size",
                      type=int,
                      default=256,
                      help="training batch size")
    args.add_argument("--val-batch-size",
                      type=int,
                      default=256,
                      help="validation batch size")
    args.add_argument("--accum-iter",
                      type=int,
                      default=4,
                      help="for gradient accumulationm, true batch-size = accum_iter * train-batch-size")
    args.add_argument("--hidden-size",
                      type=int,
                      default=256,
                      help="model hidden dimension")
    args.add_argument("--num_layers",
                      type=int,
                      default=4,
                      help="number of layers in LSTM encoder")
    args.add_argument("--dropout",
                      type=int,
                      default=0.1,
                      help="dropout probability for LSTM encoder")
    args.add_argument("--temp",
                      type=float,
                      default=0.1,
                      help="temperature for NTXent loss")
    args.add_argument("--checkpoint-file",
                      type=str,
                      default="",
                      help="resume model checkpoint file")
    return args.parse_args()
    
def load_data(args):
    with open(args.data_file, "r") as f:
        data = [x.rstrip("\n") for x in f.readlines()][:1000+args.n_thousand*1000]
    
    print(f"Loaded {len(data)//1000}k words")

    ft = panphon2.FeatureTable()
    data = [(w, ft.word_to_binary_vectors(w)) for w in tqdm(data)]
    np.random.shuffle(data)

    data_val = data[:1000]
    data_train = data[1000:]

    return data_train, data_val

def main():
    args = parse_args()

    # encoder & loss fn
    encoder = LSTM_Encoder(args.hidden_size, 
                            args.num_layers, 
                            args.dropout, 
                            DEVICE)
    criterion = NTXent_Loss(args.train_batch_size, args.temp)
    
    # TODO: add in where you got this from
    lr = 0.3 * (4096 / 256)
    # optimizer
    base_optimizer = torch.optim.SGD(encoder.parameters(), lr=lr)
    optimizer = LARS(optimizer=base_optimizer, eps=1e-8, trust_coef=0.001)
    

    # dataloaders
    data_train, data_val = load_data(args)
    train_loader = DataLoader(dataset=data_train, 
                              batch_size=args.train_batch_size, 
                              shuffle=True, 
                              collate_fn=lambda x: ([y[0] for y in x], [y[1] for y in x]), 
                              drop_last=True) # drop last to avoid issues with NTXent loss
    val_loader = DataLoader(dataset=data_val,
                            batch_size=args.val_batch_size,
                            shuffle=True, 
                            collate_fn=lambda x: ([y[0] for y in x], [y[1] for y in x]), 
                            drop_last=True)

    # evaluator
    evaluator = IntrinsicEvaluator()

    runner = ContrastiveRunner(model=encoder,
                               criterion=criterion,
                               optim=optimizer,
                               train_loader=train_loader,
                               val_loader=val_loader,
                               evaluator=evaluator,
                               n_epochs=args.n_epochs,
                               accum_iter=args.accum_iter)
    runner()


if __name__ == "__main__":
    main()