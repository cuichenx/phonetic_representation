import os
import time

import transformers
from torch.nn import MSELoss
from torch.utils.data import DataLoader
import wandb
from dataset import IPATokenPairDataset
from intrinsic_eval import IntrinsicEvaluator
from vocab import *
from model.transformer_cls import CLSPooler
from util import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'
gpuid = os.environ.get('CUDA_VISIBLE_DEVICES', 0)
print('gpu id is', gpuid)
model_save_path = f'./checkpoints/gpu{gpuid}_best_loss.pt'


def train_step(model, train_loader, optimizer, limit_iter_per_epoch=None):
    model.train()
    mse = MSELoss()

    total_loss = 0
    for i, data in enumerate(train_loader):
        tokens_word_1, tokens_word_2 = data['tokens'][0].to(device), data['tokens'][1].to(device)
        feature_array_1, feature_array_2 = data['feature_array'][0].to(device), data['feature_array'][1].to(device)
        # TODO: why do we exclude index -1?
        pooled_word_1 = model(tokens_word_1[:, :-1], feature_array_1)
        pooled_word_2 = model(tokens_word_2[:, :-1], feature_array_2)
        # TODO: may need to reshape

        target = data['feature_edit_dist']

        # TODO: why are pooled_word_1 and pooled_word_2 of different lengths
        model_output = torch.cosine_similarity(pooled_word_1, pooled_word_2)
        loss = mse(model_output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if limit_iter_per_epoch is not None and i >= limit_iter_per_epoch:
            # this is nothing more than a way to log more frequently than every (full) epoch.
            break

    N = len(train_loader)
    return {
        'train/loss': total_loss / N
    }


@torch.no_grad()
def validate_step(model, val_loader, evaluator):
    model.eval()
    # TODO: handle padding for tokens when we batch it
    mse = MSELoss()
    pooled_phon_embs = []

    total_recon, total_loss = 0, 0
    for i, data in enumerate(val_loader):
        tokens_word_1, tokens_word_2 = data['tokens'][0].to(device), data['tokens'][1].to(device)
        feature_array_1, feature_array_2 = data['feature_array'][0].to(device), data['feature_array'][1].to(device)
        # TODO: why do we exclude index -1?
        pooled_word_1 = model(tokens_word_1[:, :-1], feature_array_1)
        pooled_word_2 = model(tokens_word_2[:, :-1], feature_array_2)
        # TODO: may need to reshape

        # iterate through each batch
        for batch in pooled_word_1:
            pooled_phon_embs.append(batch)
        for batch in pooled_word_2:
            pooled_phon_embs.append(batch)

        target = data['feature_edit_dist']
        # TODO: handle batching
        model_output = torch.cosine_similarity(pooled_word_1, pooled_word_2)
        loss = mse(model_output, target)
        total_loss += loss.item()

    # during training, we calculate the MSE between feature edit distance and the cosine similarity of 2 vectors
    # during evaluation, Pearson's/Spearman's correlation coefficient is used instead
    evaluator.set_phon_embs(torch.stack(pooled_phon_embs, 0).detach().cpu().numpy())
    intrinsic_eval = evaluator.run()

    N = len(val_loader)
    return {
        'val/loss': total_loss / N,
        'val/intrinsic_pearson_correlation': intrinsic_eval['pearson'],
        'val/intrinsic_spearman_correlation': intrinsic_eval['spearman'],
    }


def paired_collate_fn(batch):
    feature_array = [(torch.tensor(b['feature_array'][0]), torch.tensor(b['feature_array'][1])) for b in batch]
    tokens = [(torch.tensor(b['tokens'][0]), torch.tensor(b['tokens'][1])) for b in batch]
    feature_array_0 = pad_sequence([pair[0] for pair in feature_array], padding_value=PAD_IDX, batch_first=True)
    feature_array_1 = pad_sequence([pair[1] for pair in feature_array], padding_value=PAD_IDX, batch_first=True)
    tokens_0 = pad_sequence([pair[0] for pair in tokens], padding_value=PAD_IDX, batch_first=True)
    tokens_1 = pad_sequence([pair[1] for pair in tokens], padding_value=PAD_IDX, batch_first=True)

    return {
        'feature_array': (feature_array_0.float(), feature_array_1.float()),
        'tokens': (tokens_0, tokens_1),
        'feature_edit_dist': torch.tensor([b['feature_edit_dist'] for b in batch]).float()
    }


def train(args, vocab, model):
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr, eps=1e-9)
    scheduler = transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_epochs,
                                                             num_training_steps=args.epochs)

    loader_kwargs = {'batch_size': args.batch_size, 'num_workers': 0, 'pin_memory': True, 'collate_fn': paired_collate_fn}
    train_dset = IPATokenPairDataset([f'data/ipa_tokens_{lang}.txt' for lang in args.lang_codes], vocab, split_bounds=(0, 0.999))
    train_loader = DataLoader(train_dset, shuffle=True, **loader_kwargs)
    val_dset = IPATokenPairDataset([f'data/ipa_tokens_{lang}.txt' for lang in args.lang_codes], vocab, split_bounds=(0.999, 1.0))
    val_loader = DataLoader(val_dset, shuffle=False, **loader_kwargs)
    best_val_loss = 1e10
    evaluator = IntrinsicEvaluator()
    # d['ipa'] returns transcriptions for a pair of words
    evaluator.set_phon_feats([transcription for d in val_dset for transcription in d['ipa']])
    best_intrinsic = -1e10

    for ep in range(args.epochs):
        t = time.time()

        train_loss_dict = train_step(model, train_loader, optimizer,
                                     limit_iter_per_epoch=args.limit_iter_per_epoch)
        train_time = time.time()
        val_loss_dict = validate_step(model, val_loader, evaluator)

        best_intrinsic = max(best_intrinsic, val_loss_dict["val/intrinsic_spearman_correlation"])
        val_loss_dict["val/BEST_intrinsic_spearman_correlation"] = best_intrinsic

        wandb.log({"train/lr": optimizer.param_groups[0]['lr'], **train_loss_dict, **val_loss_dict})

        val_loss = val_loss_dict["val/loss"]
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model, optimizer, args, ipa_vocab, ep, model_save_path)

        print(f'< epoch {ep} >  (elapsed: {time.time() - t:.2f}s, decode time: {time.time() - train_time:.2f}s)')
        print(f'  * [train]  loss: {train_loss_dict["train/loss"]:.6f}')
        print(f'  * [ val ]  loss: {val_loss_dict["val/loss"]:.6f}')

        scheduler.step()


def inference(args, model):
    ...


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang_codes', nargs='+')
    parser.add_argument('--vocab_file', type=str)
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--num_layers', help='number of Transformer encoder layers', type=int, default=4)
    parser.add_argument('--num_attention_heads', type=int, default=4)
    parser.add_argument('--hidden_dim', type=int, default=100)
    parser.add_argument('--dropout', type=float, default=0.30)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--limit_iter_per_epoch', type=int, default=200)
    parser.add_argument('--wandb_name', type=str, default="")
    parser.add_argument('--wandb_entity', type=str, default="ka")
    parser.add_argument('--sweeping', type=str2bool, default=False)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    wandb.init(project="phonological-pooling", name=args.wandb_name, entity=args.wandb_entity,
               mode='disabled' if (not args.wandb_name and not args.sweeping) else 'online')
    wandb.run.log_code(".", include_fn=lambda path: path.endswith('.py'))
    wandb.config.update(args)
    os.makedirs("checkpoints", exist_ok=True)

    ipa_vocab = Vocab(tokens_file=args.vocab_file)
    model = CLSPooler(num_layers=args.num_layers,
                      input_dim=24,  # TODO: verify input dim
                      num_heads=args.num_attention_heads,
                      hidden_dim=args.hidden_dim,
                      dropout=args.dropout).to(device)

    train(args, ipa_vocab, model)

    inference(args, model)
