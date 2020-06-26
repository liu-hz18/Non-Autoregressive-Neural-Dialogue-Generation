import argparse
import random
import torch
from torch import nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from nag.model import NAGModel
from nag.logger import LogManager
from nag.metric import BLEUMetric
from nag.vocab_helper import VocabBulider
from nag.utils import PadCollate
from nag.dataset import OpenSubDataset, IMSDBDataset


def parse_args():
    parser = argparse.ArgumentParser()
    # basic config
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
    parser.add_argument('--half', action='store_true', help='half precision floating point')
    parser.add_argument('--batchsize', type=int, default=32, help='input batch size')
    parser.add_argument('--logstep', type=int, default=50, help='log interval')
    parser.add_argument('--schedulerstep', type=int, default=10, help='step size for scheduler')
    parser.add_argument('--niter', type=int, default=100, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
    parser.add_argument('--gamma', type=float, default=0.95, help='decay coefficient of learning rate')
    parser.add_argument('--nocuda', action='store_false', help='disables cuda')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--outdir', default='.', help='folder to output images and model checkpoints')
    parser.add_argument('--manualSeed', type=int, default=random.randint(1, 10000), help='manual seed')
    parser.add_argument('--lengthratio', type=float, default=0.1, help='ratio of length loss')
    # vocab builder
    parser.add_argument('--mincount', type=int, default=5, help='min count of vocab')
    parser.add_argument('--update', action='store_true', help='min count of vocab')
    # Adam
    parser.add_argument('--beta1', type=float, default=0.90, help='hyperparameter \'beta1\' for Adam')
    parser.add_argument('--beta2', type=float, default=0.98, help='hyperparameter \'beta2\' for Adam')
    parser.add_argument('--eps', type=float, default=1e-8, help='hyperparameter \'eps\' for Adam')
    # Model Hyperparameters
    parser.add_argument('--gumbels', action='store_true', help='use gumbels sotfmax')
    parser.add_argument('--dropout', type=float, default=0.2, help='half precision floating point')
    parser.add_argument('--nhead', type=int, default=16, help='number of head in MultiheadAttention()')
    parser.add_argument('--embedsize', type=int, default=256, help='embedding size of nn.Embedding()')  # 1024
    parser.add_argument('--encoderlayer', type=int, default=1, help='number of encoder layers')
    parser.add_argument('--decoderlayer', type=int, default=1, help='number of decoder layers')
    parser.add_argument('--feedforward', type=int, default=1024, help='dimension of Feedforward Net')  # 4096
    parser.add_argument('--nolayernorm', action='store_false', help='disables layernorm')
    parser.add_argument('--delta', type=int, default=20, help='delta length between \'src\' and \'tgt\'')
    opt = parser.parse_args()
    print(opt)
    return opt


def init_seed(manual_seed):
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(manual_seed)


def train(epoch, model, dataloader, criterionM, criterionL, optimizer):
    model.train()
    total_loss = 0.
    total_loss_sentence = 0.
    bleu_score = 0.
    for i, (src, tgt, src_lens, tgt_lens) in enumerate(dataloader, 0):
        optimizer.zero_grad()
        delta_length_gold = tgt_lens - src_lens + opt.delta
        tgt_max_len = max(tgt_lens)
        output, delta_length = model(src, tgt_length=tgt_max_len)
        out_seqs = torch.argmax(output.permute(0, 2, 1), dim=2)
        loss_sentence = criterionM(output, tgt)
        loss_length = criterionL(delta_length, delta_length_gold)
        loss = loss_sentence + loss_length * opt.lengthratio
        total_loss += loss.item()
        total_loss_sentence += loss_sentence.item()
        bleu_score += bleu_metirc(tgt, out_seqs)
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=5)
        optimizer.step()
        if i % opt.logstep == 0:
            avg_loss = total_loss / opt.logstep
            avg_bleu = bleu_score / opt.logstep
            avg_sentence_loss = total_loss_sentence / opt.logstep
            avg_length = torch.mean(torch.argmax(delta_length, dim=-1).float()) + tgt_max_len - opt.delta
            mylogger.log(i, epoch, model, value=avg_loss, info=f'loss_sentence: {avg_sentence_loss:.4f} | BLEU: {avg_bleu:.6f} | avg_length: {avg_length:.1f}', is_train=True)
            total_loss = 0.
            bleu_score = 0.
            total_loss_sentence = 0.


def eval(epoch, model, dataloader):
    model.eval()
    bleu_score = 0.
    with torch.no_grad():
        for i, (src, tgt, src_lens, tgt_lens) in enumerate(dataloader, 0):
            output, delta_length = model(src, tgt_length=-1)
            out_seqs = torch.argmax(output.permute(0, 2, 1), dim=2)
            bleu_score += bleu_metirc(tgt, out_seqs)
            if i % opt.logstep == 0:
                avg_bleu = bleu_score / opt.logstep
                avg_length = torch.mean(torch.argmax(delta_length, dim=-1).float()) + max(tgt_lens) - 20
                mylogger.log(i, epoch, model, value=0, info=f'BLEU: {avg_bleu:.6f} | avg_length: {avg_length:.1f}', is_train=False)
                bleu_score = 0.


def run_model(model, train_loader, eval_loader, niter, criterionM, criterionL, optimizer, scheduler):
    print('Running Model')
    for i in range(niter):
        print('EPOCH: %d' % i)
        train(i, model, train_loader, criterionM, criterionL, optimizer)
        eval(i, model, eval_loader)
        scheduler.step()


if __name__ == '__main__':
    # TODO: dataset, dataloader, optimizer, train&eval
    device = "cuda" if torch.cuda.is_available() else "cpu"
    opt = parse_args()
    init_seed(opt.manualSeed)

    mylogger = LogManager(checkpoint_step=5,
                          save_dir='./save',
                          model_name='nag',
                          log_file_name='nag_log',
                          mode='min', device=device)

    src_file_list = ['opensub_pair_dev.post', 'opensub_pair_dev.response',
                     'opensub_pair_test.post', 'opensub_pair_test.response',
                     'opensub_pair_train.post', 'opensub_pair_train.response'
                    ]
    vocab_bulider = VocabBulider('./data/opensubtitles', src_file_list,
                                 ignore_unk_error=True, vocab_file='vocab.txt',
                                 min_count=opt.mincount, update=opt.update)
    print('most common 50:', vocab_bulider.most_common(50))
    print('vocab size: %d' % len(vocab_bulider))

    opensub_dataset = OpenSubDataset('./data/opensubtitles', vocab_bulider=vocab_bulider)
    opensub_dataloader = DataLoader(opensub_dataset, batch_size=opt.batchsize,
                                    collate_fn=PadCollate(dim=0, pad_id=vocab_bulider.padid, device=device),
                                    shuffle=True, num_workers=opt.workers)

    imsdb_dataset = IMSDBDataset('./data/imsdb', vocab_bulider=vocab_bulider)
    imsdb_dataloader = DataLoader(imsdb_dataset, batch_size=opt.batchsize,
                                  collate_fn=PadCollate(dim=0, pad_id=vocab_bulider.padid, device=device),
                                  shuffle=False, num_workers=opt.workers)

    bleu_metirc = BLEUMetric(vocab_bulider.id2vocab, ignore_smoothing_error=True)

    model = NAGModel(vocab_size=len(vocab_bulider), embed_size=opt.embedsize, dim_feedforward=opt.feedforward,
                     n_decoder_layers=opt.decoderlayer, n_encoder_layers=opt.encoderlayer, rezero=not opt.nolayernorm,
                     nhead=opt.nhead, gumbels=opt.gumbels, device=device, dropout=opt.dropout,
                     pos_embedding_perlayer=False).to(device)
    if opt.half:
        model = model.half()
    model.sample(niter=5, seq_length=50)

    optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2), eps=opt.eps)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=opt.schedulerstep, gamma=opt.gamma)
    criterionM = nn.CrossEntropyLoss()  # for Transformer
    criterionL = nn.CrossEntropyLoss()  # for length-predictor

    run_model(model, opensub_dataloader, imsdb_dataloader, opt.niter, criterionM, criterionL, optimizer, scheduler)
