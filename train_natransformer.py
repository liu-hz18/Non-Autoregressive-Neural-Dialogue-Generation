import os
import re
import math
import time
from tqdm import tqdm
import torch
from torch import nn
import torch.optim as optim
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from nag.model import NAGModel, NATransformer
from nag.modules import TransformerNonAutoRegressive, TransformerContinuous
from nag.logger import LogManager, SummaryHelper
from nag.metric import BLEUMetric, DistinctNGram
from nag.vocab_helper import VocabBulider
from nag.utils import PadCollate, get_index, restore_best_state, init_seed
from nag.dataset import OpenSubDataset, IMSDBDataset
from nag.optimizer import RAdam
from nag.options import parse_args
from nag.criterion import similarity_regularization, LabelSmoothedCrossEntropyLoss


def train(epoch, model, dataloader, criterionM, criterionL, optimizer, scheduler):
    global global_train_step
    model.train()
    total_loss = 0.
    bleu_score = 0.
    distinct_1_score, distinct_2_score = 0., 0.
    for i, (src, tgt, src_lens, tgt_lens) in tqdm(enumerate(dataloader, 0), desc='train', total=len(opensub_dataset)//opt.realbatch):
        delta_length_gold = torch.clamp(
            tgt_lens - src_lens + opt.delta, min=1, max=opt.delta*2)
        output_probs, delta_length_probs = model(src, src_lengths=src_lens, tgt_lengths=tgt_lens)
        transposed_output_probs = output_probs.permute(0, 2, 1)
        out_seqs = torch.argmax(output_probs, dim=2)
        out_lens = torch.argmax(delta_length_probs, dim=-1) + tgt_lens - opt.delta
        # loss
        loss_sentence = criterionM(transposed_output_probs, tgt)
        loss_length = criterionL(delta_length_probs, delta_length_gold)
        # loss = (loss_sentence + loss_length * opt.lengthratio + similarity_regularization(transposed_output, hidden)) / ACCUMULATION
        loss = (loss_sentence + loss_length * opt.lengthratio) / ACCUMULATION
        loss.backward()
        total_loss += loss.item()
        # calculate metrics
        bleu_score += bleu_metirc(tgt, out_seqs, out_lens)
        distinct_1_score += distinct_1(out_seqs, out_lens)
        distinct_2_score += distinct_2(out_seqs, out_lens)
        # summary writer
        global_train_step += 1
        writer.log_loss(loss.item()*ACCUMULATION, mode='train')
        if (i+1) % ACCUMULATION == 0:
            # clip_grad_norm_(model.parameters(), max_norm=5)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
        if (i+1) % opt.logstep == 0:
            avg_loss = (total_loss / opt.logstep) * ACCUMULATION
            avg_bleu = bleu_score / opt.logstep
            avg_length = torch.mean(out_lens.float())
            avg_distinct_1 = distinct_1_score / opt.logstep
            avg_distinct_2 = distinct_2_score / opt.logstep
            mylogger.log(
                i, epoch, model, value=avg_loss, is_train=True,
                info=f'loss: {avg_loss:.4f} | ppl: {math.exp(avg_loss):.4f} | BLEU: {avg_bleu:.5f} | avg_length: {avg_length:.1f} | d1: {avg_distinct_1:.3f} | d2: {avg_distinct_2:.3f}')
            total_loss = 0.
            bleu_score = 0.
            distinct_1_score, distinct_2_score = 0., 0.
            show_gen_seq(src[:2], out_seqs[:2], out_lens[:2], tgt[:2], vocab_bulider, global_train_step, mode='train')


def eval(epoch, model, dataloader):
    global global_valid_step
    model.eval()
    total_loss = 0.
    bleu_score = 0.
    distinct_1_score, distinct_2_score = 0., 0.
    fout = open(os.path.join('./save/' + model_name + '/', model_name + '_' + str(epoch)), 'w', encoding='utf-8')
    with torch.no_grad():
        for i, (src, tgt, src_lens, tgt_lens) in tqdm(enumerate(dataloader, 0), desc='eval', total=len(imsdb_dataset)//opt.realbatch):
            delta_length_gold = torch.clamp(
                tgt_lens - src_lens + opt.delta, min=1, max=opt.delta*2)
            output_probs, delta_length_probs = model(src, src_lengths=src_lens)
            transposed_output_probs = output_probs.permute(0, 2, 1)
            # loss
            min_len = min(transposed_output_probs.shape[2], tgt.shape[1])
            loss_sentence = criterionM(
                transposed_output_probs[:, :, :min_len], tgt[:, :min_len])
            loss_length = criterionL(delta_length_probs, delta_length_gold)
            loss = (loss_sentence + loss_length * opt.lengthratio)
            total_loss += loss.item()
            # calculate metrics
            out_seqs = torch.argmax(output_probs, dim=2)
            out_lens = torch.argmax(delta_length_probs, dim=-1) + tgt_lens - opt.delta
            bleu_score += bleu_metirc(tgt, out_seqs, out_lens)
            distinct_1_score += distinct_1(out_seqs, out_lens)
            distinct_2_score += distinct_2(out_seqs, out_lens)
            # show sequence
            global_valid_step += 1
            for out_seq in out_seqs:
                fout.write(' '.join(convert_ids_to_seq(out_seq, vocab_bulider)) + '\n')
            if (i+1) % opt.logstep == 0:
                show_gen_seq(
                    src[:2], out_seqs[:2], out_lens[:2], tgt[:2], vocab_bulider, global_valid_step, mode='valid')
        # summary
        avg_loss = total_loss / i
        avg_bleu = bleu_score / i
        avg_length = torch.mean(out_lens.float())
        avg_distinct_1 = distinct_1_score / i
        avg_distinct_2 = distinct_2_score / i
        writer.log_loss(avg_loss, mode='valid')
        mylogger.log(
            i, epoch, model, value=avg_bleu, is_train=False,
            info=f'loss: {avg_loss:.4f} | ppl: {math.exp(avg_loss):.4f} | BLEU: {avg_bleu:.5f} | avg_length: {avg_length:.1f} | d1: {avg_distinct_1:.3f} | d2: {avg_distinct_2:.3f}')


def run_model(model, train_loader, eval_loader, niter, criterionM, criterionL, optimizer, scheduler):
    mylogger.log_info('Running Model')
    for i in range(niter):
        mylogger.log_info(f'EPOCH: {i}, lr: {optimizer.state_dict()["param_groups"][0]["lr"]}')
        train(i, model, train_loader, criterionM, criterionL, optimizer, scheduler)
        eval(i, model, eval_loader)


def convert_ids_to_seq(id_seq, vocab_bulider):
    word_list = [vocab_bulider.id_to_word(idx) for idx in id_seq]
    return re.sub(r'(<unk> )+', r'<unk> ', ' '.join(word_list)).split()


def show_gen_seq(batch_in_seqs, batch_out_seqs, batch_out_lens, groud_truth, vocab_bulider, step, mode='train'):
    for in_id, out_id, out_len, gold_id in zip(batch_in_seqs, batch_out_seqs, batch_out_lens, groud_truth):
        in_seq = convert_ids_to_seq(in_id, vocab_bulider)
        out_seq = convert_ids_to_seq(out_id[:out_len] if out_len > 0 else out_id, vocab_bulider)
        gold_seq = convert_ids_to_seq(gold_id, vocab_bulider)
        writer.add_text(tag=mode + '_post', sentence=' '.join(in_seq[:get_index(in_seq, '<pad>')]), global_step=step)
        writer.add_text(tag=mode + '_pred', sentence=' '.join(out_seq), global_step=step)
        writer.add_text(tag=mode + '_reps', sentence=' '.join(gold_seq[:get_index(in_seq, '<pad>')]), global_step=step)


if __name__ == '__main__':
    begin_time = time.strftime("%H%M%S", time.localtime())
    model_name = 'natransformer' + begin_time
    opt = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.cuda.set_device(opt.gpuid)
    init_seed(opt.manualSeed)
    ACCUMULATION = opt.batchsize // opt.realbatch

    mylogger = LogManager(checkpoint_step=10,
                          save_dir='./save',
                          model_name=model_name,
                          log_file_name=model_name + '.log',
                          mode='min', device=device)
    mylogger.save_args(opt)
    writer = SummaryHelper(save_dir='./save', model_name=model_name)

    vocab_file_list = ['dialogue_length3_6.post']
    vocab_bulider = VocabBulider(
        './data/opensubtitles', src_files=vocab_file_list, ignore_unk_error=True,
        vocab_file='vocab.txt', min_count=opt.mincount, update=opt.update)
    print('most common 50:', vocab_bulider.most_common(50))
    mylogger.log_info('vocab size: %d' % len(vocab_bulider))

    # metircs
    bleu_metirc = BLEUMetric(vocab_bulider.id2vocab, ignore_smoothing_error=True)
    distinct_1 = DistinctNGram(ngram=1)
    distinct_2 = DistinctNGram(ngram=2)

    # dataset and dataloader
    if opt.cotk:  # use dataset in paper 'Non-Autoregressive Neural Dialogue Generation'
        opensub_file_name_list = ['opensub_pair_dev', 'opensub_pair_test', 'opensub_pair_train']
        opensub_dataset = OpenSubDataset(
            data_dir='./data/opensubtitles', vocab_bulider=vocab_bulider,
            file_name_list=opensub_file_name_list, unk_token=None,
            save_process=False, samples=opt.trainsamples, add_bos=False, add_eos=False)
    else:  # use dataset in paper 'cotk'
        opensub_file_name_list = ['dialogue_length3_6']
        opensub_dataset = OpenSubDataset(
            data_dir='./data/opensubtitles', vocab_bulider=vocab_bulider,
            file_name_list=opensub_file_name_list, unk_token='UNknown',
            save_process=False, samples=opt.trainsamples, add_bos=False, add_eos=False)
    print(opensub_dataset.sample())
    opensub_dataloader = DataLoader(
        opensub_dataset, batch_size=opt.realbatch,
        collate_fn=PadCollate(dim=0, pad_id=vocab_bulider.padid, device=device),
        shuffle=True, num_workers=opt.workers, drop_last=True)

    imsdb_file_name_list = ['imsdb_lower']
    imsdb_dataset = IMSDBDataset(
        data_dir='./data/imsdb', vocab_bulider=vocab_bulider,
        file_name_list=imsdb_file_name_list, save_process=False,
        samples=opt.validsamples, add_bos=False, add_eos=False)
    print(imsdb_dataset.sample())
    imsdb_dataloader = DataLoader(
        imsdb_dataset, batch_size=opt.realbatch,
        collate_fn=PadCollate(dim=0, pad_id=vocab_bulider.padid, device=device),
        shuffle=True, num_workers=opt.workers, drop_last=True)
    '''
    # teacher model
    saved_model = opt.energy
    energy = TransformerContinuous(
        ntoken=len(vocab_bulider)+2, d_model=opt.embedsize, nhead=opt.nhead,
        num_encoder_layers=opt.encoderlayer, num_decoder_layers=opt.decoderlayer,
        dim_feedforward=opt.feedforward, postnorm=True, dropout=opt.dropout, gumbels=opt.gumbels,
        use_src_mask=False, use_tgt_mask=True, use_memory_mask=False,
        activation='relu', use_vocab_attn=False, use_pos_attn=False,
        relative_clip=0, highway=False, device=device, max_sent_length=32,
        share_input_output_embedding=False, share_encoder_decoder_embedding=True,
        share_vocab_embedding=True, fix_pos_encoding=opt.fix,
        bos_token=2, tgt_operator='STL', dis_operator='SX').to(device)
    energy.load_state_dict(torch.load("./save/" + saved_model + "/" + saved_model + "_20.ckpt"))
    energy.show_graph()
    '''

    # model definition
    if opt.mine:
        model = TransformerNonAutoRegressive(
            ntoken=len(vocab_bulider), d_model=opt.embedsize, nhead=opt.nhead,
            num_encoder_layers=opt.encoderlayer, num_decoder_layers=opt.decoderlayer,
            dim_feedforward=opt.feedforward, postnorm=True, dropout=opt.dropout,
            gumbels=opt.gumbels, use_src_mask=False, use_tgt_mask=False, use_memory_mask=False,
            activation='relu', use_vocab_attn=opt.vocabattn, use_pos_attn=opt.posattn,
            relative_clip=4, highway=False, device=device, max_sent_length=64,
            share_input_output_embedding=False, share_encoder_decoder_embedding=True,
            share_vocab_embedding=False, fix_pos_encoding=True,
            min_length_change=-20, max_length_change=20,
            use_src_to_tgt=True).to(device)
        model.show_graph()
    else:
        model = NATransformer(
            vocab_size=len(vocab_bulider), embed_size=opt.embedsize,
            n_encoder_layers=opt.encoderlayer, n_decoder_layers=opt.decoderlayer,
            dim_feedforward=opt.feedforward, nhead=opt.nhead,
            dropout=opt.dropout, gumbels=opt.gumbels, device=device,
            fix_pos_encoding=opt.fix).to(device)
    if opt.half:
        model = model.half()
    if opt.ft:
        model = restore_best_state(model, opt.ckpt, save_dir='./save', device=model.device)

    # optimizer and scheduler
    if opt.warmup:
        # optim.Adam()
        optimizer = RAdam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=1., betas=(opt.beta1, opt.beta2), eps=opt.eps)
        rate_ratio = 1. / math.sqrt(opt.embedsize)
        # top_lr = 1 / sqrt(d_model * warmup_step) at step == warmup_step
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: rate_ratio * min(1. / math.sqrt(step+1), step*(opt.warmup_step**(-1.5))))
    else:
        # optim.Adam()
        optimizer = RAdam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=opt.lr, betas=(opt.beta1, opt.beta2), eps=opt.eps, weight_decay=opt.weight_decay)
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=opt.schedulerstep, gamma=opt.gamma)
    # loss function
    criterionM = LabelSmoothedCrossEntropyLoss(eps=0.1, ignore_index=vocab_bulider.padid)  # for Transformer
    criterionL = nn.CrossEntropyLoss()  # for length-predictor

    # run model
    global_train_step, global_valid_step = 0, 0
    run_model(
        model, opensub_dataloader, imsdb_dataloader,
        opt.niter, criterionM, criterionL, optimizer, scheduler)
    writer.close()
