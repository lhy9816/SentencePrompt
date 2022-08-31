"""
utils function to train the model
Arther: Hangyu Li
Date: 12-05-2022
"""
import csv
import torch

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from datasets import load_dataset
from sklearn.manifold import TSNE
from transformers import DataCollatorWithPadding

from models.bert_gap_model import BertGapModel
from models.roberta_gap_model import RobertaGapModel


def get_plm_model(plm_name):
    """
    Get pretrained language model according to the plm_name
    [bert, roberta]
    """
    if plm_name == 'bert' or '-bert' in plm_name:
        return BertGapModel
    elif plm_name == 'roberta' or '-roberta' in plm_name:
        return RobertaGapModel
    else:
        raise NotImplementedError("The target pretrained language model is not supported!")


def freeze_plm_params(model, plm_name):
    """
    Freeze all parameters except for the params for the sentence embedding
    """
    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Relax gap sentence embedding parameters (excluding the reparameterization layer)
    for param in model.sentence_embedding.parameters():
        param.requires_grad = True


def add_sentences_id_to_dataset(raw_datasets):
    """
    Add sentences_ids columns to the dataset.
    """
    if "train" in raw_datasets.keys():
        if 'sentences_ids' not in raw_datasets["train"].column_names:
            idxes = np.arange(raw_datasets["train"].num_rows)
            raw_datasets["train"] = raw_datasets["train"].add_column("sentences_ids", idxes)
    if "validation" in raw_datasets.keys():
        if 'sentences_ids' not in raw_datasets["validation"].column_names:
            idxes = np.arange(raw_datasets["validation"].num_rows)
            raw_datasets["validation"] = raw_datasets["validation"].add_column("sentences_ids", idxes)

    return raw_datasets


def custom_load_dataset(data_args, model_args):
    """
    Load required dataset either from local files.
    """
    data_files = {}
    
    if data_args.train_file is not None:
        data_files["train"] = data_args.train_file
    extension = data_args.train_file.split(".")[-1]
    
    if extension == "txt":
        extension = "text"
        raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )

    elif extension == 'csv':
        raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            delimiter='\t',
            column_names=['text', 'label'],
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )

    return raw_datasets


def init_sent_vector_embedding(model, model_args, tokenizer, training_args, tokenized_dataset):
    """
    init the sentence vector according to the model_args parameters
    """
    # Initialise sentence embedding in normal distribution
    if model_args.normal_init_range is not None:
        model.sentence_embedding.normal_init_embedding(mean=0.0, std=model_args.normal_init_range)
    # Initialise sentence embedding in uniform distribution
    elif model_args.uniform_init_range is not None:
        model.sentence_embedding.uniform_init_embedding(model_args.uniform_init_range)
    # Initialise with pmean of sentence embeddings
    elif model_args.init_with_pmean:
        model.cuda().eval()
        # Function to get pmean of contextual embedding
        def get_pmean(x, sent_lens, p):
            p = float(p)
            x_c = torch.tensor(x, dtype=torch.cfloat)
            return torch.pow(torch.sum(torch.pow(x_c, p), dim=1) / sent_lens, 1 / p).real

        # Batch encode data
        default_data_collator = DataCollatorWithPadding(tokenizer, return_tensors='pt')
        listed_tokenized_dataset = list(tokenized_dataset)
        batch_size = 128

        plm_avg_embedding = []
        for idx in range(0, len(listed_tokenized_dataset), batch_size):
            cur_tokenized_dataset = listed_tokenized_dataset[idx: idx + batch_size]
            batch_input = default_data_collator(cur_tokenized_dataset)
   
            for k in batch_input:
                batch_input[k] = batch_input[k].to(training_args.device)
            
            masks = batch_input['attention_mask']
            sent_lens = masks.sum(1, keepdims=True)

            with torch.no_grad():
                if model_args.plm_name == 'bert':
                    cur_plm_output = model.bert.embeddings.word_embeddings(batch_input['input_ids'])
                elif model_args.plm_name == 'roberta':
                    cur_plm_output = model.roberta.embeddings.word_embeddings(batch_input['input_ids'])
            
            masked_cur_plm_output = cur_plm_output * masks[:, :, None]
    
            cur_plm_avg_embedding = []
            
            for p in range(1, model_args.prompt_length + 1):
                cur_plm_avg_embedding.append(get_pmean(masked_cur_plm_output, sent_lens, p))
            cur_plm_avg_embedding = torch.cat(cur_plm_avg_embedding, dim=1)
            plm_avg_embedding.append(cur_plm_avg_embedding)
        
        plm_avg_embedding = torch.cat(plm_avg_embedding, dim=0).detach().cpu()
        
        # Set new sentence embeddings
        model.sentence_embedding.set_sentence_embedding(plm_avg_embedding)

    # Initialise with zero sentence embeddings
    elif model_args.init_with_zero:
        zero_embedding = torch.zeros_like(model.sentence_embedding.sentence_embedding.weight)
        model.sentence_embedding.set_sentence_embedding(zero_embedding)
    # Else do noething
    return
    

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


def compute_sentence_vector_tsne(sent_embeds, labels, output_file):
    """
    Compute tsne (n_components=2) for the trained sentence vectors
    """
    if isinstance(sent_embeds, torch.Tensor):
        sent_embeds = sent_embeds.numpy()

    # perplexity = 30
    # lr = 500
    # prefix_file = output_file.rsplit('.', 1)[0]
    # output_file = f'{prefix_file}_ppl{perplexity}_lr{lr}.png'
    tsne = TSNE(n_components=2, learning_rate='auto', init='pca')
    tsne_sent_embeds = tsne.fit_transform(sent_embeds)
    tsne_sent_x, tsne_sent_y = tsne_sent_embeds[:, 0], tsne_sent_embeds[:, 1]
    sns.scatterplot(x=tsne_sent_x, y=tsne_sent_y, hue=labels, palette='deep')
    plt.savefig(output_file, format='png')
    plt.close()


def write_final_results_to_one_file(model_args, data_args, training_args, res_metrics, save_file_path="../final_result.csv"):
    """
    Gather final result in one file
    """
    # prpare the result list
    plm_name = model_args.plm_name
    target_task = data_args.target_task
    optimizer_name = model_args.optimizer_name
    epoch = training_args.num_train_epochs
    batch_size = training_args.per_device_train_batch_size
    lr = training_args.learning_rate
    lr_scheduler = training_args.lr_scheduler_type
    prompt_length = model_args.prompt_length
    mlm_prob = data_args.mlm_probability
    warmup_ratio = training_args.warmup_ratio
    weight_decay = training_args.weight_decay
    mask_scheme = model_args.lm_head_name
    duplicate_times = data_args.duplicate_times
    seed = training_args.seed

    concat_cls_res, concat_avg_res = -1, -1
    mix_cls_res, mix_avg_res = -1, -1
    only_sentvec_res = -1
    
    for key in res_metrics:
        if 'testacc_concat_cls' in key:
            concat_cls_res = res_metrics.get(key)
        elif 'testacc_concat_avg' in key:
            concat_avg_res = res_metrics.get(key)
        elif 'testacc_mix_cls' in key:
            mix_cls_res = res_metrics.get(key)
        elif 'testacc_mix_avg' in key:
            mix_avg_res = res_metrics.get(key)
        elif 'testacc_sentvec' in key:
            only_sentvec_res = res_metrics.get(key)

    result_list = [plm_name, target_task, optimizer_name, duplicate_times, prompt_length, lr, lr_scheduler,
        mlm_prob, epoch, seed, mask_scheme, warmup_ratio, weight_decay, batch_size,
        mix_cls_res, concat_cls_res, mix_avg_res, concat_avg_res, only_sentvec_res
    ]

    # append result to the csv file
    with open(save_file_path, 'a+', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(result_list)
