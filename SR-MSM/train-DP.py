import os
import torch
import numpy as np
from sklearn import metrics
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from torch.utils.data.sampler import WeightedRandomSampler
from utils import CustomDataset, BERTClass, load_json, load_json_by_line, collate_fn
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--target", default='imp', required=True, type=str)
parser.add_argument('--cuda_num', type=int, default=0, help='CUDA device number')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# load train/dev set
prefix = 'data'
train = load_json_by_line(os.path.join(prefix, args.target, 'train.json'))
dev = load_json_by_line(os.path.join(prefix, args.target, 'dev.json'))

data_dir = '../../../dataset'
mappings_path = os.path.join(data_dir, 'mappings.json')
sym2id, _, _, _, sl2id, _ = load_json(mappings_path)
num_labels = len(sym2id) if args.target == 'exp' else len(sym2id) * len(sl2id)

model_prefix = os.path.join('saved', args.target)
os.makedirs(model_prefix, exist_ok=True)

model_dir = 'E:/ZHHB/now/Demo/V2/V2/task2/bert/bert-base-chinese/'
MAX_LEN = 128 if args.target == 'exp' else 64
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 32
EPOCHS = 50 if args.target == 'exp' else 25
LEARNING_RATE = 1e-5

tokenizer = BertTokenizer.from_pretrained(model_dir)

train_set = CustomDataset(train, tokenizer, MAX_LEN, num_labels)
dev_set = CustomDataset(dev, tokenizer, MAX_LEN, num_labels)

train_params = {
    'batch_size': TRAIN_BATCH_SIZE,
    'num_workers': 1
}

dev_params = {
    'batch_size': VALID_BATCH_SIZE,
    'shuffle': False,
    'num_workers': 1
}

weights = [sample['weight'] for sample in train]
sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

if args.target == 'exp':
    train_loader = DataLoader(train_set, collate_fn=collate_fn, **train_params)
else:
    train_loader = DataLoader(train_set, sampler=sampler, collate_fn=collate_fn, **train_params)
dev_loader = DataLoader(dev_set, collate_fn=collate_fn, **dev_params)

model = BERTClass(model_dir, num_labels)
model.to(device)

optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)


def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)


def create_prompt(symptoms):
    """ 创建Prompt句子，拼接到输入中 """
    prompt = f"患者的症状可能是以下几种：{', '.join(symptoms)}。请从下面的对话中提取出所有相关症状。"
    return prompt


def preprocess_dialogue(dialogue, tokenizer, max_len):
    """ 重组对话流为问答对，并对症状部分进行标记 """
    qa_pairs = []
    for turn in dialogue:
        if is_valid_turn(turn):  # 过滤无效信息，如寒暄、无关内容
            qa_pair = {
                'question': turn['question'],
                'answer': turn['answer'],
                'symptom': turn.get('symptom', None)  # 可能包含的症状信息
            }
            qa_pairs.append(qa_pair)

    # 对每个问答对进行标记和编码处理
    encoded_pairs = []
    for pair in qa_pairs:
        encoded_pair = tokenizer.encode_plus(
            pair['question'], pair['answer'],
            max_length=max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        encoded_pairs.append(encoded_pair)
    return encoded_pairs


def adversarial_training(data):
    """ 对抗训练，通过小的扰动增强模型鲁棒性 """
    perturbed_data = []
    for sample in data:
        if np.random.rand() < 0.1:  # 以10%的概率进行对抗扰动
            perturbed_sample = apply_noise(sample)  # 对数据进行小扰动
            perturbed_data.append(perturbed_sample)
        else:
            perturbed_data.append(sample)
    return perturbed_data


def apply_noise(sample):
    """ 对输入样本进行轻微扰动，例如替换同义词 """
    return sample  # 简单返回，待具体实现


def train_epoch(_epoch):
    symptoms = ['头痛', '发热', '咳嗽', '胸痛']  # 示例症状
    prompt = create_prompt(symptoms)

    total_loss = 0.0

    for step, data in enumerate(train_loader):
        data = adversarial_training(data)  # 加入对抗训练

        ids = data['ids'].to(device, dtype=torch.long)
        mask = data['mask'].to(device, dtype=torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
        targets = data['targets'].to(device, dtype=torch.float)

        # 将Prompt句子与输入拼接
        prompt_ids = tokenizer(prompt, return_tensors='pt', truncation=True, padding='max_length',
                               max_length=MAX_LEN).input_ids
        ids = torch.cat([prompt_ids.to(device), ids], dim=1)

        outputs = model(ids, mask, token_type_ids)
        loss = loss_fn(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if step % 10 == 0:
            print(f'Epoch [{_epoch + 1}/{EPOCHS}], Step [{step}/{len(train_loader)}], Loss: {loss.item():.4f}')

    avg_loss = total_loss / len(train_loader)
    print(f'End of Epoch {epoch + 1}, Average Loss: {avg_loss:.4f}')


def validate(_epoch):
    fin_targets = []
    fin_outputs = []
    with torch.no_grad():
        for _, data in enumerate(dev_loader):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.float)
            outputs = model(ids, mask, token_type_ids)
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
    fin_outputs = np.array(fin_outputs) >= 0.5
    accuracy = metrics.accuracy_score(fin_targets, fin_outputs)
    f1_score_micro = metrics.f1_score(fin_targets, fin_outputs, average='micro', zero_division=0)
    f1_score_macro = metrics.f1_score(fin_targets, fin_outputs, average='macro', zero_division=0)
    print("Dev epoch: {}, Acc: {}, Micro F1: {}, Macro F1: {}".format(
        _epoch + 1, round(accuracy, 4), round(f1_score_micro, 4), round(f1_score_macro, 4)))
    return accuracy, f1_score_micro, f1_score_macro


print('total steps: {}'.format(len(train_loader) * EPOCHS))
best_micro_f1 = -1

for epoch in range(EPOCHS):
    model.train()
    train_epoch(epoch)
    model.eval()
    with torch.no_grad():
        _, micro_f1, _ = validate(epoch)
    if micro_f1 > best_micro_f1:
        print('saving model to : {}'.format(model_prefix))
        torch.save(model, os.path.join(model_prefix, 'model.pkl'))
        best_micro_f1 = micro_f1
