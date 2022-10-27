import os
import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from utilities import text_helper


class VqaDataset(data.Dataset):

    def __init__(self, input_dir, input_vqa, max_qst_length=30, max_num_ans=30, transform=None):
        self.input_dir = input_dir
        self.vqa = np.load(input_dir+'/'+input_vqa,  allow_pickle=True)
        self.qst_vocab = text_helper.VocabDict(input_dir+'/vocab_questions.txt')
        self.ans_vocab = text_helper.VocabDict(input_dir+'/vocab_answers.txt')
        self.max_qst_length = max_qst_length
        self.max_num_ans = max_num_ans
        self.load_ans = ('valid_answers' in self.vqa[0]) and (self.vqa[0]['valid_answers'] is not None)
        self.transform = transform

    def __getitem__(self, idx):

        vqa = self.vqa
        qst_vocab = self.qst_vocab
        ans_vocab = self.ans_vocab
        #print("ans_vocab", ans_vocab)
        max_qst_length = self.max_qst_length
        max_num_ans = self.max_num_ans
        transform = self.transform
        load_ans = self.load_ans

        image = vqa[idx]['image_path']
        image = Image.open(image).convert('RGB')
        qst2idc = np.array([qst_vocab.word2idx('<pad>')] * max_qst_length)  # padded with '<pad>' in '"ans_vocab", ans_vocab'
        qst2idc[:len(vqa[idx]['question_tokens'])] = [qst_vocab.word2idx(w) for w in vqa[idx]['question_tokens']]
        #question = vqa[idx]['question_str']
        #print("qst2idc", qst2idc)
        #print(vqa[idx]['valid_answers'])
        #print("qst2idc", type(qst2idc), qst2idc.size)
        """for w in vqa[idx]['question_tokens']:
            print('w', w)"""
        
        sample = {'image': image, 'question': qst2idc}

        if load_ans:
            #ans = [ans_vocab.word2idx(w) for w in vqa[idx]['answer_tokens']]
            #print("ans", ans)
            #ans2 = np.random.choice(ans)
            #sample['answer_label'] = ans2         # for training
            ans2idc = np.array([ans_vocab.word2idx('<pad>')] * 2293)  # padded with '<pad>' in '"ans_vocab", ans_vocab'
            pos = []
            pos[:len(vqa[idx]['answer_tokens'])] = [ans_vocab.word2idx(w) for w in vqa[idx]['answer_tokens']]
            for i in pos:
                ans2idc[i-1] = 1
            ans2ts = torch.Tensor(ans2idc)
            #pos = np.array(pos) 
            #print("pos", type(pos), pos.size)
            #print("ans2ts", type(ans2ts), ans2ts)
            #print("ans2idc", ans2idc, type(ans2idc))
            #ans2idx = np.random.choice(ans2idc)
            #print("ans2idc", ans2idc)
            sample['answer_label'] = ans2ts
            #print(type(sample['answer_label']))
            """one_hot =  np.zeros((ans2idc.size(), ans2idc.max() + 1))
            one_hot[np.arrange(ans2idc.size()), ans2idc] = 1    """
            

            #print("ans2idc", ans2idc)
            mul2idc = list([-1] * max_num_ans)       # padded with -1 (no meaning) not used in 'ans_vocab'
            mul2idc[:len(ans2idc)] = ans2idc         # our model should not predict -1
            sample['answer_multi_choice'] = mul2idc  # for evaluation metric of 'multiple choice'

        if transform:
            sample['image'] = transform(sample['image'])

        return sample

    def __len__(self):

        return len(self.vqa)


def get_loader(input_dir, input_vqa_train, input_vqa_valid, max_qst_length, max_num_ans, batch_size, num_workers):

    transform = {
        phase: transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize((0.485, 0.456, 0.406),
                                                        (0.229, 0.224, 0.225))]) 
        for phase in ['train', 'valid']}

    vqa_dataset = {
        'train': VqaDataset(
            input_dir=input_dir,
            input_vqa=input_vqa_train,
            max_qst_length=max_qst_length,
            max_num_ans=max_num_ans,
            transform=transform['train']),
        'valid': VqaDataset(
            input_dir=input_dir,
            input_vqa=input_vqa_valid,
            max_qst_length=max_qst_length,
            max_num_ans=max_num_ans,
            transform=transform['valid'])}

    data_loader = {
        phase: torch.utils.data.DataLoader(
            dataset=vqa_dataset[phase],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers)
        for phase in ['train', 'valid']}

    return data_loader
