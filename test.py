import argparse
#from utilities import text_helper
from langdetect import detect
import cv2
from PIL import Image
import json
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
from torch.optim import lr_scheduler
from data_loader import get_loader
from models import VqaModel, SANModel
import warnings 

warnings.filterwarnings("ignore")
#from resize_images import resize_image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_str_list(fname):
    with open(fname) as f:
        lines = f.readlines()
    lines = [l.strip() for l in lines]
    return lines


qst_vocab = load_str_list("datasets/vocab_questions.txt")
ans_vocab = load_str_list("datasets/vocab_answers.txt")
word2idx_dict = {w:n_w for n_w, w in enumerate(qst_vocab)}
unk2idx = word2idx_dict['<unk>'] if '<unk>' in word2idx_dict else None
vocab_size = len(qst_vocab)

def load_image(image_path, transform=None):
    image = Image.open(image_path).convert('RGB')
    image = image.resize([224, 224], Image.LANCZOS)
    
    if transform is not None:
        image = transform(image).unsqueeze(0)
    
    return image

def visualizeAttention(model, img, layer):
    m = nn.Upsample(size=(224,224), mode='bilinear')
    pi = model.attn_features[layer].squeeze()
    print(pi.size())
    pi = pi.view(14,14)
    attn = m(pi)
    
    image = image.squeeze(0)
    img = torch.numpy(img)
    attn  = torch.numpy(attn)
#     print(image.shape, attn.shape)
    ## Visualization yet to be completed
    

def word2idx(w):
    if w in word2idx_dict:
        return word2idx_dict[w]
    elif unk2idx is not None:
         return unk2idx
 
    else:
        raise ValueError('word %s not in dictionary (while dictionary does not contain <unk>)' % w)
        
def main(args):
     
    """f = open('image_test.json', 'r', encoding="utf-8")
    data = f.read()
    image_list = json.loads(data)

    f = open('ques_test.json', 'r', encoding="utf-8")
    data = f.read()
    ques_list = json.loads(data)
    dict_ans = {}

    for key_ques in ques_list:
        if (detect(key_ques["question"]) != 'en'):
            tmp = key_ques["id"]
            #dict_ans[tmp] = key_ques["question"]
            #dict_ans[tmp] = " "
        else:
            for idx, key_image in enumerate(image_list):
                if idx <= 10:
                    if (key_image["image_id"] == key_ques["image_id"]):
                        image_path = key_image["question"]
                        question = key_ques["question"]
                        #print(key_image["image_id"])
                        #print("image_path", image_path)
                        #print("question", question)
                        #image = cv2.imread(args.image_path)
                        image = cv2.imread(image_path)
                        #print("image_path", image_path)
                        if image is None:
                            print("wrong_path")
                        else:
                            image = cv2.resize(image, dsize=(224,224), interpolation = cv2.INTER_AREA)
                            image = torch.from_numpy(image).float()
                            image = image.to(device)
                            image = image.unsqueeze(dim=0)
                            image = image.view(1,3,224,224)
                            image = image.to(device)
                        
                        max_qst_length=30
                        
                        #question = args.question
                        #print('what is this language?', detect(question))
                        q_list = list(question.split(" "))
                        #print(q_list)
                        
                        idx = 'valid'
                        qst2idc = np.array([word2idx('<pad>')] * max_qst_length)  # padded with '<pad>' in 'ans_vocab'
                        qst2idc[:len(q_list)] = [word2idx(w) for w in q_list]

                        question = qst2idc
                        question = torch.from_numpy(question).long()
                        
                        question = question.to(device)
                        question = question.unsqueeze(dim=0)
                        model = torch.load(args.saved_model)
                        model = model.to(device)
                        #print("model", model)
                        #torch.cuda.empty_cache()
                        model.eval()
                        output = model(image, question)
                        
                        print("output", output)
                        print("output_size", output.size())
                        #print("output", output)
                    #     Visualization yet to be implemented
                    #     if model.__class__.__name__ == "SANModel":
                    #         print(model.attn_features[0].size())
                    #          visualizeAttention(model, image, layer=0)
                        predicts = torch.softmax(output, 1)
                        #print("predicts_size", predicts.size())
                        #print("predicts", predicts)
                        #print("predicts", predicts.size(), predicts)
                        probs, indices = torch.topk(output, k=6, dim=1)
                        probs = probs.squeeze()
                        indices = indices.squeeze()
                        #print("indices", indices.size())
                        #print("predicted - probabilty")
                        
                        tmp = key_ques["id"]
                        #print("probs.size()", probs.size(), "indices.size()", indices.size())
                        #print("ans_vocab", ans_vocab[indices[1].item()],"probs",probs[1].item())
                        #print("'{}' - {:.4f}".format(ans_vocab[indices[0].item()], probs[0].item()))
                        
                        
                        #if ans_vocab[indices[0].item()] == "<unk>":
                            #print("'{}' - {:.4f}".format(ans_vocab[indices[1].item()], probs[1].item()))
                        #else:
                        #print("'{}' - {:.4f}".format(ans_vocab[indices[0].item()], probs[0].item()))
                        #for i in range(5):
                        #    dict_ans[tmp] += ans_vocab[indices[i].item()] + " "
                        #    print(ans_vocab[indices[i].item()] + " ")
                        #print("_____")
                        ans = ""
                        for i in range(5):
                    #         print(probs.size(), indices.size())
                    #         print(ans_vocab[indices[1].item()],probs[1].item())
                            #if ans_vocab[indices[i].item()] != "<unk>":

                            if ans_vocab[indices[i].item()] != "<unk>":
                                ans += ans_vocab[indices[i].item()] + " "
                        print(ans)
                        dict_ans[tmp]  = ans
                else:
                    break
                    
    
    #with open('results.json', 'w', encoding='utf-8') as f:
    #    json.dump(dict_ans, f, ensure_ascii=False)
"""
    image = cv2.imread(args.image_path)
    image = cv2.resize(image, dsize=(224,224), interpolation = cv2.INTER_AREA)
    image = torch.from_numpy(image).float()
    image = image.to(device)
    image = image.unsqueeze(dim=0)
    image = image.view(1,3,224,224)
    
    max_qst_length=30
    
    question = args.question
    q_list = list(question.split(" "))
    #     print(q_list)
    
    idx = 'valid'
    qst2idc = np.array([word2idx('<pad>')] * max_qst_length)  # padded with '<pad>' in 'ans_vocab'
    qst2idc[:len(q_list)] = [word2idx(w) for w in q_list]

    question = qst2idc
    question = torch.from_numpy(question).long()
    
    question = question.to(device)
    question = question.unsqueeze(dim=0)
    model = torch.load(args.saved_model)
    model = model.to(device)
    #torch.cuda.empty_cache()
    model.eval()
    output = model(image, question)
      
#     Visualization yet to be implemented
#     if model.__class__.__name__ == "SANModel":
#         print(model.attn_features[0].size())
#          visualizeAttention(model, image, layer=0)
    predicts = torch.softmax(output, 1)
    probs, indices = torch.topk(output, k=10, dim=1)
    probs = probs.squeeze()
    indices = indices.squeeze()
    print("predicted - probabilty")
    ans = ""
    for i in range(10):
#         print(probs.size(), indices.size())
#         print(ans_vocab[indices[1].item()],probs[1].item())
        #if ans_vocab[indices[i].item()] != "<unk>":

        if ans_vocab[indices[i].item()] != "<unk>":
            ans += ans_vocab[indices[i].item()] + " "
    print(ans)
if __name__ == '__main__':
   
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type = str, required=True)
    parser.add_argument('--question', type = str, required=True)
    parser.add_argument('--saved_model', type = str, required=True)
       
    args = parser.parse_args()
    main(args)
