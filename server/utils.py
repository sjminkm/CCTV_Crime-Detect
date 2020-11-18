#utils
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image

from torchtext.data.metrics import bleu_score
import nltk.translate.bleu_score as bleu
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import corpus_bleu
from datetime import datetime
from konlpy.tag import Okt

import numpy as np
global gram_1
global gram_2
global gram_3
global gram_mean
gram_1=[]
gram_2=[]
gram_3=[]
gram_mean=[]
Okt = Okt()

def print_examples(model, device, dataset):
    
    img_names=['109671650_f7bbc297fa.jpg','109738763_90541ef30d.jpg','109738916_236dc456ac.jpg',
               '109823394_83fcb735e1.jpg','111537217_082a4ba060.jpg']
    
    transform = transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    
    model.eval()
    for i in img_names:
        if(i=='109738763_90541ef30d.jpg'):
            break;
        test_img1 = transform(Image.open("test_examples/{}".format(i)).convert("RGB")).unsqueeze(0)
        print("Example 1 CORRECT: \n")
        candidate_corpus=dataset.df[dataset.df['image']==i]['caption']
        print(candidate_corpus)
        print("Example 1 OUTPUT: "+ " ".join(model.caption_image(test_img1.to(device), dataset.vocab)[1:-1]))
        c=[]
        for i in img_names:
            candidate_corpus=dataset.df[dataset.df['image']==i]['caption']

            a=np.array(candidate_corpus)
           
            for i in a:
                b=[tok for tok in Okt.morphs(i)]
                c.append(b)
            #bleu_score(c,
        bleu_1gram = sentence_bleu(c, model.caption_image(test_img1.to(device), dataset.vocab)[1:-1], weights=(1, 0, 0, 0))
        bleu_2gram = sentence_bleu(c, model.caption_image(test_img1.to(device), dataset.vocab)[1:-1], weights=(0, 1, 0, 0))
        bleu_3gram = sentence_bleu(c, model.caption_image(test_img1.to(device), dataset.vocab)[1:-1], weights=(0, 0, 1, 0))
        bleu_cum1 = sentence_bleu(c, model.caption_image(test_img1.to(device), dataset.vocab)[1:-1], weights=(0.33, 0.33, 0.33, 0)) #4그램 제외 모두곱하는 특성상 0이나옴


        print(f'1-Gram BLEU: {bleu_1gram:.2f}')
        print(f'2-Gram BLEU: {bleu_2gram:.2f}')
        print(f'3-Gram BLEU: {bleu_3gram:.2f}')
        
        print(f'Geometric Mean 3-Gram Cumulative BLEU (nltk) : {bleu_cum1:.2f}')

        gram_1.append(f'{bleu_1gram:.2f}')
        gram_2.append(f'{bleu_2gram:.2f}')
        gram_3.append(f'{bleu_3gram:.2f}')
        gram_mean.append(f'{bleu_cum1:.2f}')
        
        d={'1_gram': gram_1,'2_gram':gram_2,'3_gram':gram_3,'1~3_grams_mean':gram_mean}
        df=pd.DataFrame(d)
        df.to_csv("Second_Bleu_graph.csv",encoding="cp949",index=False)
        print("=> Saving graph.CSV")
    model.train()

def print_caption(model, device, dataset,img_name):
    transform = transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    now= datetime.now()
    model.eval()
    test_img1 = transform(Image.open("image_ex/{}".format(img_name)).convert("RGB")).unsqueeze(0)
    dt_string = f"{now.year}년{now.month}월 {now.day}일 {now.hour:02}시 {now.minute:02}분 {now.second:02}초"
    vocab_list=model.caption_image(test_img1.to(device), dataset.vocab)[1:-1]
    result="현재위치: 손민성 집의 CCTV \n 촬영시각: {} \n 상세내용: ".format(dt_string)+"".join(vocab_list)
    
    if "칼" in vocab_list :
        with open("text_output/output.txt", "w") as file:
            file.write("<위험상황>\n"+result)
    else :
        with open("text_output/output.txt", "w") as file:
            file.write("<기존상황>\n"+result)
    
                
    model.train()

def save_checkpoint(state, filename="my_checkpoint_Adam_GRU_seq2seq1.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

    


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    step = checkpoint["step"]
    return step