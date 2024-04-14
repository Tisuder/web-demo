pip install numpy
pip install pandas
pip install sentencepiece
pip install easyocr
import numpy as np
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import pandas as pd
import json
import cv2 as cv
from torch.cuda import is_available
import os
from difflib import SequenceMatcher
from PIL import Image
import easyocr as ocr

prod_categ = {
    '1.1': 'Сухие или быстрорастворимые хлопья крахмалистые продукты',
    '2.1': 'Десерты на молочной основе и крупяные продукты',
    '2.2': 'Фруктовое пюре с добавлением овощей, круп и молока или без них',
    '2.3': 'Овощные пюре',
    '2.4': 'Пюре из овощей с добавлением круп',
    '2.5': 'Пюре с сыром, не содержащее мясо или рыбу',
    '2.6': 'Пюре из мяса или рыбы, упомянутые в названии продукта первыми',
    '2.7': 'Пюре из мяса или рыбы, НЕ упомянутые в названии продукта первыми',
    '2.8': 'Пюре, состоящее только из мяса, рыбы или сыра',
    '3.1': 'Мясные, рыбные или сырные блюда с крупными кусочками',
    '3.2': 'Блюдо на овощной основе с кусочками',
    '4.1': 'Кондитерские изделия, сладкие пасты и фруктовые жевательные изделия',
    '4.2': 'Фрукты (свежие или сушеные целые фрукты или кусочки)',
    '4.3': 'Другие закуски',
    '5.1': 'Однокомпонентные или смешанные фруктовые, овощные соки или другие напитки без примесей'
}

def chek(img):
    pathes = []
    for path, _, files in os.walk('./data'):
        if len(files) == 0:
            continue

        t_pathes = [f'{path}\\' + file for file in files]
        pathes.append(t_pathes)

    raw_pathes = []


    for folder in pathes:
        for path in folder:
            raw_pathes.append(path)
            ex1 = img
    print(ex1.shape)
    ex1 = cv.resize(ex1, (round(
        ex1.shape[0] * 1.5), round(ex1.shape[1] * 1)), fx=0, fy=0, interpolation=cv.INTER_CUBIC)
    Image.fromarray(ex1)



    gray = cv.cvtColor(ex1, cv.COLOR_BGR2GRAY)
    r,gray= cv.threshold(gray,125,255,cv.THRESH_BINARY_INV)

    #gray = cv.GaussianBlur(gray, (3,3), sigmaX=1, sigmaY=1)

    Image.fromarray(ex1)
    reader = ocr.Reader(['ru'])
    result = reader.readtext(ex1, batch_size=30)
    def is_intersected(first_rect, second_rect) -> bool:
        tl1, tr1, bl1, br1 = first_rect
        tl2, tr2, bl2, br2 = second_rect

        x1_1,x2_1,y1_1,y2_1 = tl1[0], br1[0], tl1[1], br1[1]
        x1_2,x2_2,y1_2,y2_2 = tl2[0], br2[0], tl2[1], br2[1]

        x1 = max(min(x2_1, x1_1), min(x1_2, x2_2))
        y1 = max(min(y1_1, y2_1), min(y1_2, y2_2))
        x2 = min(max(x2_1, x1_1), max(x1_2, x2_2))
        y2 = min(max(y1_1, y2_1), max(y1_2, y2_2))

        return x1<x2 and y1<y2
        

    def merge_rects(first_rect, second_rect) -> list:
        tl1, tr1, bl1, br1 = first_rect
        tl2, tr2, bl2, br2 = second_rect

        x1_1,x2_1,y1_1,y2_1 = tl1[0], br1[0], tl1[1], br1[1]
        x1_2,x2_2,y1_2,y2_2 = tl2[0], br2[0], tl2[1], br2[1]

        x1 = min(min(x1_1, x2_1), min(x1_2, x2_2))
        x2 = max(max(x1_1, x2_1), max(x1_2, x2_2))
        y1 = min(min(y1_1, y2_1), min(y1_2, y2_2))
        y2 = max(max(y1_1, y2_1), max(y1_2, y2_2))

        return [(x1,y1),(x2,y1),(x1,y2),(x2,y2)]

        
    class Token:
        def __init__(self, rect : list, text : str) -> None:
            self.rect = rect
            self.text = text

    tokens = []
    for (bbox, arr, prob) in result:
        (tl, tr, br, bl) = bbox
        tl = (int(tl[0]), int(tl[1]))
        tr = (int(tr[0]), int(tr[1]))
        bl = (int(bl[0]), int(bl[1]))
        br = (int(br[0]), int(br[1]))

        tokens.append(Token([tl, tr,bl, br], arr))

        

    #tokens.sort(key=lambda token : math.sqrt(token.rect[0][0] ** 2 + token.rect[0][1] ** 2))
    #for i in range(len(tokens)):
    #    cv.rectangle(ex1, tokens[i].rect[0], tokens[i].rect[3], (0, 0, 255), 1)
    #      Image.fromarray(ex1).save(f'./shit/{i}.jpg')
    def_token = tokens[0]
    blocks_text : list[Token] = []
    for i in range(1, len(tokens)):
        any_intersection = False
        for j in range(i, len(tokens)):
            curr = tokens[j]
            if is_intersected(def_token.rect, curr.rect):
                def_token.rect = merge_rects(def_token.rect, curr.rect)
                any_intersection = True

        if not any_intersection:
            blocks_text.append(def_token)
            def_token = tokens[i]

    blocks_text.append(def_token)

    for block in blocks_text:
        cv.rectangle(ex1, block.rect[0], block.rect[3], (0, 0, 255), 3)
        
    Image.fromarray(ex1)
    # better algo
    i = 0
    while i < tokens.__len__():
        j = 0
        base_token = tokens[i]
        any_intersection = False
        while j < tokens.__len__():
            if i == j:
                j += 1
                continue

            curr = tokens[j]
            if is_intersected(base_token.rect, curr.rect):
                base_token.rect = merge_rects(base_token.rect, curr.rect)
                tokens.remove(curr) # slo-o-w
            else:
                j += 1
            
        i += 1

    for token in tokens:
        cv.rectangle(ex1, token.rect[0], token.rect[3], (0, 0, 255), 3)


    Image.fromarray(ex1)
    blocks_text = []
    for token in tokens:
        tl, tr, bl, br = token.rect
        cropped = ex1[tl[1] : tl[1] + bl[1] - tl[1], tl[0] : tl[0] + tr[0] - tl[0]]
        arr = reader.readtext_batched(cropped, detail=0, batch_size=30)[0]
        blocks_text.append(' '.join(arr))


    #'\n'.join(token_texts)
    files = []
    for _, _, f in os.walk('./limitations/'):
        files = f

    limitations = []
    for file in files:
        path = './limitations/' + file
        a = json.loads(open(path, encoding='utf-8').read())
        limitations.append(a)
    path_to_model = "ai-forever/RuM2M100-418M"

    model = M2M100ForConditionalGeneration.from_pretrained(path_to_model)
    tokenizer = M2M100Tokenizer.from_pretrained(path_to_model, src_lang="ru", tgt_lang="ru")
    def normilize(src : str):
            exclude = r'@#^&*()_+!~{}\||.,>=<№;:?`'
            for ex in exclude:
                src = src.replace(ex, '')
            
            return src.lower()
    def analyze(sentence):
            pass

    for block in blocks_text:
            sentence = normilize(block)
                    
            encodings = tokenizer(sentence, return_tensors="pt")
            generated_tokens = model.generate(
                    **encodings, forced_bos_token_id=tokenizer.get_lang_id("ru"))
            
            answer = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

            analyze(answer)
            answers.append(answer)
            print(answer)

    # ["прийдя в МГТУ я был удивлен никого не обнаружив там..."]
    idx = 0
    for i in range(len(answers)):
        if 'энерг' in answers[i][0].lower():
            idx = i

    st = ''
    for i in range(idx + 1, len(answers)):
        st += ' ' + answers[i][0].lower()
    lst = st.split(' ')
    nums = '0123456789'
    flag = False
    flagg = False
    res = []
    for elem in lst:
        for el in elem:
            if el == 'г' and elem.find('г') == len(elem) - 1:
                flagg = True
            if el not in nums:
                flag = False
            else:
                flag = True
        if flag:
            try:
                res.append(int(elem))
            except Exception:
                pass
        if flagg:
            res.append(int(elem[:len(elem) - 1]))
        
        flagg = False
        flag = False
    res = res[2:len(res)-3]
    kal, pr, fat = res
    class Category:
        def __init__(self, energy, natrium, sugar, subsugar, protein, fat, fruits, calories):
            self.energy = energy
            self.natrium = natrium
            self.sugar = sugar
            self.subsugar = subsugar
            self.protein = protein
            self.fat = fat
            self.fruits = fruits
            self.calories = calories
        def check_condition(self, energy, natrium, sugar, subsugar, protein, fat, fruits, calories):
            return energy >= self.energy and natrium <= self.natrium and sugar <= self.sugar and subsugar == self.subsugar and 3 <= protein <= 10 and fat <= self.fat and fruits <= self.fruits and  calories >= self.calories
    kashi = Category(0, 50, 0, 0, 5.5, 4.5, 10, 0)
    milk_des = Category(0, 50, 0, 0, 2.2, 4.5, 5, 60)
    fruit_pure = Category(0, 50, 0, 0, 0, 4.5, 0, 60)
    veg_pure = Category(0, 50, 0, 0, 0, 4.5, 0, 0)
    veg_pure_zl = Category(0, 50, 0, 0, 0, 4.5, 0, 60)
    pure_cheeze = Category(0, 100, 0, 0, 3, 6, 5, 60)
    pure_meat = Category(0, 50, 0, 0, 4, 6, 5, 60)
    pure_meat_cheeze = Category(0, 50, 0, 0, 7, 6, 5, 0)
    konserv_meat = Category(0, 50, 0, 0, 3, 4.5, 5, 0)
    konserv_veg = Category(0, 50, 0, 0, 3, 4.5, 5, 0)
    fruit = Category(0, 50, 0, 0, 0, 4.5, 0, 50)
    zakuski = Category(0, 50, 0, 0, 0, 4.5, 0, 50)
    tovar = Category(0, 0, 0, 0, 3.0, 2.0, 0, 62)
    pozitiv_recomendation = "Продукт одобрен Всемирной Огранизацией Здравоохранения и безопасен для употребления"
    negativ_recomendation =  "Продукт не одобрен Всемирной Огранизацией Здравоохранения и не рекомендован к употреблению"
    def check(tovar):   
        if tovar.check_condition(0, 0, 0, 0, 3.0, 2.0, 0, 62):
            return pozitiv_recomendation
        else:
            return negativ_recomendation
    check(tovar)