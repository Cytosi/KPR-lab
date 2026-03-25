from gensim.models import word2vec
import numpy as np
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontManager
import matplotlib
from matplotlib import font_manager
font_path = '/usr/share/fonts/myFonts/YouYuan.ttf'
font_manager.fontManager.addfont(font_path)
matplotlib.rc("font", family='YouYuan')

model = word2vec.Word2Vec.load('./ckpt/0325_mytext.model')

print("------------相似词------------")
sim1 = model.wv.similarity('锈湖', '劳拉')
sim2 = model.wv.similarity('锈湖', '方块')
print(f'sim(锈湖, 劳拉) = {sim1:.4f}')
print(f'sim(锈湖, 方块) = {sim2:.4f}')
print("--------------------------------")
print("\n"*3)
print("------------不相似词------------")
sim1 = model.wv.similarity('戴尔', '鹿')
sim2 = model.wv.similarity('方块', '白门')
print(f'sim(戴尔, 鹿) = {sim1:.4f}')
print(f'sim(方块, 白门) = {sim2:.4f}')
print("--------------------------------")
print("\n"*3)