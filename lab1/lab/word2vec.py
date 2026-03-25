from gensim.models import word2vec

# 加载语料
#sentences = word2vec.Text8Corpus('./data/seg_text.txt')
# 加载我的语料
sentences = word2vec.Text8Corpus('./data/seg_mytext.txt')



# 训练模型
window = 7 # 窗口大小
vector_size = 128  # 嵌入向量维度
sg = 1 # 是否使用skip-gram 0表示否
epochs = 1 # 训练轮次
negative = 2   
min_count = 2
seed = 42 # 随机种子
workers = 4 


model = word2vec.Word2Vec(
    sentences, 
    window=window, 
    vector_size=vector_size, 
    epochs=epochs, 
    seed=seed, 
    sg=sg,
    min_count=min_count,
    negative=negative,
    workers=workers
)

# model.save("./ckpt/0325.model")
model.save("./ckpt/0325_mytext.model")