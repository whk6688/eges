import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
import networkx as nx
import sys
from eges.walker import RandomWalker
import time
from eges.EGES_Model import EGES_Model
import tensorflow as tf
from eges.utils import *

path=sys.argv[1]
print('path:------------------',path)
print(path,"--->step1")
#step 1------------------------------------------------------------------

all_type={ 'resblockName' : str , 'businessdistrict' : str }
house_info = pd.read_csv('housepic/house_info.csv', dtype=all_type)
#print(house_info.dtypes)
session_list_all = []
houseidset=set()
with open(path+'/click.csv', 'r') as f:
    list1 = f.readlines()
    for str in list1:
        session_list_all.append(list(map(int,str.split())))
        a=set(map(int, str.split()))
        houseidset=houseidset.union(a)

all_houseid = pd.DataFrame({'invNo': list(houseidset)})
house_lbe = LabelEncoder()
all_houseid['houseid'] = house_lbe.fit_transform(all_houseid['invNo'])
#all_houseid['houseid_source']=house_lbe.inverse_transform(all_houseid['houseid_code'])

node_pair = dict()

for session in session_list_all:
    for i in range(1, len(session)):
            if (session[i - 1], session[i]) not in node_pair.keys():
                node_pair[(session[i - 1], session[i])] = 1
            else:
                # node的边权可以由两个节点共同出现的次数决定
                node_pair[(session[i - 1], session[i])] += 1

in_node_list = list(map(lambda x: x[0], list(node_pair.keys())))
out_node_list = list(map(lambda x: x[1], list(node_pair.keys())))
weight_list = list(node_pair.values())
graph_df = pd.DataFrame({"in_node": in_node_list
                         , "out_node": out_node_list
                         , "weight": weight_list
                        })

# 存储格式为: in_node out_node weight

graph_df['in_node']=house_lbe.transform(graph_df['in_node'])
graph_df['out_node']=house_lbe.transform(graph_df['out_node'])

graph_df.to_csv(path+"/graph_df.csv", sep = " ", index=False, header=False)

##step 2------------------------------------------------------------------------
print(path,"--->step2")
G = nx.read_edgelist(path+'/graph_df.csv'
                     , create_using=nx.DiGraph()
                     , nodetype=None
                     , data=[('weight', int)]
                    )
# p = 1, q = 1 为随机游走
walker = RandomWalker(G, p = 1, q = 1)
walker.preprocess_transition_probs()
session_reproduce = walker.simulate_walks(num_walks=6, walk_length=8, workers=1,
                                          verbose=2)

session_reproduce = list(filter(lambda x: len(x) > 2 , session_reproduce))

def get_graph_context_all_pairs(walks, window_size):
    all_pairs = []
    for k in range(len(walks)):
        for i in range(len(walks[k])):
            for j in range(i - window_size, i + window_size + 1):
                if i == j or j < 0 or j >= len(walks[k]):
                    continue
                else:
                    all_pairs.append([walks[k][i], walks[k][j]])
    return np.array(all_pairs, dtype=np.int32)

all_pairs = get_graph_context_all_pairs(session_reproduce, window_size=5)
np.savetxt(path+"/all_pairs", X=all_pairs, fmt="%d", delimiter=" ")

##step 3------------------------------------------------------------------------
print(path,"--->step3")
# add side info
all_info = pd.merge(all_houseid, house_info, on='invNo', how='left').fillna("0")
#print(all_info.dtypes)
for feat in all_info.columns:
        if feat != 'invNo' and  feat!='houseid':
            #print('name--------->',feat)
            #print(all_info[feat].unique())
            lbe = LabelEncoder()
            all_info[feat] = lbe.fit_transform(all_info[feat])

all_info.to_csv(path+'/houseid_info.csv', index=False, header=False, sep='\t', columns=['invNo', 'houseid','priceRange','resblockName','businessdistrict','subwayStation','subwayLine','houseType'])
#print(all_info)
##step 4------------------------------------------------------------------------

print(path,"--->step4")
print_every_k_iterations = 100
loss = 0
iteration = 0
start = time.time()
batch_size=2048
epochs=30
num_feat=7

# read train_data
#print('read features...')
start_time = time.time()
side_info = np.loadtxt(path+'/houseid_info.csv', dtype=np.int32, delimiter='\t')
#print(side_info)
side_info = np.delete(side_info, 0, axis=1)
#print(side_info)
all_pairs = np.loadtxt(path+'/all_pairs', dtype=np.int32, delimiter=' ')
feature_lens = []
for i in range(side_info.shape[1]):
    tmp_len = len(set(side_info[:, i]))
    feature_lens.append(tmp_len)
end_time = time.time()
print('time consumed for read features: %.2f' % (end_time - start_time))

EGES = EGES_Model(len(side_info), num_feat, feature_lens, n_sampled=10, embedding_dim=16,
                  lr=0.001)

# init model
print('init...')
start_time = time.time()
init = tf.global_variables_initializer()
config_tf = tf.ConfigProto()
config_tf.gpu_options.allow_growth = True
sess = tf.Session(config=config_tf)
sess.run(init)
end_time = time.time()
print('time consumed for init: %.2f' % (end_time - start_time))


max_iter = len(all_pairs)//batch_size*epochs
for iter in range(max_iter):
    iteration += 1
    batch_features, batch_labels = next(graph_context_batch_iter(all_pairs, batch_size, side_info,
                                                                 num_feat))
    feed_dict = {input_col: batch_features[:, i] for i, input_col in enumerate(EGES.inputs[:-1])}
    feed_dict[EGES.inputs[-1]] = batch_labels
    _, train_loss = sess.run([EGES.train_op, EGES.cost], feed_dict=feed_dict)

    loss += train_loss

    if iteration % print_every_k_iterations == 0:
        end = time.time()
        e = iteration*batch_size//len(all_pairs)
        print("Epoch {}/{}".format(e, epochs),
              "Iteration: {}".format(iteration),
              "Avg. Training loss: {:.4f}".format(loss / print_every_k_iterations),
              "{:.4f} sec/batch".format((end - start) / print_every_k_iterations))
        loss = 0
        start = time.time()
print("train end")
feed_dict_test = {input_col: list(side_info[:, i]) for i, input_col in enumerate(EGES.inputs[:-1])}
feed_dict_test[EGES.inputs[-1]] = np.zeros((len(side_info), 1), dtype=np.int32)
embedding_result = sess.run(EGES.merge_emb, feed_dict=feed_dict_test)
print('saving embedding result...')
write_embedding(embedding_result, path+"/res.emb")

##step 5------------------------------------------------------------------------
print(path,"--->step5")
emb_info = np.loadtxt(path+'/res.emb', dtype=np.str, delimiter=' ')
side_info = np.loadtxt(path+'/houseid_info.csv', dtype=np.str, delimiter='\t')
res=np.c_[side_info[:,0],emb_info]
np.savetxt(path+"/fin_emb", X=res, fmt="%s", delimiter=" ")
