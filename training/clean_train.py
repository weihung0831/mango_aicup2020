from vgg19_reg import KerasTrainer
import losses
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, f1_score


def ranking_loss(y_true, y_pred):
    y_true_ = tf.cast(y_true, tf.float32)
    partial_losses = tf.maximum(0.0, 1 - y_pred[:, None, :] + y_pred[:, :, None])
    loss = partial_losses * y_true_[:, None, :] * (1 - y_true_[:, :, None])
    return tf.reduce_sum(loss)


def increase_data(values, increase_labels=[2, 4, 10]):
    # print(linees[50:70])
    ind = 0
    lab2, lab3, lab4 = [], [], []
    values = list(values)
    for data in values:

        if data[1] > 0:
            lab2.append(data)
        if data[2] > 0:
            lab3.append(data)
        if data[5] > 0:
            lab4.append(data)
        ind += 1
    # print(data,lab4*5)
    # increase the data size of each labels#
    lab2 = lab2 * increase_labels[0]
    lab3 = lab3 * increase_labels[1]
    lab4 = lab4 * increase_labels[2]

    final_data = values + lab2 + lab3 + lab4
    print('total labels after increasing', len(final_data))
    return np.array(final_data)


def count_labels(labels):
    counts = np.count_nonzero(labels, axis=0)
    print(counts)


df = pd.read_csv('./training/train_labelsx.csv')
values = df.values
print(len(values))
values = increase_data(values, increase_labels=[5, 32, 14])

X = values[:, 0]
y = values[:, 1:]
print(len(values))
print(count_labels(y))

print(len(values))
df = pd.read_csv('./training/test_labelsx.csv')
values = df.values
Xt = values[:, 0]
yt = values[:, 1:]
print(count_labels(yt))

bce = tf.keras.losses.BinaryCrossentropy()
topK = 1
prec = tf.keras.metrics.Precision(thresholds=0.5, top_k=topK)
rec = tf.keras.metrics.Recall(thresholds=0.5, top_k=topK)
AUC = tf.keras.metrics.AUC(num_thresholds=100, curve='ROC', summation_method='interpolation', multi_label=True)
tp = tf.keras.metrics.TruePositives(thresholds=0.5)

lnl = KerasTrainer(name='resnet_reg_vgg19',
                   num_classes=5,
                   target_img_size=224,
                   batch_size=48,
                   augment=['rotate', 'flip', 'blur', 'contrast', 'brightness'],
                   # brightness [-25,25] is applied to each image during training
                   pretrain_lr=1e-5,
                   reg_lr=1e-6,
                   opt='Adam',  # 'SGD'
                   # loss = [losses.active_passive_normalized_mean(1,1)],
                   # loss = [losses.normalized_macro_bce],
                   loss=[losses.active_passive_loss(1, 0.3)],
                   # loss = [ranking_loss],
                   # loss = [losses.macro_double_soft_f1],
                   metrics=[losses.macro_f1, prec, rec],
                   validation=0.20,
                   pretrain_epoch=75,
                   epochs=75, )

# lnl.load_weights('models/resnet_reg_0102Post_Epoch_050-197.680.h5') 
lnl.fit(X, y)

y_pred = lnl.predict_proba(Xt)
print(y_pred)
print(yt)
df = pd.DataFrame(y_pred).to_csv('./training/y_pred.csv')


def TP_acc(y_true, y_pred, ):
    # convert all values more than 0.5 to 1
    y_true = np.round(y_true.astype('float'))
    y_pred = np.round(y_pred.astype('float'))

    # cast float32 to int8
    y_true = y_true.astype('int').flatten()
    y_pred = y_pred.astype('int').flatten()

    # count non zero labels total number of true labels
    label_counts = np.count_nonzero(y_true)

    # change int8 to bool after counting true lables
    y_true = y_true.astype('bool')
    y_pred = y_pred.astype('bool')

    # calculate the total number of true positive
    true_poss = np.sum(np.logical_and(y_true, y_pred).astype('float'))

    return {'True_positives': true_poss, 'Samples': label_counts, 'Ratio/W AR': true_poss / label_counts}


plt.plot(y_pred[:, 0], label='1')
plt.legend()
plt.savefig('class1.png')
plt.close()
plt.plot(y_pred[:, 1], label='2')
plt.legend()
plt.savefig('class2.png')
plt.close()
plt.plot(y_pred[:, 2], label='3')
plt.legend()
plt.savefig('class3.png')
plt.close()
plt.plot(y_pred[:, 3], label='4')
plt.legend()
plt.savefig('class4.png')
plt.close()
plt.plot(y_pred[:, 4], label='5')
plt.legend()
plt.savefig('class5.png')
plt.close()

print('W/AR score = ', TP_acc(yt, y_pred))

yt = yt.astype('int')
preds = np.round(y_pred).astype('int')
c1, c2, c3, c4, c5 = yt[:, 0], yt[:, 1], yt[:, 2], yt[:, 3], yt[:, 4]

print(classification_report(c1, preds[:, 0]))
print(classification_report(c2, preds[:, 1]))
print(classification_report(c3, preds[:, 2]))
print(classification_report(c4, preds[:, 3]))
print(classification_report(c5, preds[:, 4]))

f1_1 = f1_score(c1, preds[:, 0], average='macro')
f1_2 = f1_score(c2, preds[:, 1], average='macro')
f1_3 = f1_score(c3, preds[:, 2], average='macro')
f1_4 = f1_score(c4, preds[:, 3], average='macro')
f1_5 = f1_score(c5, preds[:, 4], average='macro')
all_f1 = np.array([f1_1, f1_2, f1_3, f1_4, f1_5])

print('classwise f1: ', all_f1)
print('macro f1 : ', all_f1.mean())
