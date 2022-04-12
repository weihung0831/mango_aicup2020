from vgg19_reg import KerasTrainer
import losses
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, f1_score

tf.config.experimental.set_memory_growth = True
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True
sess = tf.compat.v1.Session(config=config)

df = pd.read_csv('Test_mangoXYWH.csv')
Final_test = df.values
print(len(Final_test))

topK = 5
prec = tf.keras.metrics.Precision(thresholds=0.5, top_k=topK)
rec = tf.keras.metrics.Recall(thresholds=0.5, top_k=topK)

lnl = KerasTrainer(name='resnet_reg',
                   num_classes=5,
                   target_img_size=224,
                   batch_size=32,
                   augment=['rotate', 'flip', 'blur', 'contrast', 'brightness'],
                   # brightness [-25,25] is applied to each image during training
                   pretrain_lr=1e-5,
                   reg_lr=1e-6,
                   opt='Adam',  # 'SGD'
                   loss=[losses.normalized_macro_bce],
                   metrics=[losses.macro_f1, prec, rec],
                   validation=0.20,
                   pretrain_epoch=50,
                   epochs=50, )

lnl.load_weights('models/resnet_reg_vgg19Post_Epoch_073-85.310.h5')  # your model here

y_pred = lnl.predict_test(Final_test)
print(y_pred)
y_pred = np.round(y_pred)  # threshold everything by 0.5

# predictions = np.column_stack([Final_test, y_pred])
predictions = np.column_stack([Final_test[:, 0], y_pred])

df = pd.DataFrame(predictions)
df.columns = ['image_id', 'D1', 'D2', 'D3', 'D4', 'D5']
df.to_csv('Test_upload.csv', index=False)
