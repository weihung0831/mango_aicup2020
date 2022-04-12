from tensorflow.keras import backend as K
import tensorflow as tf


def L_DMI(y_true, y_pred):
    U = tf.keras.backend.dot(tf.transpose(y_pred), y_true)
    return -1.0 * tf.math.log(tf.dtypes.cast(tf.math.abs(tf.linalg.det(U)), tf.float32) + 1e-3)


def cross_entropy(y_true, y_pred):
    return K.categorical_crossentropy(y_true, y_pred)

def symmetric_cross_entropy(alpha, beta):
    def loss(y_true, y_pred):
        y_true_1 = y_true
        y_pred_1 = y_pred

        y_true_2 = y_true
        y_pred_2 = y_pred

        
        return (alpha*macro_bce(y_true_1 ,y_pred_1) + 
                                            beta*macro_bce(y_pred_2 ,y_true_2))#original axis=-1)
    return loss


def normalized_macro_bce(y, y_hat):
    """Compute the macro binary cross-entropy on a batch of observations (sum across all labels).
    https://github.com/ashrefm/multi-label-soft-f1/blob/master/utils.py
    Args:
        y (int32 Tensor): targets array of shape (BATCH_SIZE, N_LABELS)      
        y_hat (float32 Tensor): probability matrix from forward propagation of shape (BATCH_SIZE, N_LABELS)
        
    Returns:
        cost (scalar Tensor): value of the cost function for the batch
    """
    # Convert the target array to float32
    y = tf.cast(y, tf.float32)
    # Implement cross entropy loss for each observation and label
    cross_entropy = - y * tf.math.log(tf.maximum(y_hat, 1e-3)) - (1-y) * tf.math.log(tf.maximum(1-y_hat, 1e-3))
    # Sum all binary cross entropy losses over the whole batch for each label then normalize by sum of predictions
    cost = tf.reduce_sum(cross_entropy, axis=0)/tf.reduce_sum(y_hat, axis = 0)
    # Average all binary cross entropy losses over labels within the batch
    cost = tf.reduce_sum(cost)
    return cost


def macro_bce_sum(y, y_hat):
    """Compute the macro binary cross-entropy on a batch of observations (average across all labels).
    https://github.com/ashrefm/multi-label-soft-f1/blob/master/utils.py
    Args:
        y (int32 Tensor): targets array of shape (BATCH_SIZE, N_LABELS)      
        y_hat (float32 Tensor): probability matrix from forward propagation of shape (BATCH_SIZE, N_LABELS)
        
    Returns:
        cost (scalar Tensor): value of the cost function for the batch
    """
    # Convert the target array to float32
    y = tf.cast(y, tf.float32)
    # Implement cross entropy loss for each observation and label
    cross_entropy = - y * tf.math.log(tf.maximum(y_hat, 1e-3)) - (1-y) * tf.math.log(tf.maximum(1-y_hat, 1e-3))
    # Average all binary cross entropy losses over the whole batch for each label
    cost = tf.reduce_sum(cross_entropy, axis=0)
    # Average all binary cross entropy losses over labels within the batch
    cost = tf.reduce_sum(cost)
    return cost


def active_passive_loss(alpha, beta):
    def loss(y_true, y_pred):
        y_true_1 = y_true
        y_pred_1 = y_pred

        y_true_2 = y_true
        y_pred_2 = y_pred

        y_pred_1 = tf.clip_by_value(y_pred_1, 1e-3, 0.9)
        y_pred_2 = tf.clip_by_value(y_pred_2, 1e-3, 0.9)
        
        y_true_1 = tf.clip_by_value(y_true_1, 1e-3, 1.0)
        y_true_2 = tf.clip_by_value(y_true_2, 1e-3, 1.0)
        
        return alpha*normalized_macro_bce(y_true_1, y_pred_1) + beta*macro_bce_sum(y_pred_2 ,y_true_2)
    return loss

def active_passive_normalized(alpha, beta):
    def loss(y_true, y_pred):
        y_true_1 = y_true
        y_pred_1 = y_pred

        y_true_2 = y_true
        y_pred_2 = y_pred

        y_pred_1 = tf.clip_by_value(y_pred_1, 1e-3, 0.9)
        y_pred_2 = tf.clip_by_value(y_pred_2, 1e-3, 0.9)
        
        y_true_1 = tf.clip_by_value(y_true_1, 1e-3, 1.0)
        y_true_2 = tf.clip_by_value(y_true_2, 1e-3, 1.0)
        
        return alpha*normalized_macro_bce(y_true_1, y_pred_1) + beta*normalized_macro_bce(y_pred_2 ,y_true_2)
    return loss






def macro_f1(y, y_hat):
    """Compute the macro F1-score on a batch of observations (average F1 across labels)
    
    Args:
        y (int32 Tensor): labels array of shape (BATCH_SIZE, N_LABELS)
        y_hat (float32 Tensor): probability matrix from forward propagation of shape (BATCH_SIZE, N_LABELS)
        thresh: probability value above which we predict positive
        
    Returns:
        macro_f1 (scalar Tensor): value of macro F1 for the batch
    """
    thresh=0.5
    y_pred = tf.cast(tf.greater(y_hat, thresh), tf.float32)
    tp = tf.cast(tf.math.count_nonzero(y_pred * y, axis=0), tf.float32)
    fp = tf.cast(tf.math.count_nonzero(y_pred * (1 - y), axis=0), tf.float32)
    fn = tf.cast(tf.math.count_nonzero((1 - y_pred) * y, axis=0), tf.float32)
    f1 = 2*tp / (2*tp + fn + fp + 1e-9)
    macro_f1 = tf.reduce_mean(f1)
    return macro_f1
  
 
def macro_soft_f1(y, y_hat):
    """Compute the macro soft F1-score as a cost.
    Average (1 - soft-F1) across all labels.
    Use probability values instead of binary predictions.
    
    Args:
        y (int32 Tensor): targets array of shape (BATCH_SIZE, N_LABELS)
        y_hat (float32 Tensor): probability matrix of shape (BATCH_SIZE, N_LABELS)
        
    Returns:
        cost (scalar Tensor): value of the cost function for the batch
    """
    
    y = tf.cast(y, tf.float32)
    y_hat = tf.cast(y_hat, tf.float32)
    tp = tf.reduce_sum(y_hat * y, axis=0)
    fp = tf.reduce_sum(y_hat * (1 - y), axis=0)
    fn = tf.reduce_sum((1 - y_hat) * y, axis=0)
    soft_f1 = 2*tp / (2*tp + fn + fp + 1e-3)
    cost = 1 - soft_f1 # reduce 1 - soft-f1 in order to increase soft-f1
    macro_cost = tf.reduce_mean(cost) # average on all labels
    
    return macro_cost
 
def macro_double_soft_f1(y, y_hat):
    """Compute the macro soft F1-score as a cost (average 1 - soft-F1 across all labels).
    Use probability values instead of binary predictions.
    This version uses the computation of soft-F1 for both positive and negative class for each label.
    https://towardsdatascience.com/the-unknown-benefits-of-using-a-soft-f1-loss-in-classification-systems-753902c0105d
    Args:
        y (int32 Tensor): targets array of shape (BATCH_SIZE, N_LABELS)
        y_hat (float32 Tensor): probability matrix from forward propagation of shape (BATCH_SIZE, N_LABELS)
        
    Returns:
        cost (scalar Tensor): value of the cost function for the batch
    """
    y = tf.cast(y, tf.float32)
    y_hat = tf.cast(y_hat, tf.float32)
    tp = tf.reduce_sum(y_hat * y, axis=0)
    fp = tf.reduce_sum(y_hat * (1 - y), axis=0)
    fn = tf.reduce_sum((1 - y_hat) * y, axis=0)
    tn = tf.reduce_sum((1 - y_hat) * (1 - y), axis=0)
    soft_f1_class1 = 2*tp / (2*tp + fn + fp + 1e-3)
    soft_f1_class0 = 2*tn / (2*tn + fn + fp + 1e-3)
    cost_class1 = 1 - soft_f1_class1 # reduce 1 - soft-f1_class1 in order to increase soft-f1 on class 1
    cost_class0 = 1 - soft_f1_class0 # reduce 1 - soft-f1_class0 in order to increase soft-f1 on class 0
    cost = 0.5 * (cost_class1 + cost_class0) # take into account both class 1 and class 0
    macro_cost = tf.reduce_mean(cost) # average on all labels
    return macro_cost
 
def macro_bce(y, y_hat):
    """Compute the macro binary cross-entropy on a batch of observations (average across all labels).
    https://github.com/ashrefm/multi-label-soft-f1/blob/master/utils.py
    Args:
        y (int32 Tensor): targets array of shape (BATCH_SIZE, N_LABELS)      
        y_hat (float32 Tensor): probability matrix from forward propagation of shape (BATCH_SIZE, N_LABELS)
        
    Returns:
        cost (scalar Tensor): value of the cost function for the batch
    """
    # Convert the target array to float32
    y = tf.cast(y, tf.float32)
    # Implement cross entropy loss for each observation and label
    cross_entropy = - y * tf.math.log(tf.maximum(y_hat, 1e-3)) - (1-y) * tf.math.log(tf.maximum(1-y_hat, 1e-3))
    # Average all binary cross entropy losses over the whole batch for each label
    cost = tf.reduce_mean(cross_entropy, axis=0)
    # Average all binary cross entropy losses over labels within the batch
    cost = tf.reduce_mean(cost)
    return cost






class MultilabelTruePositives(tf.keras.metrics.Metric):
 
    def __init__(self,name="MLP_True_positive",return_value = 'sum', **kwargs):
        super(MultilabelTruePositives, self).__init__(name=name, **kwargs)
 
        # self.batch_size = batch_size
        # self.num_classes = num_classes    
        self.return_value = return_value
    
        self.cat_true_positives = self.add_weight(name="ctp", initializer="zeros")
        self.samples = self.add_weight(name="samples", initializer="zeros")
        # self.WAR = self.add_weight(name="samples", initializer="zeros")
    def update_state(self, y_true, y_pred,sample_weight=None):     
        # convert all values more than 0.5 to 1
        y_true = tf.round(y_true)
        y_pred = tf.round(y_pred)
 
        # cast float32 to int8
        y_true = tf.cast(K.flatten(y_true), 'int8')
        y_pred = tf.cast(K.flatten(y_pred), 'int8')
 
        # count non zero labels total number of true labels
        label_counts = tf.cast(tf.math.count_nonzero(y_true), 'float32')
 
        # change int8 to bool after counting true lables
        y_true = tf.cast(y_true, 'bool')
        y_pred = tf.cast(y_pred, 'bool')
 
        #c alculate the total number of true positive
        true_poss = K.sum(K.cast((tf.math.logical_and(y_true, y_pred)), dtype=tf.float32))
 
        # update the sample numper
        self.samples.assign_add(label_counts)
        # update the true positive 
        self.cat_true_positives.assign_add(true_poss)
        
 
    def result(self):
        if self.return_value == 'sum':
            return self.cat_true_positives
        elif self.return_value == 'sample': 
            return self.samples
        elif self.return_value == 'percent':
            return tf.math.divide(self.cat_true_positives, self.samples)




class MultilabelF1(tf.keras.metrics.Metric):
 
    def __init__(self,name="MACRO_F1_all",classes=5, **kwargs):
        super(MultilabelF1, self).__init__(name=name, **kwargs)
        
        self.true_p = tf.Variable(name="t_p",shape=(classes,),initial_value= [0 for i in range(classes)], dtype = tf.float32)
        self.false_p = tf.Variable(name="f_p",shape=(classes,),initial_value= [0 for i in range(classes)], dtype = tf.float32)
        self.false_n = tf.Variable(name="f_n",shape=(classes,),initial_value= [0 for i in range(classes)], dtype = tf.float32)
        
    def update_state(self, y_true, y_pred,sample_weight=None):     
        # convert all values more than 0.5 to 1
        y_true = tf.round(y_true)
        y_pred = tf.round(y_pred)
 
        
        tp = tf.cast(tf.math.count_nonzero(y_pred * y_true, axis=0), tf.float32)
        fp = tf.cast(tf.math.count_nonzero(y_pred * (1 - y_true), axis=0), tf.float32)
        fn = tf.cast(tf.math.count_nonzero((1 - y_pred) * y_true, axis=0), tf.float32)
        # update the true positive 
        try:
            self.true_p.assign_add(tp)
            self.false_p.assign_add(fp)
            self.false_n.assign_add(fn)
        except:
            print('failed to assign weights')
        
 
    def result(self):
        return tf.reduce_mean(2*self.true_p / (2*self.true_p + self.false_n + self.false_p + 1e-9))


