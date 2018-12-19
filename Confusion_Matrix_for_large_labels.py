import itertools
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

plt.rcParams["figure.figsize"] = (200,200)

label_true = ['A','B','C','A','B','C','A','D','B','B','A','D','A']
label_predicted = ['A','D','D','A','A','C','A','D','B','B','A','D','B'] 

cm = confusion_matrix(label_true, label_predicted) # SET YOUR CM HERE
np.set_printoptions(precision=2)
print("CM ",cm)
print(type(cm))
test_class = range(0,101,1)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Reds):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    ticks=np.linspace(0, 100,num=101)
    print("Tick",ticks)
    print("Tick_mark",tick_marks)
    plt.xticks(tick_marks, classes, fontsize=3)
    plt.yticks(tick_marks, classes, fontsize=3)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),fontsize=4,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    #plt.tight_layout()
    plt.grid(True)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')



plt.figure()
plot_confusion_matrix(cm, test_class,
                      title='Confusion matrix',normalize=True)
plt.show()
