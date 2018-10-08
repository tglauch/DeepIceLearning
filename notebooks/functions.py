import numpy
#import plot_functions_mk as pfmk
from fancy_plot import *
# from plot_conf import *
import itertools
import copy

def retag_classify(ClassLabel):
    class_scheme = {0:0, 1:0, 2:1, 3:1, 4:1, 5:2, 6:2, 7:0, 8:1, 9:0}
    ClassLabelRetag = copy.deepcopy(ClassLabel)
    for key in class_scheme.keys():
        ClassLabelRetag[ClassLabel == key] = class_scheme[key]
    return ClassLabelRetag

def retag_classify_startTrack(ClassLabel):
    class_scheme = {0:0, 1:0, 2:1, 3:3, 4:1, 5:2, 6:2, 7:0, 8:1, 9:0}
    ClassLabelRetag = copy.deepcopy(ClassLabel)
    for key in class_scheme.keys():
        ClassLabelRetag[ClassLabel == key] = class_scheme[key]
    return ClassLabelRetag

def summary_plot_binary_target(pred_target, MC_target, classes, quantity, binnumber):

    # masking the lists to the labels
    E_for_class1 = quantity[MC_target == 0]
    E_for_class2 = quantity[MC_target == 1]

    # calculating values for the confusion matrix
    absolute, xe,ye = np.histogram2d(pred_target, MC_target, bins=(len(classes),len(classes)))
    # that means that the prediction has to correspond to a event type 
    res_true = absolute/np.sum(absolute,axis=0)
    # normalization if the absolute values, so that the sum over PREDICTION is 1
    # that means that a event hast to be predicted as one type 
    res_pred = absolute/np.sum(absolute,axis=1).reshape(-1,1)

    # PLOT
    fig = plt.figure(figsize=(16, 16))
    ax1 = plt.subplot2grid((3, 2), (0, 0))
    ax1 = acc_vs_quantity_plot(pred_target, MC_target, 0, quantity,
                               "Accuracy vs. Energy for class {}".format(classes[0]),
                               binnumber, ax1)

    ax2 = plt.subplot2grid((3, 2), (0, 1))
    ax2 = acc_vs_quantity_plot(pred_target, MC_target, 1, quantity,
                               "Accuracy vs. Energy for class {}".format(classes[1]),
                               binnumber, ax2)

    ax3 = plt.subplot2grid((3, 2), (1, 0))
    bins = np.linspace(np.min(np.log10(E_for_class1)), np.max(np.log10(E_for_class1)), binnumber)
    valsAll = np.histogram(np.log10(E_for_class1), bins=bins)[0]
    ax3.semilogy(bins[:-1], valsAll, drawstyle = 'steps-mid')
    ax3.set_title("Statistic of {} Events vs. Energy".format(classes[0]), fontsize=16)
    ax3.set_xlabel(r'$\log_{10}$(E) [GeV]', fontsize=16)
    ax3.set_ylabel('amount of events', fontsize=16)

    ax4 = plt.subplot2grid((3, 2), (1, 1))
    bins = np.linspace(np.min(np.log10(E_for_class2)), np.max(np.log10(E_for_class2)), binnumber)
    valsAll = np.histogram(np.log10(E_for_class2), bins=bins)[0]
    ax4.semilogy(bins[:-1], valsAll, drawstyle = 'steps-mid')
    ax4.set_title("Statistic of {} Events vs. Energy".format(classes[1]), fontsize=16)
    ax4.set_xlabel(r'$\log_{10}$(E) [GeV]', fontsize=16)
    ax4.set_ylabel('amount of events', fontsize=16)

    ax5 = plt.subplot2grid((3, 2), (2, 0))
    plot_confusion_matrix(res_true, classes=classes, title='Confusion matrix normalized on MCTruth')

    ax6 = plt.subplot2grid((3, 2), (2, 1))
    plot_confusion_matrix(res_pred, classes=classes, title='Confusion matrix normalized on PREDICTION')

def acc_vs_energy_plot_perType(pred, true, energy, title, binnumber):

    was_NN_right = [] #was_NN_right: Mask with 1 if prediction was right, if not 0
    for i in xrange(0,len(pred)):
        if np.argmax(pred[i]) == true[i]:
            was_NN_right.append(1)
        else:
            was_NN_right.append(0)

    bins = np.linspace(np.min(np.log10(energy)), np.max(np.log10(energy)), binnumber)
    valsTrue = np.histogram(np.log10(energy), weights=was_NN_right, bins=bins)[0]
    valsAll = np.histogram(np.log10(energy), bins=bins)[0]
    acc = 1.*valsTrue/valsAll

    plt.plot(bins[:-1], acc, "x")
    plt.title(title, fontsize=18)

def plot_confusion_matrix(cm, classes,title="", thresh=0.2, cmap=plt.cm.Blues):
    """
    This function plots the confusion matrix.
    """

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=20)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=18)
    plt.yticks(tick_marks, classes, fontsize=18)

    thresh = thresh
    for i, j in itertools.product(range(len(classes)), range(len(classes))):
        plt.text(j, i, '{:.2f}'.format(cm[i, j]),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black", fontsize=14)

    plt.ylabel('Predicted label', fontsize=18)
    plt.xlabel('True label', fontsize=18)


def acc_vs_quantity_plot(pred, true_label, true, quantity, title, bins, ax, weights=1, add_mask=[]):
    
    if isinstance(bins, int):
        bins = np.linspace(np.min(quantity), np.max(quantity), bins)
    if add_mask == []:
        add_mask = np.array([True]*len(pred))
    
    mask_label = (true_label == true) & add_mask
    mask_label_pred = (true_label == true) & (pred==true_label) & add_mask
    valsTrue = np.histogram(quantity[mask_label_pred], bins=bins, weights=weights[mask_label_pred])[0]
    valsAll = np.histogram(quantity[mask_label], bins=bins, weights=weights[mask_label])[0]
    acc = 1.*valsTrue/valsAll
    acc[valsAll==0]=0
    ax.plot(bins[:-1], acc, drawstyle='steps-mid')
    
    mask_label = (pred == true) & add_mask
    mask_label_pred = (pred == true) & (pred==true_label) & add_mask
    valsTrue = np.histogram(quantity[mask_label_pred], bins=bins, weights=weights[mask_label_pred])[0]
    valsAll = np.histogram(quantity[mask_label], bins=bins, weights=weights[mask_label])[0]
    acc = 1 - 1.*valsTrue/valsAll
    acc[valsAll==0]=0
    ax.plot(bins[:-1], acc, drawstyle='steps-mid', color='blue')    
    ax.set_ylim(0.0,1.1)
    ax.set_title(title)
    return ax

def acc_loss_plot(acc_train, loss_train, acc_val, loss_val, title):
    x = np.linspace(1, len(acc_train), num=len(acc_train), endpoint=True)
    plt.plot(x, acc_train, color="#0099cc", label="acc of the training set")
    plt.plot(x, loss_train, color='#35bcf8', label="loss of the training set")
    plt.plot(x, acc_val, color='#6c1ba1', label="acc of the validation set")
    plt.plot(x, loss_val, color='#af27cd', label="loss of the validation set")
    plt.title(title, fontsize=18)
    #plt.set_ylim(0., 1.8)
    plt.legend(bbox_to_anchor=(0.59, 0.97), loc=2, borderaxespad=0.)
    plt.ylabel('loss & percentage', fontsize=16)
    plt.xlabel("epochs", fontsize=16)

def acc_of_classifier_vs_x(pred, true, x, title, xlabel, ylabel, binnumber):

    was_classifier_right = [] #was_NN_right: Mask with 1 if prediction was right, if not 0
    for i in xrange(0,len(pred)):
        if np.argmax(pred[i]) == true[i]:
            was_classifier_right.append(1)
        else:
            was_classifier_right.append(0)
            
    
    bins = np.linspace(np.min(x), np.max(x), binnumber)
    valsTrue = np.histogram(x, weights=was_classifier_right, bins=bins)[0]
    valsAll = np.histogram(x, bins=bins)[0]
    acc = 1.*valsTrue/valsAll

    plt.plot(bins[:-1], acc, "x")
    plt.title(title, fontsize=18)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    return
  
def barchart_classes(classes, values, title, ylabel, xlabel):
    fig, ax = newfig(0.9)
    y_pos = np.arange(len(classes))
    couleur = ['#0000FF', '#0099FF', '#14b3e6', '#1480e6']
    total = sum(values)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    
    for i in xrange(len(classes)):
        if 1.0*values[i]/total < 0.2:
            plt.text(y_pos[i], values[i]/2, '{:.0f}'.format(1.*values[i]), horizontalalignment="center",\
                    verticalalignment='bottom', rotation='0', color="black")
        else:
            plt.text(y_pos[i], values[i]/2, '{:.0f}'.format(1.*values[i]), horizontalalignment="center",\
                    verticalalignment='bottom', rotation='0', color="white")
            
    plt.bar(y_pos, values, align='center', alpha=1.0, color=couleur)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation='45')
    plt.show()
    

def confusion_process(input_X, true_X, classes, weights=[]):
    xedges = [-0.5, 0.5, 1.5, 2.5, 3.5]
    yedges = [-0.5, 0.5, 1.5, 2.5, 3.5]
    if weights == []:
        weights = np.ones(len(input_X))
    absolute, xe,ye = np.histogram2d(input_X, true_X, bins=(xedges, yedges), weights = weights)
    res_true = absolute/np.sum(absolute,axis=0)
    # normalization if the absolute values, so that the sum over PREDICTION is 1
    # that means that a event hast to be predicted as one type 
    res_pred = absolute/np.sum(absolute,axis=1).reshape(-1,1)
    
    fig = plt.figure(figsize=(25, 8))
    ax13 = plt.subplot2grid((1, 3), (0, 0))
    plot_confusion_matrix(res_true, classes=classes,
                          title='Confusion matrix normalized on TRUTH', thresh=0.75)

    ax14 = plt.subplot2grid((1, 3), (0, 1))
    plot_confusion_matrix(res_pred, classes=classes,
                          title='Confusion matrix normalized on PREDICTION', thresh=0.75)

    ax14 = plt.subplot2grid((1, 3), (0, 2))
    plot_confusion_matrix(absolute, classes=classes,
                          title='Confusion matrix ABSOLUTE', thresh=20000.)
    return fig

    
def performence_target(pred, true):
    NN_correct = 0
    for i in xrange(len(true)):
        if np.argmax(pred[i]) == true[i]:
            NN_correct += 1
    return 1.*NN_correct/len(true)*100


def acc_vs_hitDOMs_plot(pred, true, hitDOMs, title, binnumber):

    was_NN_right = [] #was_NN_right: Mask with 1 if prediction was right, if not 0
    for i in xrange(0,len(pred)):
        if np.argmax(pred[i]) == true:
            was_NN_right.append(1)
        else:
            was_NN_right.append(0)


    bins = np.linspace(np.min(hitDOMs), np.max(hitDOMs), binnumber)
    valsTrue = np.histogram(hitDOMs, weights=was_NN_right, bins=bins)[0]
    valsAll = np.histogram(hitDOMs, bins=bins)[0]
    acc = 1.*valsTrue/valsAll


    plt.plot(bins[:-1], acc, "x")
    plt.title(title, fontsize=18)
    plt.ylabel('accuracy', fontsize=16)
    plt.xlabel('number of hit DOMs', fontsize=16)
