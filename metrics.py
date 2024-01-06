import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def confusion_matrix(y_true, y_pred, num_classes):
    """
    Calculate the confusion matrix.

    :param y_true: List of true labels.
    :param y_pred: List of predicted labels.
    :param num_classes: Total number of classes.
    :return: Confusion matrix.
    """
    # Convert labels to integers
    unique_labels = np.unique(np.concatenate([y_true, y_pred]))
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}

    y_true = np.array([label_to_index[label] for label in y_true], dtype=int)
    y_pred = np.array([label_to_index[label] for label in y_pred], dtype=int)

    conf_mat = np.zeros((num_classes, num_classes), dtype=int)

    for true_label, pred_label in zip(y_true, y_pred):
        conf_mat[true_label, pred_label] += 1

    return conf_mat

def plot_confusion_matrix(conf_mat, classes):
    """
    Affiche la matrice de confusion.

    :param conf_mat: Matrice de confusion.
    :param classes: Liste des classes (noms) dans l'ordre.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Check if conf_mat is a 2D array
    if len(conf_mat.shape) == 2:
        sns.heatmap(conf_mat, annot=True, fmt=".3f", cmap='Blues', xticklabels=classes, yticklabels=classes, ax=ax)
    else:
        # If conf_mat is a 1D array, convert it to a 2D array for compatibility
        conf_mat = np.array([[conf_mat]])
        sns.heatmap(conf_mat, annot=True, fmt=".3f", cmap='Blues', xticklabels=classes, yticklabels=classes, ax=ax)

    ax.set_title('Matrice de confusion')
    ax.set_xlabel('Classe Prédite')
    ax.set_ylabel('Classe Réelle')
    return fig


#-----------------------------------------metrics-----------------------------------------#

def EXACTITUDE(conf_mat):
    """
    Calcule l'exactitude.

    :param conf_mat: Matrice de confusion.
    :return: Exactitude.
    """
    return np.sum(np.diag(conf_mat)) / np.sum(conf_mat)

def SPECIFICITE(conf_mat):
    """
    Calcule la spécificité.

    :param conf_mat: Matrice de confusion.
    :return: Total Spécificité.
    """

    # Spécificité pour chaque classe
    spec = np.diag(conf_mat) / np.sum(conf_mat, axis=1)

    # Spécificité totale
    return np.nanmean(spec)


def PRECISION(conf_mat):
    """
    Calcule la précision.

    :param conf_mat: Matrice de confusion.
    :return: Total Précision.
    """
    prec = np.diag(conf_mat) / np.sum(conf_mat, axis=0)
    mean_precision = np.nanmean(prec)
    return prec , mean_precision
    

def RAPPEL(conf_mat):
    """
    Calcule le rappel.

    :param conf_mat: Matrice de confusion.
    :return: Total Rappel.
    """
    rec = np.diag(conf_mat) / np.sum(conf_mat, axis=1)
    mean_recall =np.nanmean(rec)
    return rec , mean_recall


def F1_score(conf_mat):
    """
    Calculate F1-score.

    :param conf_mat: Confusion matrix.
    :return: F1-score for each class and total F1-score.
    """
    prec, prec_mean = PRECISION(conf_mat)
    rec, rec_mean = RAPPEL(conf_mat)
    
    f1 = 2 * (prec * rec) / (prec + rec)
    #get the mean if there is not Nan in the array
    f1_mean = np.nanmean(f1)
    
    return f1_mean