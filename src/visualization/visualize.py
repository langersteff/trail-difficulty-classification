import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from matplotlib.pyplot import figure


class MtbVisualizer:

    @staticmethod
    def plot_results(X, y, sample_size):
        figure(num=None, figsize=(15, 5), dpi=80, facecolor='w', edgecolor='k')
        X = np.concatenate(X, axis=0)
        plt.plot(X[:, 0])
        plt.plot(X[:, 1])
        plt.plot(X[:, 2])

        for i in range(0, y.shape[0]):
            difficulty = y[i]
            if difficulty == 0:
                color = 'blue'
            elif difficulty == 1:
                color = 'red'
            elif difficulty == 2:
                color = 'black'
            elif difficulty == 3:
                color = 'yellow'
            elif difficulty == 4:
                color = 'white'
            elif difficulty == 5:
                color = 'orange'
            else:
                continue

            plt.axvspan(i * sample_size, (i + 1) * sample_size, color=color, alpha=0.2)

        plt.show()

    @staticmethod
    def print_confusion_matrix(y, y_pred):
        labels = [0, 1, 2]
        cm = confusion_matrix(y, y_pred, labels)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(cm)
        plt.title('Confusion matrix of the classifier')
        fig.colorbar(cax)
        ax.set_xticklabels([''] + labels)
        ax.set_yticklabels([''] + labels)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()
