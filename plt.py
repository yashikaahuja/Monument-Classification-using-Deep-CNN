import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
 
objects = ('RF', 'kNN', 'SVM', 'CNN', 'NB')
y_pos = np.arange(len(objects))
performance = [78.34,51.69,37.02,97.12,43.54]
 
plt.bar(y_pos, performance, align='center', alpha=0.5 , color='g')
plt.xticks(y_pos, objects)
plt.xlabel('Classifiers')
plt.ylabel('Accuracy')
plt.title('A comparison of accuracy of different classifiers')
 
plt.show()