import numpy as np


class BatchSampler:
    def __init__(self, batch_size, data, labels):
        self.batch_size = batch_size
        self.data = data
        self.labels = labels
        self.dim = len(data)

    def sample_batch(self):
        # Potential Overlapping samples
        x = np.random.randint(0, self.dim, self.batch_size)
        return_data = np.array([])
        return_labels = np.array([])
        for i in range(self.batch_size):
            if i == 0:
                return_data = self.data[x[i]]
            else:
                return_data = np.vstack((return_data, [self.data[x[i]]]))
            return_labels = np.append(return_labels, self.labels[x[i]])
        return return_data, return_labels
