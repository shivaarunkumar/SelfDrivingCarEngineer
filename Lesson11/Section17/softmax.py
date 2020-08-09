import numpy as np

def softmax(x):
    return np.exp(x)/sum(np.exp(x))


logits = [3.0,1.0,2.0]
print(softmax(logits))
print(softmax([l*10 for l in logits]))
print(softmax([l/10 for l in logits]))