#%%
from ast import increment_lineno
from IPython import get_ipython
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

x = np.linspace(-100,100,5000)
plt.plot(x,np.exp(x)/np.sum(np.exp(x)))

# %%
import numpy as np

def softmax(L):    
    return np.exp(L)/(np.sum(np.exp(L)))

# %%
Y=[1,0,1,1] 
P=[0.4,0.6,0.1,0.5]
i=0
print(float(Y[i])*P[i] )
print(-1*np.sum([(float(Y[i])*p + (1-float(Y[i]))*(1-p)) for (i,p) in enumerate(P)]))
np.l

# %%

import numpy as np
x= 3*.4+5*.6 -2.2
print(1/(1+np.exp(-1*x)))

# %%

from IPython import get_ipython
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


n_records = 10
n_inputs = 5
n_hidden = 2
weights_input_to_hidden = np.random.normal(0, n_inputs**-0.5, size=(n_inputs, n_hidden))
print(weights_input_to_hidden)


y = np.random.normal(0, n_inputs**-0.5, size=(1, 10000))
print(n_inputs**-0.5)
n, bins, patches = plt.hist(x=y[0], bins='auto', color='#0504aa',
                            alpha=0.7, rwidth=0.85)
# %%
