#%%
import os
import numpy as np
import matplotlib.pyplot as plt

def import_squared_relative_errors(folder_path):
    squared_relative_errors = []
    
    # Walk through the directory
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file == 'squared_relative_errors.txt':
                file_path = os.path.join(root, file)
                # Read the file and import as an array
                with open(file_path, 'r') as f:
                    errors = np.loadtxt(f)
                    squared_relative_errors.append(errors)
    
    return squared_relative_errors

# Specify the folder path
folder_path = 'real_test'

# Import the squared relative errors
squared_relative_errors = import_squared_relative_errors(folder_path)

sqr1 = squared_relative_errors[1]
sqr2 = squared_relative_errors[0]
sqr3 = squared_relative_errors[2]


# Plot the squared relative errors
epochs = np.arange(0, len(sqr1)) * 100
plt.figure(figsize=(8, 6))
plt.semilogy(epochs, sqr1, label='Regular')
plt.semilogy(epochs, sqr2, label='Non-dim ODEs')
plt.semilogy(epochs, sqr3, label='Non-dim ODEs and SoftAdapt')

plt.figure(figsize=(8, 6))
plt.semilogy(epochs, sqr1, label='Regular')
plt.semilogy(epochs, sqr2, label='Non-dim ODEs')
plt.semilogy(epochs, sqr3, label='Non-dim ODEs and SoftAdapt')
plt.ylim(0, 2000)

plt.show()

plt.figure(figsize=(8, 6))
plt.loglog(epochs, sqr1, label='Regular')
plt.loglog(epochs, sqr2, label='Non-dim ODEs')
plt.loglog(epochs, sqr3, label='Non-dim ODEs and SoftAdapt')
plt.ylim(0, 2000)

plt.show()

#%%
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Main plot
fig, ax = plt.subplots(figsize=(9, 7))
fig.patch.set_facecolor('lightgrey')  # Change the background color of the entire figure
ax.set_facecolor('lightgrey')  # Change the background color of the plot
ax.semilogy(epochs, sqr1, label='Regular', linewidth=3)
ax.semilogy(epochs, sqr2, label='Non-dim ODEs', linewidth=3)
ax.semilogy(epochs, sqr3, label='Non-dim ODEs and SoftAdapt', linewidth=3, color='C4')
ax.set_xlabel('Epoch', fontsize=14)
ax.set_ylabel(r'$ SSRE $', fontsize=14)
ax.set_ylim(0, 5000)

# Inset plot
ax_inset = inset_axes(ax, width="100%", height="100%", bbox_to_anchor=(0.55, 0.47, 0.4, 0.4), bbox_transform=ax.transAxes, loc='upper left')
ax_inset.semilogy(epochs, sqr1, label='Regular', color = 'C0', linewidth=3)
ax_inset.set_ylim(5000, 600000)
ax_inset.xlabels = False
ax_inset.set_facecolor('lightgrey')

ax.legend(loc="upper center", bbox_to_anchor=(0.51, 0.98), ncol=3, fontsize=11.5, facecolor='lightgrey', edgecolor='black')

plt.savefig('plot.svg', format='svg')
plt.show()
# %%
