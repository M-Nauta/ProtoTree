import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

# These are the "Tableau 20" colors as RGB.  
tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),  
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),  
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),  
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),  
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]  

# Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.  
for i in range(len(tableau20)):  
    r, g, b = tableau20[i]  
    tableau20[i] = (r / 255., g / 255., b / 255.)  

tableau_colorblind_own_4 = [(0,107,164),(255,128,14),(137,137,137),(200,82,0)]
# Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.  
for i in range(len(tableau_colorblind_own_4)):  
    r, g, b = tableau_colorblind_own_4[i]  
    tableau_colorblind_own_4[i] = (r / 255., g / 255., b / 255.)  
# Vibrant qualitative colour scheme from https://personal.sron.nl/~pault/
paultol_palette_colorblind = [(0,119,187),(0,153,136),(238,119,51),(204,51,17),(51,187,238),(238,51,119),(187,187,187)]
# Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.  
for i in range(len(paultol_palette_colorblind)):  
    r, g, b = paultol_palette_colorblind[i]  
    paultol_palette_colorblind[i] = (r / 255., g / 255., b / 255.) 

# adapted from http://www.randalolson.com/2014/06/28/how-to-make-beautiful-data-visualizations-in-python-with-matplotlib/
chosen_palette = paultol_palette_colorblind

# You typically want your plot to be ~1.33x wider than tall. This plot is a rare  
# exception because of the number of lines being plotted on it.  
# Common sizes: (10, 7.5) and (12, 9)  
plt.figure(figsize=(12, 6))  

# Remove the plot frame lines. They are unnecessary chartjunk.  
ax = plt.subplot(111)  
ax.spines["top"].set_visible(False)  
# ax.spines["bottom"].set_visible(False)  
ax.spines["right"].set_visible(False)  
# ax.spines["left"].set_visible(False)  

# Ensure that the axis ticks only show up on the bottom and left of the plot.  
# Ticks on the right and top of the plot are generally unnecessary chartjunk.  
ax.get_xaxis().tick_bottom()  
ax.get_yaxis().tick_left()  

# Limit the range of the plot to only where the data is.  
# Avoid unnecessary whitespace.  
plt.ylim(min_ylim, ylim)  
plt.xlim(0, xlim)  

# Make sure your axis ticks are large enough to be easily read.  
# You don't want your viewers squinting to read your plot.  
plt.yticks(range(0, ylim, 10), [str(x) + "%" for x in range(0, ylim, 10)], fontsize=14)  
plt.xticks(range(0,xlim), range(1,xlim+1), fontsize=14)  

# Provide tick lines across the plot to help your viewers trace along  
# the axis ticks. Make sure that the lines are light and small so they  
# don't obscure the primary data lines.  
for y in range(10, ylim, 20):  
    plt.plot(range(0, xlim), [y] * len(range(0, xlim)), "--", lw=0.4, color="black", alpha=0.25)  

# Now that the plot is prepared, it's time to actually plot the data!  

for rank, d in enumerate(datasets):  
    # Plot each line separately with its own color, using the Tableau 20  
    # color set in order. 
    mean = data[data['dataset']==d]['mean'].values[0]
    std = data[data['dataset']==d]['std'].values[0]
    plt.fill_between(range(len(mean)), mean - std,  
                    mean + std, color=chosen_palette[rank], alpha=0.3)  
    plt.plot(range(len(mean)),  
            mean,  
            lw=2.5, color=chosen_palette[rank], marker='o')  
    min_height = data[data['dataset']==d]['min_height'].values[0]
    plt.axvline(x = min_height-1, linewidth=1.7, linestyle = "dotted", color=chosen_palette[rank], alpha=0.9)
    
    ensemble_acc = data[data['dataset']==d]['ensemble_acc'].values[0]
    plt.plot(range(len(ensemble_acc)),  
            ensemble_acc,  
            lw=2.5, color=chosen_palette[rank], marker='d', linestyle = "dashed")  
    # Add a text label to the right end of every line. Most of the code below  
    # is adding specific offsets y position because some labels overlapped.  
    y_pos = mean[-1]-0.9

    # Again, make sure that all labels are large enough to be easily read  
    # by the viewer.  
    plt.text(xlim-0.9, y_pos, d, fontsize=16, color=chosen_palette[rank]) 

custom_lines = [Line2D([0], [0], color="dimgray", linestyle = "solid", lw=2),
                Line2D([0], [0], color="dimgray", linestyle = "dashed", lw=2),
                Line2D([0], [0], color="dimgray", linestyle = "dotted", lw=2)]
ax.legend(custom_lines, ["Single Tree", "Ensemble", "Min. Height"], loc='lower right', fontsize=14, fancybox=True, framealpha=0.5)
# plt.text(xlim-0.9, 50, "Single ProtoTree", fontsize=16, color="grey")  
# plt.text(xlim-0.9, 40, "Ensemble 5 ProtoTrees", fontsize=16, color="grey") 

plt.xlabel("Height of ProtoTree", fontsize=16)  
plt.ylabel("Accuracy" , fontsize=16)
# Remove the tick marks; they are unnecessary with the tick lines we just plotted.  
plt.tick_params(axis="both", which="both", bottom=False, top=False,  
                labelbottom=True, left=False, right=False, labelleft=True)  

plt.show()

# bbox_inches="tight" removes all the extra whitespace on the edges of your plot.  
# plt.savefig("plot_height_accuracy.pdf", bbox_inches="tight")