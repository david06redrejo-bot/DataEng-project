# Auxiliary file where we will have our functions, as well as complementary items that will make the notebook be cleaner.
import pandas as pd
import matplotlib.pyplot as plt


features = {'numContinuous':('duration_ms', 'danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'), 
            'numDiscrete':('popularity', 'time_signature'), 
            'catNominal':('track_genre', 'key'), 
            'catOrdinal':(),
            'binary':('explicit', 'mode'), 
            'text':('track_id', 'artists', 'album_name', 'track_name')
            }



def elaborateBoxPlots(featureList: tuple | list, data: pd.DataFrame) -> None:
    ### Pass in the tuple of features to plot.
    #
    n = len(featureList)
    fig, axes = plt.subplots((n + 1) // 2, 2, figsize=(25,22))
    for feature, ax in zip(featureList, axes.flatten()):
        
        
        ax.boxplot(data[feature], vert=False, widths=0.6,showmeans=True, flierprops={
            'marker': '|',
            'markerfacecolor': 'm',
            'markeredgecolor': 'm',
            'markersize':3,
            'alpha':0.4
        }
        )
        ax.set_title(feature)
        ax.set_yticks([])


    plt.show()

