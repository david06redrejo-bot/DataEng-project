# Auxiliary file where we will have our functions, as well as complementary items that will make the notebook be cleaner.

features = {'numContinuous':('duration_ms', 'danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'), 
            'numDiscrete':('popularity', 'time_signature'), 
            'catNominal':('track_genre', 'key'), 
            'catOrdinal':(),
            'binary':('explicit', 'mode'), 
            'text':('track_id', 'artists', 'album_name', 'track_name')
            }

