# Number-of-Vehicles-Prediction-in-Besiktas-Using-Neural-Networks

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://numberofvehiclesprediction.herokuapp.com)

**App Link** : https://numberofvehiclesprediction.herokuapp.com

**Traffic Data** : https://data.ibb.gov.tr/dataset/saatlik-trafik-yogunluk-veri-seti  **(2020.01.01 - 2020.12.27)**

**Weather Data** : https://www.worldweatheronline.com

It is a little project that aims to predict the number of vehicles in different places of Besiktas,Istanbul.

<img src="capture.gif" width="900" height="500" />

## Coordinates Used

| No        | Latitude                | Longitute               |
|-----------|-------------------------|-------------------------|
|     1     |     41.0421752929688    |     29.0093994140625    |
|     2     |     41.0476684570312    |     29.0093994140625    |
|     3     |     41.0366821289062    |     28.9984130859375    |
|     4     |     41.0531616210938    |     29.0093994140625    |
|     5     |     41.0586547851562    |     29.0093994140625    |
|     6     |     41.0476684570312    |     28.9874267578125    |
|     7     |     41.0421752929688    |     28.9874267578125    |
|     8     |     41.0366821289062    |     28.9874267578125    |
|     9     |     41.0311889648438    |     28.9874267578125    |
|     10    |     41.0586547851562    |     28.9984130859375    |
|     11    |     41.0641479492188    |     29.0093994140625    |
|     12    |     41.0476684570312    |     29.0203857421875    |
|     13    |     41.0421752929688    |     28.9984130859375    |
|     14    |     41.0476684570312    |     28.9984130859375    |
|     15    |     41.0531616210938    |     28.9984130859375    |
|     16    |     41.0641479492188    |     28.9984130859375    |

## Libraries
```
streamlit
tensorflow-cpu
keras
folium
pandas
numpy
sklearn
matplotlib
seaborn
plotly
streamlit-folium
wwo-hist
```
Slug size was more than 500 MB so I used tensorflow-cpu to decrease it. Maybe I can deploy the project with Docker later.

**runtime.txt**
```
python-3.7.10
```


**License**
[MIT](https://choosealicense.com/licenses/mit/)
