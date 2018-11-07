# BigDEAL Forecasting Competition 2018 - LSTM example

Code for training LSTM Neural Networks for energy load forecasting. The competition was hosted by Dr. Tao Hong in Novemeber of 2018 : http://blog.drhongtao.com/2018/11/leaderboard-for-bfcom2018-qualifying-match.html

## Defining Models

- Define models in `modelLib.py`

- Can hard code parameters in or make them parameteric and set them inside `config.toml`

## Training Models

- run `train.py` after setting the values in `config.toml` to your liking

- models will be saved inside your model directory `modelDir`

- the `config.toml` file used in training will also be copied to the model directory

## Testing Models

- run 'test.py' with the path to the `config.toml` file of the model as the first argument

  ```terminal
  python test.py ./models/LSTM01_001/config.toml
  ```