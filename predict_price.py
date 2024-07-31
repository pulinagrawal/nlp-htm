import os
import math
from matplotlib.pyplot import bar
import numpy as np
import datetime
from tqdm import tqdm

from htm.bindings.sdr import SDR, Metrics
from htm.algorithms.anomaly_likelihood import AnomalyLikelihood
from htm.bindings.algorithms import Predictor
from htm.encoders.rdse import RDSE, RDSE_Parameters 
from htm.encoders.scalar_encoder import ScalarEncoder, ScalarEncoderParameters
from htm.encoders.date import DateEncoder

from encoder import num_sp, token_ids, stringify
from htm_helpers import HTMInputRegion, HTMModel, HTMRegion
from vectordb import VectorDB, manhattan_distance

import yfinance as yf

_EXAMPLE_DIR = os.path.dirname(os.path.abspath(__file__))
_INPUT_FILE_PATH = os.path.join(_EXAMPLE_DIR, "OIG-small-chip2.txt")

default_parameters = {
  # there are 2 (3) encoders: "value" (RDSE) & "time" (DateTime weekend, timeOfDay)
 'enc': {
      "curve" :
          {'activeBits': 3, 'min': -10, 'max': 10, 'size': 30},
      "value" :
         {'resolution': 0.88, 'size': 700, 'sparsity': 0.02},
      "time": 
         {'timeOfDay': (30, 1), 'season': 21, 'dayOfWeek': (24, 3)}
 },
 'predictor': {'sdrc_alpha': 0.1},
 'sp': {'boostStrength': 3.0,
        'columnCount': 1638,
        'localAreaDensity': 0.04395604395604396,
        'potentialRadius': 1024,
        'potentialPct': 0.85,
        'synPermActiveInc': 0.04,
        'synPermConnected': 0.13999999999999999,
        'synPermInactiveDec': 0.006},
 'tm': {'activationThreshold': 17,
        'cellsPerColumn': 7,
        'initialPerm': 0.21,
        'maxSegmentsPerCell': 128,
        'maxSynapsesPerSegment': 64,
        'minThreshold': 10,
        'newSynapseCount': 32,
        'permanenceDec': 0.1,
        'permanenceInc': 0.1,
        'externalPredictiveInputs': 0},
  'anomaly': {'period': 1000},
}


def main(parameters=default_parameters, argv=None, verbose=True):

  if verbose:
    import pprint
    print("Parameters:")
    pprint.pprint(parameters, indent=4)
    print("")


  dateEncoder = DateEncoder(timeOfDay= parameters["enc"]["time"]["timeOfDay"], 
                            dayOfWeek= parameters["enc"]["time"]["dayOfWeek"],
                            season= parameters["enc"]["time"]["season"]) 

  priceEncoderParams            = RDSE_Parameters()
  priceEncoderParams.size       = parameters["enc"]["value"]["size"]
  priceEncoderParams.sparsity   = parameters["enc"]["value"]["sparsity"]
  priceEncoderParams.resolution = parameters["enc"]["value"]["resolution"]
  priceEncoder = RDSE( priceEncoderParams )

  curveEncoderParams = ScalarEncoderParameters()
  curveEncoderParams.activeBits = parameters["enc"]["curve"]["activeBits"]
  curveEncoderParams.minimum    = parameters["enc"]["curve"]["min"]
  curveEncoderParams.maximum    = parameters["enc"]["curve"]["max"]
  curveEncoderParams.size       = parameters["enc"]["curve"]["size"]
  curveEncoder = ScalarEncoder( curveEncoderParams )

  # three curves high-low / close-open, open-high, open-low, close-high, close-low
  encodingWidth = (priceEncoderParams.size + dateEncoder.size + curveEncoder.size*5)
  enc_info = Metrics( [encodingWidth], 999999999 )

  model = HTMModel()
  parameters["enc"]["encodingWidth"] = encodingWidth
  model.add_region("token_region", {"encodingWidth": encodingWidth}, HTMInputRegion)
  model.add_region("region1", parameters, HTMRegion) # Prediction Error .31

  model.add_link("token_region", "region1", "BU")

  model.initialize()

  # Make the HTM.  SpatialPooler & TemporalMemory & associated tools.
  sp_info = Metrics(model["region1"].sp.getColumnDimensions(), 999999999)
  tm_info = Metrics([model["region1"].tm.numberOfCells()], 999999999)

  anomaly_history = AnomalyLikelihood(parameters["anomaly"]["period"])

  predictor = Predictor(steps=[1, 5], alpha=parameters["predictor"]['sdrc_alpha'])
  predictor_resolution = 1

  # Iterate through every datum in the dataset, record the inputs & outputs.
  inputs = []
  anomaly = []
  anomalyProb = []
  predictions = {1: [], 5: []}

  datas = []
  # create a date for each day of the year
  for i in range(1, 367):
    if i <179 or i>206:
      continue
    sdate = datetime.datetime.strptime(f"2024-{i:03d}", "%Y-%j").strftime("%Y-%m-%d")
    edate = datetime.datetime.strptime(f"2024-{i+1:03d}", "%Y-%j").strftime("%Y-%m-%d")
    data = yf.download(tickers="SPY", start=sdate, end=edate, interval="1m")
    datas.append(data)

  import pandas as pd
  data = pd.concat(datas)
  for count, (date, record) in tqdm(enumerate(data.iterrows())):

    dateString = datetime.datetime.strptime(str(date)[:-6], "%Y-%m-%d %H:%M:%S")
    bar_range = record['High'] - record['Low']
    oc = record['Open'] - record['Close']
    oh = record['Open'] - record['High']
    ol = record['Open'] - record['Low']
    ch = record['Close'] - record['High']
    cl = record['Close'] - record['Low']
    price = record['Close']

    # append predictor
    inputs.append(int((oc/bar_range)*10))

    date_bits = dateEncoder.encode(dateString)
    oc_bits = curveEncoder.encode((oc/bar_range)*10)
    oh_bits = curveEncoder.encode((oh/bar_range)*10)
    ol_bits = curveEncoder.encode((ol/bar_range)*10)
    ch_bits = curveEncoder.encode((ch/bar_range)*10)
    cl_bits = curveEncoder.encode((cl/bar_range)*10)
    price_bits = priceEncoder.encode(price)

    # Concatenate all these encodings into one large encoding for Spatial Pooling.
    encoding = SDR(encodingWidth).concatenate([date_bits, oc_bits, oh_bits, ol_bits, ch_bits, cl_bits, price_bits]) 
    enc_info.addData(encoding)

    # Compute the HTM region
    model.compute({"token_region": encoding}, learn=True)

    # Record Encoder, Spatial Pooler & Temporal Memory statistics.
    enc_info.addData( encoding )
    sp_info.addData(model["region1"].getActiveColumns())
    tm_info.addData(model["region1"].getActiveCells().flatten())

    # Predict what will happen, and then train the predictor based on what just happened.
    columns = SDR(len(model["region1"].getActiveColumns().dense))
    columns.sparse = model["region1"].get_columns_from_cells(model["region1"].getPredictiveCells().sparse)
    pdf = predictor.infer(model["region1"].getPredictiveCells())
    for n in (1, 5):
      if pdf[n]:
        predictions[n].append( np.argmax( pdf[n] ) * predictor_resolution )
      else:
        predictions[n].append(float('nan'))


    anomaly.append(model["region1"].tm.anomaly)
    anomalyProb.append(anomaly_history.compute(model["region1"].tm.anomaly))

    predictor.learn(count, model["region1"].getPredictiveCells(), int((oc/bar_range)*10))

  # Print information & statistics about the state of the HTM.
  print("Encoded Input", enc_info)
  print("")
  print("Spatial Pooler Mini-Columns", sp_info)
  print(str(model["region1"].sp))
  print("")
  print("Temporal Memory Cells", tm_info)
  print(str(model["region1"].tm))
  print("")

  # Shift the predictions so that they are aligned with the input they predict.
  for n_steps, pred_list in predictions.items():
    for x in range(n_steps):
        pred_list.insert(0, float('nan'))
        pred_list.pop()

  # Calculate the predictive accuracy, Root-Mean-Squared
  accuracy         = {1: 0, 5: 0}
  accuracy_samples = {1: 0, 5: 0}

  for idx, inp in enumerate(inputs):
    for n in predictions: # For each [N]umber of time steps ahead which was predicted.
      val = predictions[n][ idx ]
      if not math.isnan(val):
        accuracy[n] += (inp - val) ** 2
        accuracy_samples[n] += 1

  for n in sorted(predictions):
    accuracy[n] = (accuracy[n] / accuracy_samples[n]) ** .5
    print("Predictive Error (RMS)", n, "steps ahead:", accuracy[n])

  # Show info about the anomaly (mean & std)
  print("Anomaly Mean", np.mean(anomaly))
  print("Anomaly Std ", np.std(anomaly))

  # Plot the Predictions and Anomalies.
  if verbose:
    try:
      import matplotlib.pyplot as plt
    except:
      print("WARNING: failed to import matplotlib, plots cannot be shown.")
      return -accuracy[5]

    plt.subplot(2,1,1)
    plt.title("Predictions")
    plt.xlabel("Time")
    plt.ylabel("Power Consumption")
    plt.plot(np.arange(len(inputs)), inputs, 'red',
             np.arange(len(inputs)), predictions[1], 'blue',
             np.arange(len(inputs)), predictions[5], 'green',)
    plt.legend(labels=('Input', '1 Step Prediction, Shifted 1 step', '5 Step Prediction, Shifted 5 steps'))

    plt.subplot(2,1,2)
    plt.title("Anomaly Score")
    plt.xlabel("Time")
    plt.ylabel("Power Consumption")
    inputs = np.array(inputs) / max(inputs)
    plt.plot(np.arange(len(inputs)), inputs, 'black',
             np.arange(len(inputs)), anomaly, 'blue',
             np.arange(len(inputs)), anomalyProb, 'red',)
    plt.legend(labels=('Input', 'Instantaneous Anomaly', 'Anomaly Likelihood'))
    plt.show()

  return model["region1"]

def run_model(htm_model, learn, record):
    # Call the encoders to create bit representations for each value.  These are SDR objects.
    tokenBits = num_sp(htm_model.encodingWidth, 0.02, record)

      # Concatenate all these encodings into one large encoding for Spatial Pooling.
    encoding = SDR(htm_model.encodingWidth)
    encoding.dense = tokenBits.tolist()

      # Compute the HTM region
    htm_model.compute(encoding, learn)


if __name__ == "__main__":
  htm_model = main()