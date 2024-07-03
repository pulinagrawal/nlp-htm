import os
import math
import numpy as np
from pytest import param
from tqdm import tqdm

from htm.bindings.sdr import SDR, Metrics
from htm.encoders.rdse import RDSE, RDSE_Parameters
from htm.encoders.date import DateEncoder
from htm.bindings.algorithms import SpatialPooler
from htm.bindings.algorithms import TemporalMemory
from htm.algorithms.anomaly_likelihood import AnomalyLikelihood
from htm.bindings.algorithms import Predictor

from encoder import num_sp, token_ids, tokens
from htm_helpers import HTMRegion
from vectordb import VectorDB, manhattan_distance

_EXAMPLE_DIR = os.path.dirname(os.path.abspath(__file__))
_INPUT_FILE_PATH = os.path.join(_EXAMPLE_DIR, "OIG-small-chip2.txt")

default_parameters = {
  # there are 2 (3) encoders: "value" (RDSE) & "time" (DateTime weekend, timeOfDay)
 'enc': {
      "encodingWidth": 1024,
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

  region1_params = parameters.copy()
  region2_params = parameters.copy()
  region2_params['enc']['encodingWidth'] = region1_params['sp']['columnCount']*region1_params['tm']['cellsPerColumn']
  region1_params['tm']['externalPredictiveInputs'] = region2_params['sp']['columnCount']*region2_params['tm']['cellsPerColumn']

  if verbose:
    import pprint
    print("Parameters:")
    pprint.pprint(parameters, indent=4)
    print("")


  # Read the input file.
  with open(_INPUT_FILE_PATH, "r") as fin:
    lines = fin.readlines()
    records = ''.join(lines)

  # print(records)
  # Make the Encoders.  These will convert input data into binary representations.
  encodingWidth = parameters['enc']["encodingWidth"]
  enc_info = Metrics( [encodingWidth], 999999999 )

  # Make the HTM.  SpatialPooler & TemporalMemory & associated tools.
  region2 = HTMRegion(region2_params)
  region1 = HTMRegion(region1_params)
  print("Total Parameters:", region1.total_params() + region2.total_params())

  sp_info = Metrics(region1.sp.getColumnDimensions(), 999999999)
  tm_info = Metrics([region1.tm.numberOfCells()], 999999999)

  anomaly_history = AnomalyLikelihood(parameters["anomaly"]["period"])

  predictor = Predictor(steps=[1, 5], alpha=parameters["predictor"]['sdrc_alpha'])
  predictor_resolution = 1

  # Iterate through every datum in the dataset, record the inputs & outputs.
  inputs = []
  anomaly = []
  anomalyProb = []
  predictions = {1: [], 5: []}
  vecdb = VectorDB()

  token_nums = token_ids(records[:20000])
  for count, record in tqdm(enumerate(token_nums[:2000])):

    inputs.append(record)

    # Call the encoders to create bit representations for each value.  These are SDR objects.
    tokenBits = num_sp(encodingWidth, 0.02, record)

    # Concatenate all these encodings into one large encoding for Spatial Pooling.
    encoding = SDR(encodingWidth)
    encoding.dense = tokenBits.tolist()

    # Compute the HTM region
    region1.compute(encoding, True, region2.getPredictiveCells(), region2.getPredictiveCells())
    region2.compute(region1.getActiveCells(), learn=True)

    vecdb.add_vector(region1.getActiveColumns().dense, record)

    # Record Encoder, Spatial Pooler & Temporal Memory statistics.
    enc_info.addData( encoding )
    sp_info.addData(region1.getActiveColumns())
    tm_info.addData(region1.getActiveCells().flatten())

    # Predict what will happen, and then train the predictor based on what just happened.
    columns = SDR(len(region1.getActiveColumns().dense))
    columns.sparse = region1.get_columns_from_cells(region1.getPredictiveCells().sparse)
    pred_list = vecdb.search_similar_vectors(columns.dense, k=1, distance_func=manhattan_distance)
    pdf = predictor.infer(region1.getPredictiveCells())
    for n in (1, 5):
      if pdf[n]:
        predictions[n].append(np.argmax(pdf[n]) * predictor_resolution)
      else:
        predictions[n].append(float('nan'))
      if n == 1:
        predictions[n][-1] = pred_list[0][1]

    anomaly.append(region1.tm.anomaly)
    anomalyProb.append(anomaly_history.compute(region1.tm.anomaly))

    predictor.learn(count, region1.getPredictiveCells(), int(record))

  # Print information & statistics about the state of the HTM.
  print("Encoded Input", enc_info)
  print("")
  print("Spatial Pooler Mini-Columns", sp_info)
  print(str(region1.sp))
  print("")
  print("Temporal Memory Cells", tm_info)
  print(str(region1.tm))
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
        accuracy[n] += (inp - val)==0 
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

  return -accuracy[5]


if __name__ == "__main__":
  main()
