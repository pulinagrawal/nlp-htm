import csv
import datetime
import numpy as np
import math

from pytest import param
from tqdm import tqdm
from pathlib import Path
from htm.bindings.sdr import SDR, Metrics
from htm.encoders.rdse import RDSE, RDSE_Parameters
from htm.encoders.date import DateEncoder
from htm.algorithms.anomaly_likelihood import AnomalyLikelihood
from htm.bindings.algorithms import Predictor

from utils.htm_helpers import HTMInputRegion, HTMModel, HTMRegion

_EXAMPLE_DIR = Path(__file__).resolve().parent
_INPUT_FILE_PATH = _EXAMPLE_DIR / 'datasets' / 'gymdata.csv'

default_parameters = {
  # there are 2 (3) encoders: "value" (RDSE) & "time" (DateTime weekend, timeOfDay)
 'enc': {
      "encodingWidth": 1024,
      "value" :
         {'resolution': 0.88, 'size': 700, 'sparsity': 0.02},
      "time": 
         {'timeOfDay': (30, 1), 'weekend': 21}
 },
 'predictor': {'sdrc_alpha': 0.1},
 'sp': {'model': 'nupic',
        'boostStrength': 3.0,
        'columnCount': 1638,
        'localAreaDensity': 0.04395604395604396,
        'potentialRadius': 1024,
        'potentialPct': 0.85,
        'synPermActiveInc': 0.04,
        'synPermConnected': 0.13999999999999999,
        'synPermInactiveDec': 0.006},
 'tm': {'activationThreshold': 17,
        'cellsPerColumn': 13,
        'initialPerm': 0.21,
        'maxSegmentsPerCell': 128,
        'maxSynapsesPerSegment': 64,
        'minThreshold': 10,
        'newSynapseCount': 32,
        'permanenceDec': 0.1,
        'permanenceInc': 0.1,
        'synPermConnected': 0.13999999999999999,
        'externalPredictiveInputs': 0},
  'anomaly': {'period': 1000},
}

def main(model_size=None, parameters=default_parameters, argv=None, verbose=True):
  if verbose:
    import pprint
    print("Parameters:")
    pprint.pprint(parameters, indent=4)
    print("")

  # Read the input file.
  records = []
  with open(_INPUT_FILE_PATH, "r") as fin:
    reader = csv.reader(fin)
    headers = next(reader)
    next(reader)
    next(reader)
    for record in reader:
      records.append(record)

  # Make the Encoders.  These will convert input data into binary representations.
  dateEncoder = DateEncoder(timeOfDay= parameters["enc"]["time"]["timeOfDay"], 
                            weekend  = parameters["enc"]["time"]["weekend"]) 
  
  scalarEncoderParams            = RDSE_Parameters()
  scalarEncoderParams.size       = parameters["enc"]["value"]["size"]
  scalarEncoderParams.sparsity   = parameters["enc"]["value"]["sparsity"]
  scalarEncoderParams.resolution = parameters["enc"]["value"]["resolution"]
  scalarEncoder = RDSE( scalarEncoderParams )
  encodingWidth = (dateEncoder.size + scalarEncoder.size)
  enc_info = Metrics( [encodingWidth], 999999999 )
  parameters["enc"]["encodingWidth"] = encodingWidth
  
  if model_size == 1: 
    model = HTMModel()
    model.add_region("token_region", {"encodingWidth": encodingWidth}, HTMInputRegion)
    model.add_region("region1", parameters, HTMRegion)
    # model.add_region("region2", parameters, HTMRegion)

    model.add_link("token_region", "region1", "BU")
    # model.add_link("region1", "region2", "BU")
    # model.add_link("region2", "region1", "TD")
  else:
    model = HTMModel()
    model.add_region("token_region", {"encodingWidth": encodingWidth}, HTMInputRegion)
    model.add_region("region1", parameters, HTMRegion)
    model.add_region("region2", parameters, HTMRegion)

    model.add_link("token_region", "region1", "BU")
    model.add_link("region1", "region2", "BU")
    model.add_link("region2", "region1", "TD")

  model.initialize()

  # Make the HTM.  SpatialPooler & TemporalMemory & associated tools.
  sp_info = Metrics(model["region1"].sp.getColumnDimensions(), 999999999)
  tm_info = Metrics([model["region1"].tm.numberOfCells()], 999999999)

  anomaly_history = AnomalyLikelihood(parameters["anomaly"]["period"])

  predictor = Predictor( steps=[1, 5], alpha=parameters["predictor"]['sdrc_alpha'] )
  predictor_resolution = 1

  # Iterate through every datum in the dataset, record the inputs & outputs.
  inputs      = []
  anomaly     = []
  anomalyProb = []
  predictions = {1: [], 5: []}
  for count, record in tqdm(enumerate(records)):

    # Convert date string into Python date object.
    dateString = datetime.datetime.strptime(record[0], "%m/%d/%y %H:%M")
    # Convert data value string into float.
    consumption = float(record[1])
    inputs.append( consumption )

    # Call the encoders to create bit representations for each value.  These are SDR objects.
    dateBits        = dateEncoder.encode(dateString)
    consumptionBits = scalarEncoder.encode(consumption)

    # Concatenate all these encodings into one large encoding for Spatial Pooling.
    encoding = SDR( encodingWidth ).concatenate([consumptionBits, dateBits])
        # Compute the HTM region
    model.compute({"token_region": encoding}, learn=True)

    # Record Encoder, Spatial Pooler & Temporal Memory statistics.``
    enc_info.addData( encoding )
    sp_info.addData(model["region1"].getActiveColumns())
    tm_info.addData(model["region1"].getActiveCells().flatten())

    # Create an SDR to represent active columns, This will be populated by the
    # compute method below. It must have the same dimensions as the Spatial Pooler.
    # Predict what will happen, and then train the predictor based on what just happened.
    pdf = predictor.infer(model["region1"].getPredictiveCells())
    for n in (1, 5):
      if pdf[n]:
        predictions[n].append( np.argmax( pdf[n] ) * predictor_resolution )
      else:
        predictions[n].append(float('nan'))

    anomaly.append(model["region1"].tm.anomaly )
    anomalyProb.append( anomaly_history.compute(model["region1"].tm.anomaly) )

    predictor.learn(count, model["region1"].tm.getPredictiveCells(), int(consumption / predictor_resolution))


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
      if not math.isnan(val) and inp != 0:
        accuracy[n] += ((inp - val)/inp) ** 2
        accuracy_samples[n] += 1
  for n in sorted(predictions):
    accuracy[n] = (accuracy[n] / accuracy_samples[n]) ** .5
    print("Normalized Predictive Error (RMS)", n, "steps ahead:", accuracy[n])

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

  return -accuracy[1]


if __name__ == "__main__":
  main()
