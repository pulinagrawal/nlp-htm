import unittest
import os, csv
import datetime
import numpy as np
from pathlib import Path
from collections import defaultdict
from htm.bindings.sdr import SDR
from htm.encoders.rdse import RDSE, RDSE_Parameters
from htm.encoders.date import DateEncoder
from utils.htm_helpers import HTMModel, HTMInputRegion, HTMRegion  # Assuming these classes are defined in predict_text.py

SPARSEY_SP_PARAMS = {
    "sp": {
        "model": "sparsey",
        "autosize_grid": True,
        "grid_layout": "hexagonal",
        "num_macs": 16,
        "num_cms_per_mac": 10,
        "num_neurons_per_cm": 100,
        "mac_grid_num_rows": 4,
        "mac_grid_num_cols": 4,
        "mac_receptive_field_size": 3,
        "prev_layer_num_cms_per_mac": 1,
        "prev_layer_num_neurons_per_cm": 1462,
        "prev_layer_mac_grid_num_rows": 1,
        "prev_layer_mac_grid_num_cols": 1,
        "prev_layer_num_macs": 1,
        "prev_layer_grid_layout": "hexagonal",
        "layer_index": 1,
        "sigmoid_phi": 5.0,
        "sigmoid_lambda": 28,
        "saturation_threshold": 0.5,
        "permanence_steps": 10,
        "permanence_convexity": 0.3,
        "activation_threshold_min": 0.1,
        "activation_threshold_max": 1,
        "min_familiarity": 0.2,
        "sigmoid_chi": 2.5,
        "device": "cpu",
    }
}


SPARSEY_SP_PARAMS['sp']['columnCount'] = SPARSEY_SP_PARAMS['sp']['num_macs'] * SPARSEY_SP_PARAMS['sp']['num_cms_per_mac'] * SPARSEY_SP_PARAMS['sp']['num_neurons_per_cm']

NUPIC_SP_PARAMS = {
    "sp": {
        "model": "nupic",
        "boostStrength": 3.0,
        "columnCount": 1638,
        "localAreaDensity": 0.04395604395604396,
        "potentialRadius": 1024,
        "potentialPct": 0.85,
        "synPermActiveInc": 0.04,
        "synPermConnected": 0.13999999999999999,
        "synPermInactiveDec": 0.006,
    },
}

def any_different(arrays):
    """
    Check if any numpy array in the list is different from the others.
    
    Parameters:
    arrays (list of np.ndarray): List of one-dimensional numpy arrays.
    
    Returns:
    bool: True if there is any inconsistency, False otherwise.
    """
    if not arrays:
        return False  # No arrays to compare
    
    # Use the first array as the reference
    reference_array = arrays[0]
    
    for array in arrays[1:]:
        if not np.array_equal(reference_array, array):
            return True  # Found an inconsistency
    
    return False

class TestRepresentationalInconsistencyHTMModel(unittest.TestCase):
    def __init__(self, methodname, astest=True) -> None:
      super().__init__(methodName=methodname)
      self.astest = astest

    def setUp(self, encodingWidth=1024):
      self.default_parameters = {
        # there are 2 (3) encoders: "value" (RDSE) & "time" (DateTime weekend, timeOfDay)
        'enc': {
              "encodingWidth": encodingWidth,
        },
        'predictor': {'sdrc_alpha': 0.1},
        'sp': { 'model': 'sparsey',
                'boostStrength': 3.0,
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
                'synPermConnected': 0.139,
                'externalPredictiveInputs': 0},
          'anomaly': {'period': 1000},
      }
      self.default_parameters.update(SPARSEY_SP_PARAMS)

      # Initialize the HTM model
      self.model = HTMModel()
      
      self.model.add_region(
          'token_region',
          {'encodingWidth': self.default_parameters['enc']['encodingWidth']},
          HTMInputRegion,
      )
      self.model.add_region('region1', self.default_parameters, HTMRegion)
      self.model.add_link('token_region', 'region1', 'BU')

      self.model.initialize()

    def test_real_data(self):
      parameters = {
        # there are 2 (3) encoders: "value" (RDSE) & "time" (DateTime weekend, timeOfDay)
        'enc': {
              'value' :
                {'resolution': 0.88, 'size': 700, 'sparsity': 0.02},
              'time': 
                {'timeOfDay': (30, 1), 'weekend': 21}
        },
      }

      _EXAMPLE_DIR = Path(__file__).resolve().parent.parent
      _INPUT_FILE_PATH = os.path.join(_EXAMPLE_DIR, 'gymdata.csv')
      # Feed input data to the model
      records = []
      with open(_INPUT_FILE_PATH, 'r') as fin:
        reader = csv.reader(fin)
        headers = next(reader)
        next(reader)
        next(reader)
        for record in reader:
          records.append(record)

      # Make the Encoders.  These will convert input data into binary representations.
      dateEncoder = DateEncoder(timeOfDay= parameters['enc']['time']['timeOfDay'], 
                                weekend  = parameters['enc']['time']['weekend']) 
      
      scalarEncoderParams            = RDSE_Parameters()
      scalarEncoderParams.size       = parameters['enc']['value']['size']
      scalarEncoderParams.sparsity   = parameters['enc']['value']['sparsity']
      scalarEncoderParams.resolution = parameters['enc']['value']['resolution']
      scalarEncoder = RDSE( scalarEncoderParams )
      encodingWidth = (dateEncoder.size + scalarEncoder.size)
      self.setUp(encodingWidth)

      inputs = []
      representations_dictionary = defaultdict(list)
      import random
      random.shuffle(records)
      for count, record in enumerate(records[:200]):

        # Convert date string into Python date object.
        dateString = datetime.datetime.strptime(record[0], "%m/%d/%y %H:%M")
        # Convert data value string into float.
        consumption = float(record[1])

        # Call the encoders to create bit representations for each value.  These are SDR objects.
        dateBits        = dateEncoder.encode(dateString)
        consumptionBits = scalarEncoder.encode(consumption)

        # Concatenate all these encodings into one large encoding for Spatial Pooling.
        encoding = SDR( encodingWidth ).concatenate([consumptionBits, dateBits])

        self.model.compute({'token_region': encoding}, learn=True)

        current_representation = self.model['region1'].getActiveColumns()
        repr_dict_key = str(encoding.sparse)
        if repr_dict_key in representations_dictionary:
          if current_representation not in [x for _, x in representations_dictionary[repr_dict_key]]:
            representations_dictionary[repr_dict_key].append((encoding, current_representation))
        else:
          representations_dictionary[repr_dict_key].append((encoding, current_representation))

      # Check for inconsistencies in the representations
      # This is a placeholder for your actual inconsistency check logic
      assert_flag = False
      for key, value in representations_dictionary.items():
        self.assertFalse(any_different([x.sparse for x, _ in value]))
        if len(value) > 1:
          print(f"Found {len(value)} representations for the same input data: {key}")
          for i, (encoding, representation) in enumerate(value):
            print(f"Representation {i+1}: {representation.dense}")
            print(f"Encoding: {encoding.dense}")
          print("")
          assert_flag = True

      if self.astest:
        self.assertFalse(assert_flag)
      else:
        return representations_dictionary

if __name__ == '__main__':
    unittest.main()
