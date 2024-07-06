import numpy as np
from htm.bindings.sdr import SDR
from htm.bindings.algorithms import SpatialPooler, TemporalMemory

class HTMRegion:
  def __init__(self, parameters):
    self.activeColumns = SDR(parameters["sp"]["columnCount"])
    self.activeColumns.dense = np.zeros(parameters["sp"]["columnCount"])
    self.spParams = parameters["sp"]
    self.tmParams = parameters["tm"]
    self.encodingWidth = parameters["enc"]["encodingWidth"]
    self.externalPredictiveInputsActive = 0
    self.externalPredictiveInputsWinners = 0
    self.sp = SpatialPooler(
    inputDimensions=(self.encodingWidth,),
    columnDimensions=(self.spParams["columnCount"],),
    potentialPct=self.spParams["potentialPct"],
    potentialRadius=self.spParams["potentialRadius"],
    globalInhibition=True,
    localAreaDensity=self.spParams["localAreaDensity"],
    synPermInactiveDec=self.spParams["synPermInactiveDec"],
    synPermActiveInc=self.spParams["synPermActiveInc"],
    synPermConnected=self.spParams["synPermConnected"],
    boostStrength=self.spParams["boostStrength"],
    wrapAround=True
    )
    self.tm = TemporalMemory(
    columnDimensions=(self.spParams["columnCount"],),
    cellsPerColumn=self.tmParams["cellsPerColumn"],
    activationThreshold=self.tmParams["activationThreshold"],
    initialPermanence=self.tmParams["initialPerm"],
    connectedPermanence=self.spParams["synPermConnected"],
    minThreshold=self.tmParams["minThreshold"],
    maxNewSynapseCount=self.tmParams["newSynapseCount"],
    permanenceIncrement=self.tmParams["permanenceInc"],
    permanenceDecrement=self.tmParams["permanenceDec"],
    predictedSegmentDecrement=0.0,
    maxSegmentsPerCell=self.tmParams["maxSegmentsPerCell"],
    maxSynapsesPerSegment=self.tmParams["maxSynapsesPerSegment"],
    externalPredictiveInputs=self.tmParams["externalPredictiveInputs"]
    )
  
  def total_params(self):
    cells = self.spParams["columnCount"] * self.tmParams["cellsPerColumn"]
    synapses_per_cell = self.tmParams["maxSegmentsPerCell"] * self.tmParams["maxSynapsesPerSegment"] 
    tm_params_bu = cells * synapses_per_cell
    tm_params_td = self.tmParams["externalPredictiveInputs"] * synapses_per_cell
    sp_params = self.spParams["columnCount"] * self.spParams["potentialRadius"]
    return tm_params_bu + tm_params_td + sp_params

  def compute(self, encoding, learn=True, externalPredictiveInputsActive=0, externalPredictiveInputsWinners=0):
    self.activeColumns = SDR(self.sp.getColumnDimensions())
    self.sp.compute(encoding, True, self.activeColumns)
    if not externalPredictiveInputsWinners and not externalPredictiveInputsActive:
      self.tm.compute(self.activeColumns, learn)
    else:
      self.externalPredictiveInputsActive = externalPredictiveInputsActive
      self.externalPredictiveInputsWinners = externalPredictiveInputsWinners
      self.tm.compute(self.activeColumns, learn, externalPredictiveInputsActive, externalPredictiveInputsWinners)

  def getActiveColumns(self):
    return self.activeColumns

  def getActiveCells(self):
    return self.tm.getActiveCells()

  def getPredictiveCells(self):
    if not self.externalPredictiveInputsWinners and not self.externalPredictiveInputsActive:
      self.tm.activateDendrites(False)
    else:
      self.tm.activateDendrites(False, self.externalPredictiveInputsActive, self.externalPredictiveInputsWinners)
    return self.tm.getPredictiveCells()

  def get_columns_from_cells(self, cells):
    columns = set()
    for cell in cells:
      columns.add(self.tm.columnForCell(cell))
    return list(columns)

  def get_sp_reconstruction(self, active_columns):
    reconstruction = np.zeros(self.sp.getNumInputs(), dtype=np.float32)
    for col in active_columns:
      permanences = np.array(self.sp.getPermanence(col, np.ndarray(1, dtype=np.int32), 0))
      reconstruction += (permanences > self.sp.getSynPermConnected()).astype(int)
    return reconstruction

  def get_predicted_columns(self):
    columns = SDR(len(self.getActiveColumns().dense))
    columns.sparse = self.get_columns_from_cells(self.getPredictiveCells().sparse)
    return columns
