from collections import defaultdict
from utils.tsort import topological_sort_kahn
import numpy as np
from htm.bindings.sdr import SDR
from htm.bindings.algorithms import SpatialPooler, TemporalMemory

from utils.tsort import topological_sort_kahn

LINK_TYPES = ["BU", "TD", "FF_LATERAL"]

class HTMModel:
  def __init__(self):
    self.regions = {}
    self.region_params = {}
    self.links = defaultdict(dict) 

  def add_region(self, name, region_params, class_obj):
    self.region_params[name] = region_params
    self.regions[name] = class_obj(region_params)

  def add_link(self, src_region_name, dest_region_name, type):
    if dest_region_name not in self.links:
      self.links[dest_region_name] = defaultdict(list)
    self.links[dest_region_name][type].append(src_region_name)
  
  def _get_total_cells(self, region_name):
    return self.regions[region_name].total_cells_count()

  def initialize(self):
    for region_name, region_params in self.region_params.items():
      for link_type in self.links.get(region_name, {}):
        match link_type:
          case "BU":
            dest_enc_width = sum(self._get_total_cells(linked_region) for linked_region in self.links[region_name][link_type])
            self.region_params[region_name]["enc"]["encodingWidth"] = dest_enc_width
          case "TD":
            external_predictive_inputs = sum(self._get_total_cells(linked_region) for linked_region in self.links[region_name][link_type])
            self.region_params[region_name]["tm"]["externalPredictiveInputs"] = external_predictive_inputs  
          case "FF_LATERAL":
            raise NotImplementedError("FF_LATERAL not implemented yet")

      self.regions[region_name] = HTMRegion(region_params)
    self.compute_order = self._get_compute_order()

  def region_pairs(self):
    ordered_pairs = []
    for dest_region_name, link_types in self.links.items():
      if "BU" in link_types:
        ordered_pairs.extend((src_region_name, dest_region_name) for src_region_name in link_types["BU"])
    return ordered_pairs

  def _get_compute_order(self):
    region_pairs = self.region_pairs()
    adjacency_dict = {}
    for src_region, dest_region in region_pairs:
      if src_region not in adjacency_dict:
        adjacency_dict[src_region] = []
      if dest_region not in adjacency_dict:
        adjacency_dict[dest_region] = []  # Add regions without outgoing links as dictionary keys
      adjacency_dict[src_region].append(dest_region)
    ordered_regions = topological_sort_kahn(adjacency_dict)
    return ordered_regions

  def compute(self, inputs):
    for region_name in self.compute_order:
      if self.regions[region_name].__class__.__name__ == "HTMInputRegion":
        self.regions[region_name].compute(inputs[region_name])
      else:
        bu_input = []
        td_input = []
        for linked_region, link_type in self.links[region_name]:
          if link_type == "BU":
            bu_input.extend(self.regions[region_name].getActiveColumns().dense)
          elif link_type == "TD":
            td_input.extend(self.regions[region_name].getPredictedColumns().dense)
          elif link_type == "FF_LATERAL":
            raise NotImplementedError("FF_LATERAL not implemented yet")

        bu_sdr = SDR(len(bu_input))
        td_sdr = SDR(len(td_input))
        self.regions[region_name].compute(bu_sdr, True, td_sdr, td_sdr)


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

  def total_cells_count(self):
    return self.sp.getNumColumns() * self.tm.getCellsPerColumn()

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


class HTMInputRegion(HTMRegion):
  def __init__(self, encodingWidth=0):
    self.last_data = SDR((encodingWidth,)) 

  def compute(self, data):
    self.last_data = data
    return data

  def getActiveColumns(self):
    return self.last_data
  
  def getActiveCells(self):
    return self.last_data

  def getPredictiveCells(self):
    return self.last_data

  def total_cells_count(self):
    return self.last_data.dense.size

