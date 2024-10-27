import torch
from collections import defaultdict
import random

class TemporalMemory:
    """
    Temporal Memory implementation in PyTorch.
    Conforms to the interface provided by the NuPIC code.
    """

    def __init__(
        self,
        columnDimensions=(2048,),
        cellsPerColumn=32,
        activationThreshold=13,
        initialPermanence=0.21,
        connectedPermanence=0.5,
        minThreshold=10,
        maxNewSynapseCount=20,
        permanenceIncrement=0.1,
        permanenceDecrement=0.1,
        predictedSegmentDecrement=0.0,
        maxSegmentsPerCell=255,
        maxSynapsesPerSegment=255,
        seed=42,
        device='cpu',
        **kwargs
    ):
        """
        Initialize the Temporal Memory.
        """
        # Validate parameters
        if not len(columnDimensions):
            raise ValueError("Number of column dimensions must be greater than 0")

        if cellsPerColumn <= 0:
            raise ValueError("Number of cells per column must be greater than 0")

        if minThreshold > activationThreshold:
            raise ValueError(
                "The min threshold can't be greater than the activation threshold"
            )

        # Save member variables
        self.columnDimensions = columnDimensions
        self.cellsPerColumn = cellsPerColumn
        self.activationThreshold = activationThreshold
        self.initialPermanence = initialPermanence
        self.connectedPermanence = connectedPermanence
        self.minThreshold = minThreshold
        self.maxNewSynapseCount = maxNewSynapseCount
        self.permanenceIncrement = permanenceIncrement
        self.permanenceDecrement = permanenceDecrement
        self.predictedSegmentDecrement = predictedSegmentDecrement
        self.maxSegmentsPerCell = maxSegmentsPerCell
        self.maxSynapsesPerSegment = maxSynapsesPerSegment
        self.seed = seed
        self.device = device

        # Initialize random seed
        if seed is not None:
            torch.manual_seed(seed)
            random.seed(seed)

        # Total number of columns and cells
        self.columnCount = self.numberOfColumns()
        self.totalCells = self.numberOfCells()

        # Initialize cell states
        self._activeCells = torch.zeros(self.totalCells, dtype=torch.bool, device=self.device)
        self._winnerCells = torch.zeros(self.totalCells, dtype=torch.bool, device=self.device)

        # Previous time step states
        self.prevActiveCells = torch.zeros(self.totalCells, dtype=torch.bool, device=self.device)
        self.prevWinnerCells = torch.zeros(self.totalCells, dtype=torch.bool, device=self.device)

        # Initialize segments and synapses
        # For each cell, we'll keep a list of segments
        # Each segment will have a list of synapses
        self.segments = defaultdict(list)  # Key: cell index, Value: list of segments
        # Each segment is represented as a dictionary:
        # {'synapses': dict(presynaptic_cell_index: permanence)}

        # Variables to keep track of active and matching segments
        self._activeSegments = []
        self._matchingSegments = []

        # Predictive cells
        self._predictiveCells = torch.zeros(self.totalCells, dtype=torch.bool, device=self.device)

        # Additional variables
        self.numActiveConnectedSynapsesForSegment = []
        self.numActivePotentialSynapsesForSegment = []

        self.iteration = 0
        self.lastUsedIterationForSegment = []

    # ==============================
    # Main methods
    # ==============================

    def compute(self, activeColumns, learn=True):
        """
        Perform one time step of the Temporal Memory algorithm.

        This method calls activateCells, then calls activateDendrites.
        """
        self.activateCells(activeColumns, learn)
        self.activateDendrites(learn)

    def activateCells(self, activeColumns, learn=True):
        """
        Calculate the active cells, using the current active columns and dendrite
        segments. Grow and reinforce synapses.
        """
        # Save current active cells and winner cells as previous time step states
        self.prevActiveCells = self._activeCells.clone()
        self.prevWinnerCells = self._winnerCells.clone()

        # Reset active and winner cells for the current time step
        self._activeCells = torch.zeros(self.totalCells, dtype=torch.bool, device=self.device)
        self._winnerCells = torch.zeros(self.totalCells, dtype=torch.bool, device=self.device)

        # Convert activeColumns to set for faster lookup
        activeColumnsSet = set(activeColumns)

        # Process columns
        for column in range(self.columnCount):
            colCells = self.cellsForColumn(column)
            if column in activeColumnsSet:
                # Active column
                predictiveCells = []
                for cell in colCells:
                    if self._cellWasPredicted(cell):
                        predictiveCells.append(cell)

                if predictiveCells:
                    # Activate predicted cells
                    for cell in predictiveCells:
                        self._activeCells[cell] = True
                        self._winnerCells[cell] = True

                    if learn:
                        # Reinforce segments that predicted correctly
                        for cell in predictiveCells:
                            segments = self.segments[cell]
                            for segment in segments:
                                if self._segmentWasActive(segment):
                                    self._adaptSegment(
                                        segment,
                                        positiveReinforcement=True
                                    )
                                    # Grow new synapses if needed
                                    numActivePotential = segment.get('numActivePotential', 0)
                                    nGrowDesired = self.maxNewSynapseCount - numActivePotential
                                    if nGrowDesired > 0:
                                        self._growSynapses(segment, nGrowDesired)
                else:
                    # Burst column: activate all cells
                    for cell in colCells:
                        self._activeCells[cell] = True
                    # Choose a winner cell
                    winnerCell = self._getWinnerCell(column)
                    self._winnerCells[winnerCell] = True

                    if learn:
                        # If there are matching segments, reinforce them
                        bestMatchingSegment = self._bestMatchingSegment(winnerCell)
                        if bestMatchingSegment:
                            self._adaptSegment(
                                bestMatchingSegment,
                                positiveReinforcement=True
                            )
                            numActivePotential = bestMatchingSegment.get('numActivePotential', 0)
                            nGrowDesired = self.maxNewSynapseCount - numActivePotential
                            if nGrowDesired > 0:
                                self._growSynapses(bestMatchingSegment, nGrowDesired)
                        else:
                            # Create a new segment
                            newSegment = self._createSegment(winnerCell)
                            nGrowExact = min(self.maxNewSynapseCount, len(self.prevWinnerCells.nonzero()))
                            if nGrowExact > 0:
                                self._growSynapses(newSegment, nGrowExact)
            else:
                # Inactive column
                # Punish segments that predicted this column
                if learn:
                    for cell in colCells:
                        if self._cellWasPredicted(cell):
                            segments = self.segments[cell]
                            for segment in segments:
                                if self._segmentWasMatching(segment):
                                    self._adaptSegment(
                                        segment,
                                        positiveReinforcement=False
                                    )

    def activateDendrites(self, learn=True):
        """
        Calculate dendrite segment activity, using the current active cells.
        """
        self._activeSegments = []
        self._matchingSegments = []

        for cellIndex in range(self.totalCells):
            for segment in self.segments[cellIndex]:
                numActiveConnected = 0
                numActivePotential = 0
                for presynCell, permanence in segment['synapses'].items():
                    # Use prevActiveCells from the previous time step
                    if self.prevActiveCells[presynCell]:
                        if permanence >= self.connectedPermanence:
                            numActiveConnected += 1
                        if permanence >= 0:
                            numActivePotential += 1
                if numActiveConnected >= self.activationThreshold:
                    segment['activeState'] = True
                    self._activeSegments.append(segment)
                else:
                    segment['activeState'] = False

                if numActivePotential >= self.minThreshold:
                    segment['matchingState'] = True
                    segment['numActivePotential'] = numActivePotential
                    self._matchingSegments.append(segment)
                else:
                    segment['matchingState'] = False

                if learn:
                    # Update last used iteration
                    if segment['activeState']:
                        segment['lastUsedIteration'] = self.iteration
        if learn:
            self.iteration += 1

        # Update predictive cells
        self._predictiveCells = torch.zeros(self.totalCells, dtype=torch.bool, device=self.device)
        for segment in self._activeSegments:
            self._predictiveCells[segment['cellIndex']] = True

    def reset(self):
        """
        Indicates the start of a new sequence. Clears any predictions and makes sure
        synapses don't grow to the currently active cells in the next time step.
        """
        self._activeCells = torch.zeros(self.totalCells, dtype=torch.bool, device=self.device)
        self._winnerCells = torch.zeros(self.totalCells, dtype=torch.bool, device=self.device)
        self._activeSegments = []
        self._matchingSegments = []
        self._predictiveCells = torch.zeros(self.totalCells, dtype=torch.bool, device=self.device)

    # ==============================
    # Helper methods
    # ==============================

    def _cellWasPredicted(self, cellIndex):
        """
        Check if a cell was in the predictive state in the previous time step.
        """
        for segment in self.segments[cellIndex]:
            if segment.get('activeState', False):
                return True
        return False

    def _segmentWasActive(self, segment):
        """
        Check if a segment was active in the previous time step.
        """
        return segment.get('activeState', False)

    def _segmentWasMatching(self, segment):
        """
        Check if a segment was matching in the previous time step.
        """
        return segment.get('matchingState', False)

    def _getWinnerCell(self, column):
        """
        Get the winner cell for a bursting column.
        """
        colCells = self.cellsForColumn(column)
        # Find cells with least number of segments
        minSegments = float('inf')
        leastUsedCells = []
        for cell in colCells:
            numSegments = len(self.segments[cell])
            if numSegments < minSegments:
                minSegments = numSegments
                leastUsedCells = [cell]
            elif numSegments == minSegments:
                leastUsedCells.append(cell)
        # Randomly choose one of the least used cells
        return random.choice(leastUsedCells)

    def _bestMatchingSegment(self, cellIndex):
        """
        Get the best matching segment for a cell.
        """
        bestSegment = None
        bestScore = -1
        for segment in self.segments[cellIndex]:
            if segment.get('matchingState', False):
                numActivePotential = segment.get('numActivePotential', 0)
                if numActivePotential > bestScore:
                    bestScore = numActivePotential
                    bestSegment = segment
        return bestSegment

    def _adaptSegment(self, segment, positiveReinforcement):
        """
        Adapt the permanence values of synapses in a segment.
        """
        for presynCell, permanence in segment['synapses'].items():
            # Use prevActiveCells from the previous time step
            if self.prevActiveCells[presynCell]:
                if positiveReinforcement:
                    segment['synapses'][presynCell] = min(
                        permanence + self.permanenceIncrement, 1.0)
                else:
                    segment['synapses'][presynCell] = max(
                        permanence - self.predictedSegmentDecrement, 0.0)
            else:
                if positiveReinforcement:
                    segment['synapses'][presynCell] = max(
                        permanence - self.permanenceDecrement, 0.0)
                # No change if negative reinforcement and presynaptic cell was inactive

    def _growSynapses(self, segment, nDesiredNewSynapses):
        """
        Grow new synapses to winner cells from the previous time step.
        """
        potentialCells = torch.nonzero(self.prevWinnerCells).squeeze().tolist()
        # Exclude cells already connected
        connectedCells = set(segment['synapses'].keys())
        candidates = list(set(potentialCells) - connectedCells)
        random.shuffle(candidates)
        nActual = min(nDesiredNewSynapses, len(candidates))

        # Enforce maxSynapsesPerSegment
        overrun = len(segment['synapses']) + nActual - self.maxSynapsesPerSegment
        if overrun > 0:
            self._destroyMinPermanenceSynapses(segment, overrun)

        for i in range(nActual):
            presynCell = candidates[i]
            segment['synapses'][presynCell] = self.initialPermanence

    def _createSegment(self, cellIndex):
        """
        Create a new segment on the given cell.
        """
        # Enforce maxSegmentsPerCell
        while len(self.segments[cellIndex]) >= self.maxSegmentsPerCell:
            # Remove least recently used segment
            leastUsedSegment = min(
                self.segments[cellIndex],
                key=lambda seg: seg.get('lastUsedIteration', self.iteration)
            )
            self.segments[cellIndex].remove(leastUsedSegment)

        newSegment = {
            'cellIndex': cellIndex,
            'synapses': dict(),
            'activeState': False,
            'matchingState': False,
            'numActivePotential': 0,
            'lastUsedIteration': self.iteration
        }
        self.segments[cellIndex].append(newSegment)
        return newSegment

    def _destroyMinPermanenceSynapses(self, segment, nDestroy):
        """
        Destroy synapses with the lowest permanence.
        """
        synapses = segment['synapses']
        if not synapses:
            return
        sortedSynapses = sorted(synapses.items(), key=lambda item: item[1])
        for i in range(min(nDestroy, len(sortedSynapses))):
            presynCell = sortedSynapses[i][0]
            del synapses[presynCell]

    # ==============================
    # Property decorators
    # ==============================

    @property
    def activeCells(self):
        """
        Returns the indices of the active cells.
        """
        return torch.nonzero(self._activeCells).squeeze().tolist()

    @property
    def predictiveCells(self):
        """
        Returns the indices of the predictive cells.
        """
        return torch.nonzero(self._predictiveCells).squeeze().tolist()

    @property
    def winnerCells(self):
        """
        Returns the indices of the winner cells.
        """
        return torch.nonzero(self._winnerCells).squeeze().tolist()

    @property
    def activeSegments(self):
        """
        Returns the active segments.
        """
        return self._activeSegments

    @property
    def matchingSegments(self):
        """
        Returns the matching segments.
        """
        return self._matchingSegments

    @property
    def cellsPerColumn(self):
        """
        Returns the number of cells per column.
        """
        return self._cellsPerColumn

    @cellsPerColumn.setter
    def cellsPerColumn(self, value):
        self._cellsPerColumn = value

    @property
    def columnDimensions(self):
        """
        Returns the dimensions of the columns in the region.
        """
        return self._columnDimensions

    @columnDimensions.setter
    def columnDimensions(self, value):
        self._columnDimensions = value

    @property
    def activationThreshold(self):
        """
        Returns the activation threshold.
        """
        return self._activationThreshold

    @activationThreshold.setter
    def activationThreshold(self, value):
        self._activationThreshold = value

    @property
    def initialPermanence(self):
        """
        Get the initial permanence.
        """
        return self._initialPermanence

    @initialPermanence.setter
    def initialPermanence(self, value):
        self._initialPermanence = value

    @property
    def minThreshold(self):
        """
        Returns the min threshold.
        """
        return self._minThreshold

    @minThreshold.setter
    def minThreshold(self, value):
        self._minThreshold = value

    @property
    def maxNewSynapseCount(self):
        """
        Returns the max new synapse count.
        """
        return self._maxNewSynapseCount

    @maxNewSynapseCount.setter
    def maxNewSynapseCount(self, value):
        self._maxNewSynapseCount = value

    @property
    def permanenceIncrement(self):
        """
        Get the permanence increment.
        """
        return self._permanenceIncrement

    @permanenceIncrement.setter
    def permanenceIncrement(self, value):
        self._permanenceIncrement = value

    @property
    def permanenceDecrement(self):
        """
        Get the permanence decrement.
        """
        return self._permanenceDecrement

    @permanenceDecrement.setter
    def permanenceDecrement(self, value):
        self._permanenceDecrement = value

    @property
    def predictedSegmentDecrement(self):
        """
        Get the predicted segment decrement.
        """
        return self._predictedSegmentDecrement

    @predictedSegmentDecrement.setter
    def predictedSegmentDecrement(self, value):
        self._predictedSegmentDecrement = value

    @property
    def connectedPermanence(self):
        """
        Get the connected permanence.
        """
        return self._connectedPermanence

    @connectedPermanence.setter
    def connectedPermanence(self, value):
        self._connectedPermanence = value

    @property
    def maxSegmentsPerCell(self):
        """
        Get the maximum number of segments per cell
        """
        return self._maxSegmentsPerCell

    @maxSegmentsPerCell.setter
    def maxSegmentsPerCell(self, value):
        self._maxSegmentsPerCell = value

    @property
    def maxSynapsesPerSegment(self):
        """
        Get the maximum number of synapses per segment.
        """
        return self._maxSynapsesPerSegment

    @maxSynapsesPerSegment.setter
    def maxSynapsesPerSegment(self, value):
        self._maxSynapsesPerSegment = value

    # ==============================
    # Utility methods
    # ==============================

    def columnForCell(self, cell):
        """
        Returns the index of the column that a cell belongs to.
        """
        self._validateCell(cell)
        return int(cell / self.cellsPerColumn)

    def cellsForColumn(self, column):
        """
        Returns the indices of cells that belong to a column.
        """
        self._validateColumn(column)
        start = self.cellsPerColumn * column
        end = start + self.cellsPerColumn
        return list(range(start, end))

    def numberOfColumns(self):
        """
        Returns the number of columns in this layer.
        """
        return int(torch.prod(torch.tensor(self.columnDimensions)).item())

    def numberOfCells(self):
        """
        Returns the number of cells in this layer.
        """
        return self.numberOfColumns() * self.cellsPerColumn

    # Validation methods
    def _validateColumn(self, column):
        """
        Raises an error if column index is invalid.
        """
        if column >= self.numberOfColumns() or column < 0:
            raise IndexError("Invalid column")

    def _validateCell(self, cell):
        """
        Raises an error if cell index is invalid.
        """
        if cell >= self.numberOfCells() or cell < 0:
            raise IndexError("Invalid cell")

    def mapCellsToColumns(self, cells):
        """
        Maps cells to the columns they belong to.
        """
        cellsForColumns = defaultdict(set)
        for cell in cells:
            column = self.columnForCell(cell)
            cellsForColumns[column].add(cell)
        return cellsForColumns

    # Equality methods (__eq__, __ne__)
    def __eq__(self, other):
        """
        Equality operator for TemporalMemory instances.
        Checks if two instances are functionally identical.
        """
        if not isinstance(other, TemporalMemory):
            return False
        if self.columnDimensions != other.columnDimensions:
            return False
        if self.cellsPerColumn != other.cellsPerColumn:
            return False
        if self.activationThreshold != other.activationThreshold:
            return False
        if abs(self.initialPermanence - other.initialPermanence) > 1e-5:
            return False
        if abs(self.connectedPermanence - other.connectedPermanence) > 1e-5:
            return False
        if self.minThreshold != other.minThreshold:
            return False
        if self.maxNewSynapseCount != other.maxNewSynapseCount:
            return False
        if abs(self.permanenceIncrement - other.permanenceIncrement) > 1e-5:
            return False
        if abs(self.permanenceDecrement - other.permanenceDecrement) > 1e-5:
            return False
        if abs(self.predictedSegmentDecrement - other.predictedSegmentDecrement) > 1e-5:
            return False
        if self.maxSegmentsPerCell != other.maxSegmentsPerCell:
            return False
        if self.maxSynapsesPerSegment != other.maxSynapsesPerSegment:
            return False
        # Note: Comparing segments and synapses may require deep comparison
        return True

    def __ne__(self, other):
        """
        Non-equality operator for TemporalMemory instances.
        """
        return not self.__eq__(other)

    # ==============================
    # Supporting functions
    # ==============================

    @staticmethod
    def getCellIndices(cells):
        """
        Returns the indices of the cells passed in.
        """
        return [cell for cell in cells]

    @staticmethod
    def getCellIndex(cell):
        """
        Returns the index of the cell.
        """
        return cell


# Example usage

# Parameters
columnDimensions = (2048,)
cellsPerColumn = 13 

# Initialize Temporal Memory

from spatial_pooler import timer

tm = TemporalMemory(
    columnDimensions=columnDimensions,
    cellsPerColumn=cellsPerColumn,
    activationThreshold=13,
    initialPermanence=0.21,
    connectedPermanence=0.5,
    minThreshold=10,
    maxNewSynapseCount=20,
    permanenceIncrement=0.1,
    permanenceDecrement=0.1,
    predictedSegmentDecrement=0.01,
    maxSegmentsPerCell=255,
    maxSynapsesPerSegment=255,
    seed=42,
    device='cpu'
)

@timer
def execute(tm):
  # Simulate input over time
  time_steps = 10
  for t in range(time_steps):
      # Generate random active columns
      activeColumns = random.sample(range(tm.numberOfColumns()), 40)
      # Compute active cells
      tm.compute(activeColumns)
      # Access state information
  
for i in range(20):
  execute(tm)