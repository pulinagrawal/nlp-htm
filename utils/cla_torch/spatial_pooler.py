import torch
import torch.nn as nn

class SpatialPooler:
    def __init__(
        self,
        input_size,
        column_count=2048,
        potential_pct=0.5,
        potential_radius=None,
        stimulus_threshold=0,
        boost_strength=0.0,
        syn_perm_connected=0.2,
        syn_perm_active_inc=0.03,
        syn_perm_inactive_dec=0.015,
        min_overlap_duty_cycle=0.001,
        duty_cycle_period=1000,
        num_active_columns_per_inh_area=40,
        seed=None,
        device='cpu'
    ):
        """
        Initialize the Spatial Pooler.
        """
        self.input_size = input_size
        self.column_count = column_count
        self.device = device

        if seed is not None:
            torch.manual_seed(seed)

        # Initialize parameters
        self.potential_pct = potential_pct
        self.potential_radius = potential_radius if potential_radius is not None else input_size
        self.stimulus_threshold = stimulus_threshold
        self.boost_strength = boost_strength
        self.syn_perm_connected = syn_perm_connected
        self.syn_perm_active_inc = syn_perm_active_inc
        self.syn_perm_inactive_dec = syn_perm_inactive_dec
        self.min_overlap_duty_cycle = min_overlap_duty_cycle
        self.duty_cycle_period = duty_cycle_period
        self.num_active_columns_per_inh_area = num_active_columns_per_inh_area

        # Initialize data structures
        self._init_columns()
        self._init_permanences()

        # Duty cycles and boosts
        self.active_duty_cycles = torch.zeros(self.column_count, device=self.device)
        self.overlap_duty_cycles = torch.zeros(self.column_count, device=self.device)
        self._boosts = torch.ones(self.column_count, device=self.device)

        # Inhibition radius (starts as potential_radius)
        self.inhibition_radius = self.potential_radius

        # State information
        self._active_columns = None
        self._overlaps = None

    def _init_columns(self):
        """
        Initialize potential synapses for each column.
        """
        # Fixed number of potential synapses per column
        self.potential_synapse_count = int(self.potential_pct * self.input_size)

        # For each column, randomly select potential synapses
        self.potential_synapses = torch.zeros(
            (self.column_count, self.potential_synapse_count), dtype=torch.long, device=self.device
        )

        for c in range(self.column_count):
            # Potential synapses indices for column c
            potential_indices = torch.randperm(self.input_size, device=self.device)[:self.potential_synapse_count]
            self.potential_synapses[c] = potential_indices

    def _init_permanences(self):
        """
        Initialize permanence values for the potential synapses.
        """
        # Initialize permanences around the connected threshold
        self._permanences = torch.rand(
            (self.column_count, self.potential_synapse_count), device=self.device
        ) * self.syn_perm_connected * 2

        # Determine connected synapses (boolean mask)
        self.connected_mask = self._permanences >= self.syn_perm_connected

    def compute(self, input_vector, learn=True):
        """
        Compute the active columns given the input vector.
        """
        # Ensure input_vector is on the correct device
        input_vector = input_vector.to(self.device)

        # Phase 2: Compute the overlap with the current input for each column
        # Get the input values for each synapse
        synapse_inputs = input_vector[self.potential_synapses]

        # Determine active synapses (connected and input is active)
        active_synapses = self.connected_mask & synapse_inputs.bool()

        # Count active synapses per column
        overlaps = active_synapses.sum(dim=1).float()

        # Apply boosting
        overlaps *= self._boosts

        # Store overlaps for state information
        self._overlaps = overlaps.clone()

        # Phase 3: Apply inhibition to determine winning columns
        # Global inhibition (for simplicity)
        if self.num_active_columns_per_inh_area >= self.column_count:
            min_local_activity = self.stimulus_threshold
        else:
            sorted_overlaps, _ = torch.sort(overlaps, descending=True)
            min_local_activity = sorted_overlaps[self.num_active_columns_per_inh_area - 1]

        # Active columns after inhibition
        active_columns = torch.nonzero(
            (overlaps > self.stimulus_threshold) & (overlaps >= min_local_activity)
        ).squeeze()

        # Store active columns for state information
        self._active_columns = active_columns.clone()

        # Phase 4: Learning
        if learn:
            self._update_permanences(active_columns, synapse_inputs)
            self._update_duty_cycles(active_columns)
            self._update_boosts()

        return active_columns

    def _update_permanences(self, active_columns, synapse_inputs):
        """
        Update synapse permanences for active columns.
        """
        # Create masks for active columns
        active_columns_mask = torch.zeros(self.column_count, dtype=torch.bool, device=self.device)
        active_columns_mask[active_columns] = True

        # Mask for synapses in active columns
        synapse_active_cols = active_columns_mask.unsqueeze(1).expand_as(self._permanences)

        # Synapse inputs for active columns
        synapse_inputs_active = synapse_inputs[synapse_active_cols].view(-1, self.potential_synapse_count)
        permanences_active = self._permanences[synapse_active_cols].view(-1, self.potential_synapse_count)

        # Update permanences
        increase_mask = synapse_inputs_active.bool()
        decrease_mask = ~increase_mask

        # Apply increments and decrements
        permanences_active[increase_mask] += self.syn_perm_active_inc
        permanences_active[decrease_mask] -= self.syn_perm_inactive_dec

        # Clamp permanences
        permanences_active.clamp_(0.0, 1.0)

        # Update the main permanence tensor
        self._permanences[synapse_active_cols] = permanences_active.view(-1)

        # Update connected synapses mask
        self.connected_mask = self._permanences >= self.syn_perm_connected

    def _update_duty_cycles(self, active_columns):
        """
        Update the active and overlap duty cycles for columns.
        """
        # Update active duty cycles
        activity = torch.zeros(self.column_count, device=self.device)
        activity[active_columns] = 1.0
        decay = (self.duty_cycle_period - 1) / self.duty_cycle_period
        self.active_duty_cycles = self.active_duty_cycles * decay + activity / self.duty_cycle_period

        # Update overlap duty cycles
        overlap = (self._permanences >= self.syn_perm_connected).sum(dim=1).float()
        overlap = (overlap > self.stimulus_threshold).float()
        self.overlap_duty_cycles = self.overlap_duty_cycles * decay + overlap / self.duty_cycle_period

        # Boost permanences for columns with low overlap duty cycle
        below_min_duty = self.overlap_duty_cycles < self.min_overlap_duty_cycle
        if below_min_duty.any():
            # Increase permanences
            self._permanences[below_min_duty] += 0.1 * self.syn_perm_connected
            self._permanences.clamp_(0.0, 1.0)
            # Update connected synapses mask
            self.connected_mask = self._permanences >= self.syn_perm_connected

    def _update_boosts(self):
        """
        Update the boost factors for columns based on their duty cycles.
        """
        if self.boost_strength == 0.0:
            return

        mean_duty_cycle = self.active_duty_cycles.mean()
        duty_cycle_ratio = self.active_duty_cycles / (mean_duty_cycle + 1e-6)
        # Exponential boosting function
        self._boosts = torch.exp(-self.boost_strength * (duty_cycle_ratio - 1.0))

    def add_columns(self, num_new_columns):
        """
        Add new columns to the Spatial Pooler during runtime.
        """
        # Update column count
        new_column_indices = torch.arange(self.column_count, self.column_count + num_new_columns, device=self.device)
        self.column_count += num_new_columns

        # Update potential synapses
        new_potential_synapses = torch.zeros(
            (num_new_columns, self.potential_synapse_count), dtype=torch.long, device=self.device
        )
        for idx, c in enumerate(new_column_indices):
            potential_indices = torch.randperm(self.input_size, device=self.device)[:self.potential_synapse_count]
            new_potential_synapses[idx] = potential_indices

        self.potential_synapses = torch.cat([self.potential_synapses, new_potential_synapses], dim=0)

        # Initialize permanences for new columns
        new_permanences = torch.rand(
            (num_new_columns, self.potential_synapse_count), device=self.device
        ) * self.syn_perm_connected * 2

        self._permanences = torch.cat([self._permanences, new_permanences], dim=0)

        # Update connected synapses mask
        new_connected_mask = new_permanences >= self.syn_perm_connected
        self.connected_mask = torch.cat([self.connected_mask, new_connected_mask], dim=0)

        # Update duty cycles and boosts
        self.active_duty_cycles = torch.cat([self.active_duty_cycles, torch.zeros(num_new_columns, device=self.device)])
        self.overlap_duty_cycles = torch.cat([self.overlap_duty_cycles, torch.zeros(num_new_columns, device=self.device)])
        self._boosts = torch.cat([self._boosts, torch.ones(num_new_columns, device=self.device)])

        # Note: Inhibition radius remains unchanged

    # New methods to access state information
    @property
    def active_columns(self):
        """
        Get the indices of the active columns from the last compute cycle.
        """
        return self._active_columns

    @property
    def overlaps(self):
        """
        Get the overlaps of all columns from the last compute cycle.
        """
        return self._overlaps

    @property
    def boosts(self):
        """
        Get the boost factors of all columns.
        """
        return self._boosts

    @property
    def permanences(self):
        """
        Get the permanences of all columns.
        """
        return self._permanences

# Example usage:

# Initialize Spatial Pooler
sp = SpatialPooler(
    input_size=1000,
    column_count=2048,
    potential_pct=0.8,
    stimulus_threshold=2,
    boost_strength=1.0,
    device='cpu'
)

# Generate a random input vector
input_vector = torch.randint(0, 2, (1000,), device='cpu')

# Compute active columns
active_columns = sp.compute(input_vector)

# Access state information
print("Active Columns:", sp.active_columns)
print("Overlaps:", sp.overlaps)
print("Boost Factors:", sp.boosts)
print("Permanences Shape:", sp.permanences.shape)

# Example usage:
import time

def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        print(f'Function {func.__name__} took {end_time - start_time:.6f} seconds to run')
        return result
    return wrapper

@timer
def execute(sp):
  # Generate a random input vector
  input_vector = torch.randint(0, 2, (1000,), device='cpu')

  # Compute active columns
  active_columns = sp.compute(input_vector)

  # Add new columns during runtime
  sp.add_columns(10)

  # Recompute with the same input
  active_columns = sp.compute(input_vector)

execute(sp)