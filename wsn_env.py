import numpy as np
import gymnasium as gym
from gymnasium import spaces

class WSNEnv(gym.Env):
    """
    A pure Python custom environment for the WSN Module 1 simulation.
    This class simulates the network physics and state changes.
    """
    def __init__(self):
        super(WSNEnv, self).__init__()

        # --- Constants ---
        self.NUM_NODES = 100
        self.SENSOR_NODES = 97
        self.INITIAL_ENERGY = 0.5  # Joules
        self.STATE_SIZE = 5
        self.ACTION_SIZE = 4
        self._max_steps = 200

        # --- First-Order Radio Model Parameters ---
        self.E_ELEC = 50e-9
        self.E_FS = 10e-12
        self.E_MP = 0.0013e-12
        self.D0 = np.sqrt(self.E_FS / self.E_MP)
        self.PACKET_BITS = 100 * 8

        # --- Gym Spaces ---
        self.action_space = spaces.Discrete(self.ACTION_SIZE)
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.STATE_SIZE,), dtype=np.float32)
        
        # Will be initialized in reset()
        self.nodes = None
        self.node_positions = None
        self.current_ch_id = None
        self._total_steps = 0

    def _consume_energy(self, node_id, bits, distance):
        if self.nodes[node_id]['residual_energy'] <= 0: return
        cost = self.E_ELEC * bits
        if distance > 0:
            cost += self.E_FS * bits * (distance ** 2) if distance < self.D0 else self.E_MP * bits * (distance ** 4)
        self.nodes[node_id]['residual_energy'] -= cost

    def _update_dynamic_threshold(self, node_id, gamma=0.1):
        node = self.nodes[node_id]
        node['ewma_mu'] = (1 - gamma) * node['ewma_mu'] + gamma * node['sensed_value']
        variance = (1 - gamma) * node['ewma_sigma_sq'] + gamma * ((node['sensed_value'] - node['ewma_mu']) ** 2)
        node['ewma_sigma_sq'] = variance
        node['dynamic_threshold'] = node['ewma_mu'] + 3 * np.sqrt(max(0, variance))

    def _elect_cluster_head(self):
        max_energy = -1
        new_ch_id = -1
        for i in range(self.SENSOR_NODES):
            if self.nodes[i]['residual_energy'] > max_energy:
                max_energy = self.nodes[i]['residual_energy']
                new_ch_id = i
        self.current_ch_id = new_ch_id if new_ch_id != -1 else 0

    def _get_observation(self):
        ch_node = self.nodes[self.current_ch_id]
        return np.array([
            min(1.0, ch_node['normal_q_len'] / 20.0),
            min(1.0, ch_node['hp_q_len'] / 20.0),
            max(0.0, ch_node['residual_energy'] / self.INITIAL_ENERGY),
            min(1.0, ch_node['packet_age'] / 10.0),
            1.0 if ch_node['hp_q_len'] > 0 else 0.0
        ], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.nodes = [{
            'residual_energy': self.INITIAL_ENERGY, 'sensed_value': 50.0,
            'prev_sensed_value': 50.0, 'ewma_mu': 50.0, 'ewma_sigma_sq': 10.0,
            'dynamic_threshold': 80.0, 'normal_q_len': 0, 'hp_q_len': 0, 'packet_age': 0.0
        } for _ in range(self.NUM_NODES)]
        self.node_positions = self.np_random.random((self.NUM_NODES, 2)) * 100
        self._elect_cluster_head()
        self._total_steps = 0
        return self._get_observation(), {}

    def step(self, action):
        ch_node = self.nodes[self.current_ch_id]
        
        if action == 0 and ch_node['hp_q_len'] > 0:
            ch_node['hp_q_len'] -= 1; self._consume_energy(self.current_ch_id, self.PACKET_BITS, 70)
        elif action == 1 and ch_node['normal_q_len'] > 0:
            ch_node['normal_q_len'] -= 1; ch_node['packet_age'] = 0; self._consume_energy(self.current_ch_id, self.PACKET_BITS, 70)
        elif action == 2 and ch_node['normal_q_len'] > 0:
            ch_node['normal_q_len'] -= 1; ch_node['packet_age'] = 0
        else: self._consume_energy(self.current_ch_id, 1, 0)

        sender_id = self.np_random.integers(0, self.SENSOR_NODES)
        if sender_id != self.current_ch_id and self.nodes[sender_id]['residual_energy'] > 0:
            sender = self.nodes[sender_id]
            sender['prev_sensed_value'] = sender['sensed_value']
            sender['sensed_value'] = self.np_random.uniform(20, 90)
            self._consume_energy(sender_id, 20, 0)
            dist = np.linalg.norm(self.node_positions[sender_id] - self.node_positions[self.current_ch_id])
            self._consume_energy(sender_id, self.PACKET_BITS, dist)
            self._consume_energy(self.current_ch_id, self.PACKET_BITS, 0)
            self._update_dynamic_threshold(sender_id)
            if abs(sender['sensed_value'] - sender['prev_sensed_value']) > 20: ch_node['hp_q_len'] += 1
            elif sender['sensed_value'] > sender['dynamic_threshold']: ch_node['normal_q_len'] += 1

        if ch_node['normal_q_len'] > 0: ch_node['packet_age'] += 0.2
        else: ch_node['packet_age'] = 0.0
            
        if self._total_steps > 0 and self._total_steps % 20 == 0: self._elect_cluster_head()
        self._total_steps += 1
        
        done = ch_node['residual_energy'] <= (self.INITIAL_ENERGY * 0.05)
        truncated = self._total_steps >= self._max_steps
        
        return self._get_observation(), 0.0, done, truncated, {}
