from dataclasses import dataclass

import numpy as np

from utilities.utils import load_config

CONFIG = load_config()
CHANNEL_CFG = CONFIG["channel"]

# Channel Model:
#   Path loss (dB):
#     PL(d) = path_loss_offset + path_loss_slope * log10(d_km)
#     d_km = max(d_m / 1000, min_distance_km)
#   Received power (dBm):
#     P_rx(d) = P_tx - PL(d)
#   Interference (dBm), linear ramp toward cell edge:
#     I(d) = interference_base_dbm + interference_edge_gain_dbm * (d / cell_radius)
#   Noise floor (dBm):
#     N = thermal_noise_density_dbm_hz + 10 * log10(noise_bandwidth_hz)
#   Convert dBm to linear (mW):
#     x_linear = 10^(x_dbm / 10)
#   SINR (linear):
#     SINR = P_rx / (I + N)


@dataclass
class Channel:
    cell_radius_km: float = CHANNEL_CFG["cell_radius_km"]
    p_tx_dbm: float = CHANNEL_CFG["p_tx_dbm"]
    path_loss_offset: float = CHANNEL_CFG["path_loss_offset"]
    path_loss_slope: float = CHANNEL_CFG["path_loss_slope"]
    interference_base_dbm: float = CHANNEL_CFG["interference_base_dbm"]
    interference_edge_gain_dbm: float = CHANNEL_CFG["interference_edge_gain_dbm"]
    noise_bandwidth_hz: float = CHANNEL_CFG["noise_bandwidth_hz"]
    thermal_noise_density_dbm_hz: float = CHANNEL_CFG["thermal_noise_density_dbm_hz"]
    min_distance_km: float = CHANNEL_CFG["min_distance_km"]

    @property
    def noise_floor_dbm(self) -> float:
        return self.thermal_noise_density_dbm_hz + 10 * np.log10(self.noise_bandwidth_hz)

    def compute_path_loss_db(self, distance_m: float) -> float:
        distance_km = max(distance_m / 1000.0, self.min_distance_km)
        return self.path_loss_offset + self.path_loss_slope * np.log10(distance_km)

    def compute_received_power_dbm(self, distance_m: float) -> float:
        return self.p_tx_dbm - self.compute_path_loss_db(distance_m=distance_m)

    def compute_interference_dbm(self, distance_m: float) -> float:
        edge_fraction = distance_m / (self.cell_radius_km * 1000) if self.cell_radius_km > 0 else 0.0
        return self.interference_base_dbm + self.interference_edge_gain_dbm * edge_fraction

    @staticmethod
    def dbm_to_linear(value_dbm: float) -> float:
        return 10 ** (value_dbm / 10)

    def compute_sinr_linear(self, distance_m: float) -> float:
        p_rx_linear = self.dbm_to_linear(self.compute_received_power_dbm(distance_m=distance_m))
        i_linear = self.dbm_to_linear(self.compute_interference_dbm(distance_m=distance_m))
        n_linear = self.dbm_to_linear(self.noise_floor_dbm)
        return p_rx_linear / (i_linear + n_linear)
