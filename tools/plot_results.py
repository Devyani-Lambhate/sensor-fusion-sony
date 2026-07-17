import matplotlib.pyplot as plt
import numpy as np

# Basic line plot
'''
x = ['lift', 'moderate', 'heavy']
map_bev = [0.4941,0.4939,0.2875]
nds_bev= [0.5373,0.5373,0.4215]
map_di = [0.5082,0.5071,0.3102]
nds_di= [0.5330,0.5313,0.4116]

x = ['0.5', '1.0', '1.5', '2.0', '2.5']
map_bev = [0.4968,0.5009,0.5011,0.5091,0.506]
nds_bev= [0.5353,0.5374,0.5411,0.5425,0.5372]   
map_di = [0.5374,0.5463,0.5515,0.5503,0.5432]
nds_di= [0.5414,0.5492,0.5541,0.5508,0.5461]


x=[5,20,50,80,100]
map_bev = [0.4687,0.4664,0.4752,0.4637,0.4714]
nds_bev= [0.5144,0.5126,0.5181,0.5121,0.5168]
map_di = [0.4932,0.4922,0.4940,0.4878,0.4966]
nds_di = [0.5139,0.5166,0.5143,0.5133,0.5172]

'''
x=[0.01,0.04]
map_bev_lidar = [0.571,0.4875]
nds_bev_lidar = [0.58,0.5255]
map_bev_cam =[0.547,0.5449]
nds_bev_cam =[0.5718,0.5704]
map_bev = [0.5378,0.4112]
nds_bev = [0.5635,0.48]
map_bev_clear = 0.5756
nds_bev_clear = 0.5805
map_di = [0.465446,0.3428]
nds_di = [0.3389,0.2751]
map_di_lidar = [0.58613,0.5079]
nds_di_lidar = [0.5761,0.5246]
map_di_cam = [0.575138,0.463686]
nds_di_cam = [0.565007,0.3397]
map_di_clear = 0.5836
nds_di_clear = 0.5804
"""


plt.figure(figsize=(10, 6))
plt.plot(x, map_bev_lidar, 'b--', linewidth=2, label='mAP BEV LIDAR-W CAM-C')  # blue solid line
plt.plot(x, map_bev_cam, 'g--', linewidth=2, label='mAP BEV LIDAR-C CAM-W')  # blue solid line
plt.plot(x, map_bev, 'k--', linewidth=2, label='mAP BEV LIDAR-W CAM-W')  # black solid line
plt.plot(x, map_di_lidar, 'b-', linewidth=2, label='mAP DI LIDAR-W CAM-C')  # blue solid line
plt.plot(x, map_di_cam, 'g-', linewidth=2, label='mAP DI LIDAR-C CAM-W')  # blue dashed line
plt.plot(x, map_di, 'k-', linewidth=2, label='mAP DI LIDAR-W CAM-W')  # blue dashed line

# plt.plot(x, nds_bev_lidar, 'b--', linewidth=2, label='NDS BEV LIDAR-W CAM-C')  # red dashed line
# plt.plot(x, nds_bev_cam, 'g--', linewidth=2, label='NDS BEV LIDAR-C CAM-W')  # red dashed line
# plt.plot(x, nds_bev, 'k--', linewidth=2, label='NDS BEV LIDAR-W CAM-W')  # red dashed line
# plt.plot(x, nds_di_lidar, 'b-', linewidth=2, label='NDS DI LIDAR-W CAM-C')  # blue solid line
# plt.plot(x, nds_di_cam, 'g-', linewidth=2, label='NDS DI LIDAR-C CAM-W')  # blue solid line
# plt.plot(x, nds_di, 'k-', linewidth=2, label='NDS DI LIDAR-W CAM-W')  # blue solid line


plt.xlabel('fog rate')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
"""

import matplotlib.pyplot as plt
import numpy as np

plt.style.use("seaborn-v0_8-whitegrid")   # base style

# Global style tweaks
plt.rcParams.update({
    "figure.figsize": (6, 4.5),
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.edgecolor":"#666666",
    "axes.linewidth": 1.0,
    "grid.color":"#dddddd",
    "grid.linestyle": "-",
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "legend.frameon": False,
})

# Example data
x = np.arange(2)
#labels = ["None", "rain_rate_5.0", "rain_rate_20.0",
 #         "rain_rate_50.0", "rain_rate_80.0", "rain_rate_100.0"]
#di = [0.59, 0.49, 0.49, 0.49, 0.49, 0.50]
#bev = [0.58, 0.47, 0.47, 0.47, 0.46, 0.47]
labels = [ "Light fog (0.01)", "Dense fog (0.04)"]

width = 0.05  # the width of the bars
fig, ax = plt.subplots()

ax.bar(x - width, map_bev_lidar, width, label="LiDAR (fog) + Camera (w/o fog)",
       color="#004b8d")          # dark blue
ax.bar(x , map_bev_cam, width, label="LiDAR (w/o fog) + Camera (fog)",
       color="#b7e3b5")          # light green
ax.bar(x + width, map_bev, width, label="LiDAR (fog) + Camera (fog)",
       color="#e3b5e1")          # light pink
       
ax.axhline(y=map_bev_clear, color="black", linestyle="--", linewidth=1.5, label ="Clear weather")  # black dashed line

ax.set_title("mAP Comparison BEV")
ax.set_ylabel("mAP")
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=20, ha="right")
ax.set_ylim(0, 0.75)             # similar headroom at top
ax.legend()

plt.tight_layout()
plt.show()
