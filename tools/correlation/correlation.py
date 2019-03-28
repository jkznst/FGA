import numpy as np
import pickle
import matplotlib.pyplot as plt

boundaryregerrors_path = "./boundary_reg_error"
boundaryclsprob_path = "./boundary_cls_prob"

with open(boundaryregerrors_path, 'rb') as f:
    boundary_reg_errors_dict = pickle.load(f)
    f.close()

with open(boundaryclsprob_path, 'rb') as f:
    boundary_cls_probs_dict = pickle.load(f)
    f.close()

boundary_reg_errors = np.empty(shape=(0,))
boundary_cls_probs = np.empty(shape=(0,))
for cid, reg_errors in boundary_reg_errors_dict.items():
    boundary_reg_errors = np.append(boundary_reg_errors, reg_errors)

for cid, cls_probs in boundary_cls_probs_dict.items():
    boundary_cls_probs = np.append(boundary_cls_probs, cls_probs)

powered_cls_prob = np.power(boundary_cls_probs,4)
mean_cls_prob = np.mean(powered_cls_prob)

boundary_reg_errors = np.reshape(boundary_reg_errors, newshape=(-1, 2)) / 300. * 512.
boundary_reg_errors_x = boundary_reg_errors[:, 0]
boundary_reg_errors_y = boundary_reg_errors[:, 1]
boundary_reg_distance = np.sqrt(np.square(boundary_reg_errors_x) + np.square(boundary_reg_errors_y))

corrcoef = np.corrcoef(boundary_reg_distance, boundary_cls_probs)

plt.scatter(boundary_reg_distance, boundary_cls_probs, s=1, alpha=0.5)
plt.title("Correlation", fontsize=18)
plt.xlabel("regression error", fontsize=9)
plt.ylabel("cls prob", fontsize=9)
plt.savefig("./correlation.jpg", dpi=1000)


