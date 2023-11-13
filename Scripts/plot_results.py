import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
import numpy as np
#
# # Data
# # 2 Clients
# client_21 = [0.335840993, 0.091600802, 0.073736538, 0.058619788, 0.052826779, 0.045305138, 0.047421929, 0.040420067,
#              0.038007774, 0.039562441, 0.036246055, 0.035379326, 0.029831103, 0.033985643, 0.028491421, 0.029675697,
#              0.027835061, 0.025468326, 0.025688127, 0.02468077]
# client_22 = [0.338357031, 0.095308618, 0.073113762, 0.062625722, 0.057585837, 0.050634498, 0.052078395, 0.041811375,
#              0.037430761, 0.040315729, 0.033851158, 0.033925106, 0.03413691, 0.030566476, 0.030425856, 0.027195681,
#              0.026864805, 0.029039255, 0.024984724, 0.025519042]
# # 3 Clients
# client_31 = [
#     0.456084265, 0.134767403, 0.092890968, 0.079048062, 0.064170307, 0.056570306, 0.054606029, 0.055177186,
#     0.048998386, 0.046729074, 0.043820406, 0.043083545, 0.041797426, 0.037477453, 0.035643963, 0.034331007,
#     0.034692289, 0.033913962, 0.037017216, 0.034102623]
#
# client_32 = [
#     0.445404235, 0.134705754, 0.088757139, 0.081081897, 0.067035766, 0.063873194, 0.055378536, 0.053868797,
#     0.048344055, 0.049018322, 0.042748239, 0.04088473, 0.03839329, 0.037452389, 0.035717807, 0.036284209,
#     0.034568492, 0.033598726, 0.036154872, 0.03129754]
#
# client_33 = [
#     0.448148906, 0.114893438, 0.095079534, 0.075057508, 0.070165262, 0.064399198, 0.055310841, 0.057693175,
#     0.050397038, 0.051462761, 0.047800967, 0.042350835, 0.044133978, 0.039745248, 0.037483727, 0.039466172,
#     0.032425039, 0.032855614, 0.03869602, 0.030996005]
#
# # 5 Clients
# client_51 = [
#     0.641951958, 0.189052517, 0.143514823, 0.121745327, 0.090662009,
#     0.077309909, 0.091924898, 0.068433743, 0.074729035, 0.061234624,
#     0.058124805, 0.057661783, 0.050064855, 0.05404978, 0.055975467,
#     0.047747955, 0.042070417, 0.045632694, 0.044394357, 0.04653707]
#
# client_52 = [
#     0.643105801, 0.191686121, 0.132227904, 0.126353762, 0.104853043,
#     0.088856079, 0.069618278, 0.073008255, 0.065987565, 0.065916404,
#     0.054646632, 0.056588827, 0.052226854, 0.053171541, 0.049457286,
#     0.04908809, 0.052323877, 0.042301952, 0.044717027, 0.041293908]
#
# client_53 = [
#     0.642992863, 0.162002749, 0.138446558, 0.121152728, 0.100178617,
#     0.088057297, 0.074738161, 0.070618233, 0.068095298, 0.064368683,
#     0.071156971, 0.057299944, 0.054364234, 0.049994806, 0.053676367,
#     0.044244711, 0.046877995, 0.044053737, 0.041529879, 0.038174923]
#
# client_54 = [
#     0.642905174, 0.204008404, 0.140009001, 0.121369876, 0.108292745,
#     0.089341082, 0.087638369, 0.069000425, 0.069315331, 0.06600143,
#     0.05675286, 0.053765305, 0.056337328, 0.050727176, 0.049653066,
#     0.050294199, 0.045162569, 0.048098834, 0.046947059, 0.043857566]
#
# client_55 = [
#     0.639903327, 0.204412973, 0.13510359, 0.102147602, 0.102838838,
#     0.083611698, 0.075316596, 0.070685474, 0.069570511, 0.062902947,
#     0.070229128, 0.058198007, 0.057868284, 0.059465354, 0.056526037,
#     0.050384829, 0.052071209, 0.052863571, 0.047117363, 0.043107813]
#
# # 7 Clients
#
# client_71 = [
#     0.778036491, 0.204926075, 0.174196663, 0.145414104, 0.152778262,
#     0.105633863, 0.085933225, 0.087291783, 0.076081877, 0.066625895,
#     0.067826656, 0.067524705, 0.071502221, 0.055159779, 0.055500206,
#     0.051707203, 0.049749027, 0.055620831, 0.048030093, 0.046472475]
#
# client_72 = [
#     0.781251738, 0.218003663, 0.174161076, 0.145975924, 0.151388434,
#     0.107072665, 0.09280885, 0.084054244, 0.076487572, 0.075155603,
#     0.061471982, 0.066587143, 0.062112983, 0.067373262, 0.059515544,
#     0.05772227, 0.050204106, 0.054259213, 0.047267982, 0.048031225]
#
# client_73 = [
#     0.78014435, 0.218577387, 0.181869204, 0.148514075, 0.13698409,
#     0.118987246, 0.093048443, 0.078272816, 0.078188839, 0.075118069,
#     0.074103115, 0.068743685, 0.053753274, 0.058131971, 0.057363851,
#     0.053379266, 0.055928693, 0.053950581, 0.049103449, 0.045488208]
#
# client_74 = [
#     0.78097628, 0.209479529, 0.197921849, 0.143433251, 0.135568708,
#     0.10187068, 0.096404775, 0.096368926, 0.082481535, 0.067249842,
#     0.068589008, 0.060699053, 0.064581903, 0.061791159, 0.0634193,
#     0.053080936, 0.051129766, 0.046305698, 0.050567558, 0.04819643]
#
# client_75 = [
#     0.78382293, 0.239377968, 0.185071827, 0.139937551, 0.144342208,
#     0.124929674, 0.099264514, 0.104531954, 0.070712419, 0.081877305,
#     0.067713033, 0.068305785, 0.062279616, 0.058604444, 0.053485849,
#     0.055972901, 0.056635657, 0.052401836, 0.059625918, 0.044185706]
#
# client_76 = [
#     0.772927715, 0.249519815, 0.18163002, 0.144337025, 0.150277739,
#     0.106608024, 0.112422208, 0.091306107, 0.079762429, 0.078158734,
#     0.066810481, 0.063015982, 0.065789247, 0.06100946, 0.059406538,
#     0.058117687, 0.059355258, 0.056547274, 0.052068776, 0.047215122]
#
# client_77 = [
#     0.777133626, 0.249966317, 0.200639628, 0.13394007, 0.12762857,
#     0.102099236, 0.104619138, 0.083076122, 0.075500926, 0.075841003,
#     0.067986892, 0.06058831, 0.057041761, 0.065303828, 0.066859011,
#     0.055939253, 0.053915706, 0.05707438, 0.049210355, 0.051218472]
#
#
# # X-axis values (assuming 1 to 20 for each data point)
# epochs = list(range(1, 21))
#
# # Create a figure with four subplots
# fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
# fig.suptitle('Training Loss Comparison', fontsize=16)
#
# # Subplot 1
# axes[0, 0].plot(epochs, client_21, label='Client 1', color='red')
# axes[0, 0].plot(epochs, client_22, label='Client 2', color='red')
# # axes[0, 0].set_title('Training Loss')
# axes[0, 0].set_xlabel('Epoch')
# axes[0, 0].set_ylabel('Training Loss')
# axes[0, 0].legend()
# axes[0, 0].grid(True)
#
# # Subplot 2
# axes[0, 1].plot(epochs, client_31, label='Client 1', color='red')
# axes[0, 1].plot(epochs, client_32, label='Client 2', color='blue')
# axes[0, 1].plot(epochs, client_33, label='Client 3', color='green')
# axes[0, 1].set_xlabel('Epoch')
# axes[0, 1].set_ylabel('Training Loss')
# axes[0, 1].legend()
# axes[0, 1].grid(True)
#
# # Subplot 3
# axes[1, 0].plot(epochs, client_51, label='Client 1', color='red')
# axes[1, 0].plot(epochs, client_52, label='Client 2', color='blue')
# axes[1, 0].plot(epochs, client_53, label='Client 3', color='green')
# axes[1, 0].plot(epochs, client_54, label='Client 4', color='yellow')
# axes[1, 0].plot(epochs, client_55, label='Client 5', color='black')
# axes[1, 0].set_xlabel('Epoch')
# axes[1, 0].set_ylabel('Training Loss')
# axes[1, 0].legend()
# axes[1, 0].grid(True)
#
# # Subplot 4
# axes[1, 1].plot(epochs, client_71, label='Client 1', color='red')
# axes[1, 1].plot(epochs, client_72, label='Client 2', color='blue')
# axes[1, 1].plot(epochs, client_73, label='Client 3', color='green')
# axes[1, 1].plot(epochs, client_74, label='Client 4', color='yellow')
# axes[1, 1].plot(epochs, client_75, label='Client 5', color='black')
# axes[1, 1].plot(epochs, client_76, label='Client 6', color='orange')
# axes[1, 1].plot(epochs, client_77, label='Client 7', color='brown')
# axes[1, 1].set_xlabel('Epoch')
# axes[1, 1].set_ylabel('Training Loss')
# axes[1, 1].legend()
# axes[1, 1].grid(True)
#
# # Adjust layout
# plt.tight_layout()
# plt.subplots_adjust(top=0.85)
#
# # Save the plot as a high-resolution image (e.g., PNG or PDF) for publication
# plt.savefig('subplots_training_loss.png', dpi=300, bbox_inches='tight')
#
# # Show the plot (uncomment this line if you want to display the plot interactively)
# plt.show()
#
#
# """Confusion matrices """
# import numpy as np
# import matplotlib.pyplot as plt
#
# # Define the data
# data = {
#     "Normal": {"TP": 9351, "TN": 10210, "FP": 2623, "FN": 360},
#     "DoS": {"TP": 6211, "TN": 14864, "FP": 220, "FN": 1249},
#     "Probe": {"TP": 1801, "TN": 19629, "FP": 494, "FN": 620},
#     "R2L": {"TP": 724, "TN": 18778, "FP": 881, "FN": 2161},
#     "U2R": {"TP": 35, "TN": 22273, "FP": 204, "FN": 32},
# }
#
# # Create subplots for confusion matrices
# fig, axes = plt.subplots(1, 5, figsize=(15, 3))
# fig.suptitle("Confusion Matrices for Different Classes")
#
# for i, (class_name, class_data) in enumerate(data.items()):
#     TP = class_data["TP"]
#     TN = class_data["TN"]
#     FP = class_data["FP"]
#     FN = class_data["FN"]
#
#     confusion_matrix = np.array([[TN, FP], [FN, TP]])  # Closing parenthesis here
#
#     ax = axes[i]
#     ax.matshow(confusion_matrix, cmap='Blues')
#
#     for x in range(2):
#         for y in range(2):
#             ax.text(x, y, str(confusion_matrix[y, x]), va='center', ha='center', color='r', fontsize=14)
#
#     ax.set_xticks([0, 1])
#     ax.set_yticks([0, 1])
#
#     ax.set_xticklabels(['Actual Negative', 'Actual Positive'])
#     ax.set_yticklabels(['Predicted Negative', 'Predicted Positive'])
#
#     ax.set_title(class_name)
#
# # Adjust layout
# plt.tight_layout()
# plt.subplots_adjust(top=0.8)
#
# # Show the subplots
# plt.show()


# """ 1 client split learning and centralised learning train loss plot"""
#
# # Your data
# epochs = list(range(1, 20))
# SL = [0.210870934, 0.056026394, 0.04425592, 0.036903052, 0.031748183, 0.027955038, 0.02533597, 0.023069682,
#       0.02122187, 0.019766141, 0.01861548, 0.018146876, 0.017250362, 0.016879137, 0.016019771, 0.015288935,
#       0.015074104, 0.014651917, 0.014202914]
# SL_NLS = [0.212406261, 0.055642255, 0.043432365, 0.036297544, 0.030724855, 0.027581304, 0.024815284, 0.022945183,
#           0.021150291, 0.019906222, 0.018781682, 0.018278632, 0.017191459, 0.017332814, 0.015999935, 0.015745781,
#           0.015618321, 0.015345966, 0.014408134]
# FFNN = [0.214142733, 0.053500461, 0.041613175, 0.034024569, 0.02854262, 0.02641896, 0.023217021, 0.021238591,
#         0.020289338, 0.018907945, 0.017746552, 0.016722494, 0.016292562, 0.015699314, 0.015627573, 0.014900325,
#         0.014523337, 0.014275299, 0.013804741]
#
# # Create a figure and axis
# fig, ax = plt.subplots()
#
# # Plot the data
# ax.plot(epochs, SL, '-o', label='SL')
# ax.plot(epochs, SL_NLS, '-s', label='SL-NLS')
# ax.plot(epochs, FFNN, '-^', label='FFNN')
#
# # Add labels and title
# ax.set_xlabel('Epochs')
# ax.set_xticks(range(0, 21, 2))
# ax.set_ylabel('Train Loss')
#
# # Add a grid
# ax.grid(True)
#
# # Add a legend
# ax.legend(loc='lower right')
#
# # Create an inset axis for zoomed-in view
# axins = zoomed_inset_axes(ax, 2, loc='upper right')
# axins.grid(True)
# axins.plot(epochs[1:7], SL[1:7], '-o', label='SL')
# axins.plot(epochs[1:7], SL_NLS[1:7], '-s', label='SL-NLS')
# axins.plot(epochs[1:7], FFNN[1:7], '-^', label='FFNN')
# axins.set_xlim(2, 7)
# axins.set_ylim(0.02, 0.075)
# mark_inset(ax, axins, loc1=2, loc2=3, fc="none", ec="0.5")
#
# # Save the chart as an image
# plt.savefig('loss_chart.png')
#
# # Show the chart (optional)
# plt.show()

"""Validation Accuracies SM-IDS"""
# epochs = list(range(1, 21))
# SL = [
#     97.65696966, 98.26167058, 98.56624967, 98.87528601, 99.01494666,
#     99.19620836, 99.24375241, 99.29872523, 99.34626928, 99.37301281,
#     99.35964104, 99.3700413, 99.42204261, 99.46661516, 99.53793124,
#     99.51564496, 99.55576026, 99.48147268, 99.42352836, 99.38787
# ]
#
# SL_NLS = [
#     97.73423813, 98.55264588, 98.79922395, 98.96081167, 99.13251518,
#     99.2398133, 99.27850954, 99.23853226, 99.34211979, 99.40484657,
#     99.45387947, 99.45109653, 99.44128995, 99.51192383, 99.53864897,
#     99.49566788, 99.43625413, 99.44314524, 99.45785512, 99.4265
# ]
#
# FFNN = [
#     97.72828574, 98.12052417, 98.39538823, 98.58407869, 98.80248418,
#     99.05654771, 99.1427213, 99.22740915, 99.38192732, 99.39975634,
#     99.41907111, 99.43541438, 99.45770065, 99.48147268, 99.47255817,
#     99.49781595, 99.48890144, 99.49484444, 99.51564496, 99.57061778
# ]
#
# # Create a figure and axis
# fig, ax = plt.subplots()
#
# # Plot the data
# ax.plot(epochs, SL, '-o', label='SL')
# ax.plot(epochs, SL_NLS, '-s', label='SL-NLS')
# ax.plot(epochs, FFNN, '-^', label='Centralised')
#
# # Add labels and title
# ax.set_xlabel('Epochs', fontsize=14)
# ax.set_xticks(range(0, 21, 2))
# ax.set_ylabel('Validation Accuracy (%)', fontsize=14)  # You can change this to 'Loss' if needed#
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)
#
# # Add a legend
# ax.legend(loc='lower right', fontsize=14)
# # Add a grid
# ax.grid(True)
#
#
# # Save the chart as an image
# plt.savefig("validation_accuracies_sm_ids.png", dpi=300, bbox_inches='tight')
# # Show the chart (optional)
# plt.show()


""" Figure of evaluation metrics (bar chart SM-IDS)"""
# # Data from the table
# categories = ["Normal", "DoS", "Probe", "R2L", "U2R"]
# SL
# precision = [0.78, 0.97, 0.78, 0.45, 0.15]
# recall = [0.96, 0.83, 0.74, 0.25, 0.52]
# f1_score = [0.86, 0.89, 0.76, 0.32, 0.23]

# SL-NLS
# precision = [0.73, 0.95, 0.84, 0.61, 0.2]
# recall = [0.96, 0.84, 0.74, 0.2, 0.43]
# f1_score = [0.83, 0.89, 0.78, 0.3, 0.27]
#
# Centralised
# precision = [0.76, 0.96, 0.78, 0.61, 0.22]
# recall = [0.96, 0.85, 0.76, 0.26, 0.48]
# f1_score = [0.85, 0.9, 0.77, 0.37, 0.3]
#
# # Bar width
# bar_width = 0.2
#
# # Bar positions
# x = range(len(categories))
#
# # Create subplots
# fig, ax = plt.subplots()
# plt.xticks(fontsize=18)
# plt.yticks(fontsize=18)
# # Plot each metric as grouped bars
# plt.bar([i - 1.5 * bar_width for i in x], precision, bar_width, label="Precision", alpha=0.8, color='royalblue')
# plt.bar([i - 0.5 * bar_width for i in x], recall, bar_width, label="Recall", alpha=0.8, color='forestgreen')
# plt.bar([i + 0.5 * bar_width for i in x], f1_score, bar_width, label="F1-Score", alpha=0.8, color='gold')
#
#
# # Set labels, title, and legend
# ax.set_xlabel('Categories', fontsize=18)
# ax.set_xticks(x)
# ax.set_xticklabels(categories)
#
# plt.legend(loc='upper right', fontsize=18)
# plt.xticks(rotation=0)  # Rotate category labels for better readability
# plt.grid(axis='y', linestyle='--', alpha=0.7)
# plt.tight_layout()  # Ensure all labels and the figure fit properly
#
# # Save the figure as a high-quality image
# plt.savefig("performance_metrics_chart_centralised.png", dpi=300, bbox_inches='tight')
# plt.show()


""" Federated Learning Performance Validation NAN-IDS"""
# epochs = list(range(1, 21))
# FL_2 = [96.637744, 97.8159451, 98.2022405, 98.3508157, 98.5305916, 98.6048792, 98.7415684, 98.7935697,
#         98.9391733, 99.066948, 99.0253469, 99.1679791, 99.2600957, 99.2615814, 99.2779247, 99.2912965,
#         99.2600957, 99.3685556, 99.365584, 99.4443289]
#
# FL_3 = [94.7032954, 97.2394734, 97.6302261, 97.9036044, 98.3374439, 98.4622471, 98.5276201, 98.5008766,
#         98.5959647, 98.671738, 98.7460256, 98.8099129, 98.7772264, 98.9792886, 99.0208897, 98.905001,
#         99.1516358, 99.2036371, 99.1798651, 99.2660387]
#
# FL_5 = [93.1373132, 94.9335869, 95.7997801, 97.3360473, 97.6361691, 97.9036044, 97.95412, 98.1413247,
#         98.3433869, 98.4102457, 98.4637328, 98.4696758, 98.5305916, 98.5558494, 98.6345942, 98.6613378,
#         98.7281966, 98.7400826, 98.8203132, 98.8738003]
# FL_7 = [91.2043503, 94.0510504, 95.0242178, 95.4536, 97.1414138, 97.5336523, 97.6643984, 97.774344,
#         97.8768609, 97.9882923, 98.1086382, 98.2364128, 98.3656732, 98.4251033, 98.4592755, 98.4800761,
#         98.5068196, 98.4696758, 98.5751642, 98.6435088]
#
# # Create a figure and axis
# fig, ax = plt.subplots()
#
# # Plot the data
# ax.plot(epochs, FL_2, '-o', label='2 Clients')
# ax.plot(epochs, FL_3, '-o', label='3 Clients')
# ax.plot(epochs, FL_5, '-o', label='5 Clients')
# ax.plot(epochs, FL_7, '-o', label='7 Clients')
# # Add labels and title
# plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
# ax.set_xlabel('Epochs', fontsize=18)
# ax.set_xticks(range(0, 21, 2))
# ax.set_ylabel('Validation Accuracy (%)', fontsize=16)  # You can change this to 'Loss' if needed#
# plt.xticks(fontsize=18)
# plt.yticks(fontsize=18)
#
# # Add a legend
# ax.legend(loc='lower right', fontsize=18)
# # Add a grid
# ax.grid(True)
#
#
# # Save the chart as an image
# plt.savefig("validation_accuracies_nan_ids_fl.png", dpi=300, bbox_inches='tight')
# # Show the chart (optional)
# plt.show()

""" Split Learning Performance Validation NAN-IDS"""
# epochs = list(range(1, 21))
# SL_2 = [97.57822482, 98.19629751, 98.34338692, 98.59447895, 98.79951267, 98.92431581, 98.9584881, 99.03574718,
#         99.22592339, 99.11300627, 99.12340653, 99.03574718, 99.03277568, 99.01940392, 98.99711764, 99.14123555,
#         99.11449202, 99.32992601, 99.28981072, 99.40421]
#
#
# SL_3 = [96.54414168, 98.00760705, 98.21264077, 98.39687398, 98.45036104, 98.49939084, 98.75196862, 98.84111372,
#         98.94511633, 99.00603215, 99.07140522, 99.14717856, 99.15312156, 99.2288949, 99.25118117, 99.27941045,
#         99.29723947, 99.16203607, 99.30169673, 99.3745]
#
# SL_5 = [97.22610168, 97.70154221, 97.94817698, 98.14578195, 98.28989986, 98.37607346, 98.42213176, 98.58259293,
#         98.6910528, 98.71928208, 98.77276914, 98.86785725, 98.88568627, 98.90500104, 98.99711764, 99.01048941,
#         99.08923424, 99.09517725, 99.1397498, 99.16055]
#
# SL_7 = [95.05690429, 97.35684783, 97.64211214, 97.99126378, 98.15915372, 98.15766796, 98.24978457, 98.31812914,
#         98.43104627, 98.53653463, 98.61825097, 98.42361751, 98.62865123, 98.79208392, 98.83071346, 98.88271477,
#         98.84408522, 98.90054379, 98.95105934, 99.03575]
#
# # Create a figure and axis
# fig, ax = plt.subplots()
#
# # Plot the data
# ax.plot(epochs, SL_2, '-o', label='2 Clients')
# ax.plot(epochs, SL_3, '-o', label='3 Clients')
# ax.plot(epochs, SL_5, '-o', label='5 Clients')
# ax.plot(epochs, SL_7, '-o', label='7 Clients')
# # Add labels and title
# plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
# ax.set_xlabel('Epochs', fontsize=18)
# ax.set_xticks(range(0, 21, 2))
# ax.set_ylabel('Validation Accuracy (%)', fontsize=14)  # You can change this to 'Loss' if needed#
# plt.xticks(fontsize=18)
# plt.yticks(fontsize=18)
#
# # Add a legend
# ax.legend(loc='lower right', fontsize=18)
# # Add a grid
# ax.grid(True)
#
#
# # Save the chart as an image
# plt.savefig("validation_accuracies_nan_ids_sl.png", dpi=300, bbox_inches='tight')
# # Show the chart (optional)
# plt.show()

""" Split Learning-NLS Performance Validation NAN-IDS"""
# epochs = list(range(1, 20))
# SLNLS_2 = [97.51005616, 97.66824515, 97.81081031, 97.92940693, 98.03063963, 98.11504752, 98.19014605, 98.2582279,
#            98.31849333, 98.37249482, 98.42126376, 98.46612319, 98.50796328, 98.54714708, 98.58333335, 98.61673353,
#            98.64733743, 98.67544431, 98.70178013]
#
# SLNLS_3 = [96.30269822, 96.76596987, 97.1020302, 97.33731803, 97.5230679, 97.66907473, 97.78977354, 97.89213589,
#            97.97993468, 98.05662015, 98.12485643, 98.18574362, 98.24061101, 98.2909895, 98.33653395, 98.37820447,
#            98.41725697, 98.45370779, 98.48765763]
#
# SLNLS_5 = [96.58867932, 96.917869,	97.18309712, 97.35955094, 97.48758945, 97.58749589, 97.67135065, 97.74239883,
#            97.80251693, 97.85635968, 97.90578442, 97.95101094, 97.99322174, 98.03304839, 98.07074207, 98.10679464,
#            98.14172156, 98.17497286, 98.20612838]
#
#
# SLNLS_7 = [96.85977385, 97.0348174,	97.1920195, 97.3335738, 97.45605784, 97.54537817, 97.61670704, 97.68084651,
#            97.73846069, 97.79127063, 97.84006961, 97.88444476, 97.92470486, 97.96153807, 97.99550736, 98.02682157,
#            98.05622229, 98.08426961, 98.11049099]
#
# # Create a figure and axis
# fig, ax = plt.subplots()
#
# # Plot the data
# ax.plot(epochs, SLNLS_2, '-o', label='2 Clients')
# ax.plot(epochs, SLNLS_3, '-o', label='3 Clients')
# ax.plot(epochs, SLNLS_5, '-o', label='5 Clients')
# ax.plot(epochs, SLNLS_7, '-o', label='7 Clients')
# # Add labels and title
# plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
# ax.set_xlabel('Epochs', fontsize=18)
# ax.set_xticks(range(0, 21, 2))
# ax.set_ylabel('Validation Accuracy (%)', fontsize=14)  # You can change this to 'Loss' if needed#
# plt.xticks(fontsize=18)
# plt.yticks(fontsize=18)
#
# # Add a legend
# ax.legend(loc='lower right', fontsize=18)
# # Add a grid
# ax.grid(True)
#
#
# # Save the chart as an image
# plt.savefig("validation_accuracies_nan_ids_sl_nls.png", dpi=300, bbox_inches='tight')
# # Show the chart (optional)
# plt.show()


""" Federated Learning NAN-IDS radar chart test"""
# labels = ['Precision', 'Recall', 'F1-score', 'Accuracy', 'Precision']
# markers = [0.75, 0.76, 0.77, 0.78, 0.79, 0.80, 0.81, 0.82]
# # Number of variables we're plotting.
# num_vars = len(labels)
#
# FL_2 = [0.789212651, 0.797580731, 0.781264638, 0.7990596, 0.789212651]
# FL_3 = [0.781170156, 0.794784865, 0.768690561, 0.794269, 0.781170156]
# FL_5 = [0.776886089, 0.781314762, 0.765878282, 0.7843329, 0.776886089]
# FL_7 = [0.781924681, 0.789070263, 0.767612225, 0.7892122, 0.781924681]
# label_loc = np.linspace(start=0, stop=2 * np.pi, num=len(labels))
#
# plt.figure(figsize=(8, 8), facecolor="white")
# ax = plt.subplot(polar=True)
# ax.plot(label_loc, FL_2, 'o--', color='green', label='2 Clients')
# ax.plot(label_loc, FL_3, 'o--', color='blue', label='3 Clients')
# ax.plot(label_loc, FL_5, 'o--', color='red', label='5 Clients')
# ax.plot(label_loc, FL_7, 'o--', color='gold', label='7 Clients')
#
# # #fill plot
#
#
# # Fix axis to go in the right order and start at 12 o'clock.
# ax.set_theta_offset(np.pi / 2)
# ax.set_theta_direction(-1)
#
# # Draw axis lines for each angle and label.
# ax.set_thetagrids(np.degrees(label_loc), labels)
#
# # Go through labels and adjust alignment based on where
# # it is in the circle.
# for label, angle in zip(ax.get_xticklabels(), label_loc):
#     if 0 < angle < np.pi:
#         label.set_horizontalalignment('left')
#     else:
#         label.set_horizontalalignment('right')
#
# # Ensure radar goes from 0 to 100. it also removes the extra line
# # Set the axis limits and style
# ax.set_ylim(0.75, 0.80)  # Adjust the limits as needed
#
# ax.tick_params(axis='x', labelsize=22)
# ax.tick_params(axis='y', labelsize=22)
# # Change the color of the circular gridlines.
# ax.grid(color='#AAAAAA')
# # Change the color of the outermost gridline (the spine).
# ax.spines['polar'].set_color('#eaeaea')
# # Change the background color inside the circle itself.
# ax.set_facecolor('#FAFAFA')
#
# # Add a legend as well.
# ax.legend(loc='lower left', bbox_to_anchor=(-0.2, -0.1), fontsize=22)  # bbox_to_anchor=(1.3, 1.1)
# plt.savefig("radar_chart_nan_ids_fl.png", dpi=300, bbox_inches='tight')
# plt.show()

""" SL NAN-IDS Radar Chart"""
# labels = ['Precision', 'Recall', 'F1-score', 'Accuracy', 'Precision']
# markers = [0.75, 0.76, 0.77, 0.78, 0.79, 0.80, 0.81, 0.82]
# # Number of variables we're plotting.
# num_vars = len(labels)
#
# SL_2 = [0.804918382, 0.806460256, 0.788453247, 0.8060681, 0.804918382]
# SL_3 = [0.818918116, 0.809629613, 0.791394163, 0.8096611, 0.818918116]
# SL_5 = [0.806392832, 0.812278211, 0.798905696, 0.81183546, 0.806392832]
# SL_7 = [0.79695928, 0.808417317, 0.784642033, 0.805181, 0.79695928]
# label_loc = np.linspace(start=0, stop=2 * np.pi, num=len(labels))
#
# plt.figure(figsize=(8, 8), facecolor="white")
# ax = plt.subplot(polar=True)
# ax.plot(label_loc, SL_2, 'o--', color='green', label='2 Clients')
# ax.plot(label_loc, SL_3, 'o--', color='blue', label='3 Clients')
# ax.plot(label_loc, SL_5, 'o--', color='red', label='5 Clients')
# ax.plot(label_loc, SL_7, 'o--', color='gold', label='7 Clients')
#
# # #fill plot
#
#
# # Fix axis to go in the right order and start at 12 o'clock.
# ax.set_theta_offset(np.pi / 2)
# ax.set_theta_direction(-1)
#
# # Draw axis lines for each angle and label.
# ax.set_thetagrids(np.degrees(label_loc), labels)
#
# # Go through labels and adjust alignment based on where
# # it is in the circle.
# for label, angle in zip(ax.get_xticklabels(), label_loc):
#     if 0 < angle < np.pi:
#         label.set_horizontalalignment('left')
#     else:
#         label.set_horizontalalignment('right')
#
# # Ensure radar goes from 0 to 100. it also removes the extra line
# # Set the axis limits and style
# ax.set_ylim(0.76, 0.82)  # Adjust the limits as needed
#
# ax.tick_params(axis='x', labelsize=22)
# ax.tick_params(axis='y', labelsize=22)
# # Change the color of the circular gridlines.
# ax.grid(color='#AAAAAA')
# # Change the color of the outermost gridline (the spine).
# ax.spines['polar'].set_color('#eaeaea')
# # Change the background color inside the circle itself.
# ax.set_facecolor('#FAFAFA')
#
# # Add a legend as well.
# ax.legend(loc='lower left', bbox_to_anchor=(-0.2, -0.1), fontsize=22)
# plt.savefig("radar_chart_nan_ids_sl.png", dpi=300, bbox_inches='tight')
# plt.show()


"""SL-NLS NAN-IDS Radar Chart"""
labels = ['Precision', 'Recall', 'F1-score', 'Accuracy', 'Precision']
markers = [0.75, 0.76, 0.77, 0.78, 0.79, 0.80, 0.81, 0.82]
# Number of variables we're plotting.
num_vars = len(labels)

SLNLS_2 = [0.805603265, 0.80764017, 0.792274219, 0.8077537, 0.805603265]
SLNLS_3 = [0.782362491, 0.796731725,	0.776787615, 0.7963538, 0.782362491]
SLNLS_5 = [0.777510646, 0.796465578, 0.773008339, 0.7958659, 0.777510646]
SLNLS_7 = [0.797806955, 0.79865951, 0.784888662, 0.8005678, 0.797806955]
label_loc = np.linspace(start=0, stop=2 * np.pi, num=len(labels))

plt.figure(figsize=(8, 8), facecolor="white")
ax = plt.subplot(polar=True)
ax.plot(label_loc, SLNLS_2, 'o--', color='green', label='2 Clients')
ax.plot(label_loc, SLNLS_3, 'o--', color='blue', label='3 Clients')
ax.plot(label_loc, SLNLS_5, 'o--', color='red', label='5 Clients')
ax.plot(label_loc, SLNLS_7, 'o--', color='gold', label='7 Clients')

# #fill plot


# Fix axis to go in the right order and start at 12 o'clock.
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)

# Draw axis lines for each angle and label.
ax.set_thetagrids(np.degrees(label_loc), labels)

# Go through labels and adjust alignment based on where
# it is in the circle.
for label, angle in zip(ax.get_xticklabels(), label_loc):
    if 0 < angle < np.pi:
        label.set_horizontalalignment('left')
    else:
        label.set_horizontalalignment('right')

# Ensure radar goes from 0 to 100. it also removes the extra line
# Set the axis limits and style
ax.set_ylim(0.75, 0.81)  # Adjust the limits as needed

ax.tick_params(axis='x', labelsize=22)
ax.tick_params(axis='y', labelsize=22)
# Change the color of the circular gridlines.
ax.grid(color='#AAAAAA')
# Change the color of the outermost gridline (the spine).
ax.spines['polar'].set_color('#eaeaea')
# Change the background color inside the circle itself.
ax.set_facecolor('#FAFAFA')

# Add a legend as well.
ax.legend(loc='lower left', bbox_to_anchor=(-0.2, -0.1), fontsize=22)
plt.savefig("radar_chart_nan_ids_sl_nls.png", dpi=300, bbox_inches='tight')
plt.show()


""" Extrapolation estimation performance validation set"""
# FL = [99.3982, 99.2036, 98.8396, 98.5811, 98.2254, 97.8969, 97.5684, 97.2399, 96.9114, 96.5829, 96.2544]
# SL = [99.4874, 99.3745, 99.1917, 99.0357, 98.8463, 98.6670, 98.4876, 98.3082, 98.1289, 97.9495, 97.7701]
# SL_NLS = [98.70178, 98.487657, 98.20612, 98.1104, 97.8192, 97.5846, 97.3500, 97.1154, 96.8807, 96.6461, 96.4115]
# clients = [2, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
# # Create a figure and axis
# fig, ax = plt.subplots()
# ax.set_xlim(2, 21)
#
# prev_x = 2
# for i in range(len(clients)):
#     if clients[i] <= 7:
#         line_style = '-'
#     else:
#         line_style = '--'
#     # Connect the segments without a gap
#     ax.plot([prev_x, clients[i]], [FL[i - 1], FL[i]], line_style, label='FL', color='blue')
#     ax.plot([prev_x, clients[i]], [SL[i - 1], SL[i]], line_style, label='SL', color='red')
#     ax.plot([prev_x, clients[i]], [SL_NLS[i - 1], SL_NLS[i]], line_style, label='SL_NLS', color='gold')
#     prev_x = clients[i]
# # Add labels and title
# plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
# ax.set_xlabel('Clients', fontsize=14)
# ax.set_ylabel('Validation Accuracy (%)', fontsize=14)  # You can change this to 'Loss' if needed#
# ax.axvline(x=7, color='black', linestyle='--', label='')  # vertical line at x =7
#
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)
# ax.set_xticks([2, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21])
# # Add a legend
#
# legend_elements = [
#     mlines.Line2D([], [], color='blue', linestyle='-', label='FL'),
#     mlines.Line2D([], [], color='red', linestyle='-', label='SL'),
#     mlines.Line2D([], [], color='gold', linestyle='-', label='SL_NLS'),
#     mlines.Line2D([], [], color='blue', linestyle='--', label='FL extrapolated'),
#     mlines.Line2D([], [], color='red', linestyle='--', label='SL extrapolated'),
#     mlines.Line2D([], [], color='gold', linestyle='--', label='SL_NLS extrapolated')
# ]
# # Add the custom legend
# ax.legend(handles=legend_elements, loc='lower left')
#
# # Add a grid
# ax.grid(True)
#
#
# # Save the chart as an image
# plt.savefig("validation_accuracies_nan_ids_extrapolation.png", dpi=300, bbox_inches='tight')
# # Show the chart (optional)
# plt.show()

