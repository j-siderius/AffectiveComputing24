import pyxdf
import json
import os
import glob
import numpy as np
import scipy
import heartpy as hp
import matplotlib.pyplot as plt

# Enter participant data folder here:
folder = "data/test7"
# Enter participant identifier here:
title = "Test 7"


def calculate_rsq_score(rsq_results):
    rsq_results["Q1"] = 6 - rsq_results["Q1"]
    rsq_results["Q2"] = 6 - rsq_results["Q2"]
    rsq_results["Q3"] = 6 - rsq_results["Q3"]
    rsq_results["Q10"] = 6 - rsq_results["Q10"]
    return sum(rsq_results.values())


def calculate_lsl_timestamp(unix_timestamp):
    global slope, intercept
    return slope * np.float64(unix_timestamp) + intercept


def find_index(data, target_name):
    for index, element in enumerate(data):
        if element['info']['name'] and element['info']['name'][0] == target_name:
            return index
    return -1  # Return -1 if the target name is not found


measurement = glob.glob(os.path.join(folder, '*.xdf'))[0]
session_name = glob.glob(os.path.join(folder, '*json'))[0]

data, _ = pyxdf.load_xdf(measurement)
session = json.load(open(session_name))

# Figure out at which index sensors are
timestamp_index = find_index(data, 'UnixTimestampStream')
shimmer_index = find_index(data, 'Shimmer')
polar_index = find_index(data, 'PolarH10')

unix_timestamps = data[timestamp_index]["time_series"].T[0]
lsl_timestamps = data[timestamp_index]["time_stamps"]
slope, intercept, _, _, _ = scipy.stats.linregress(unix_timestamps, lsl_timestamps)

# Order: rsq0, game0, sound0, rsq1, game1, sound1, rsq2, game2, sound2, rsq3
start_timestamp_lsl = calculate_lsl_timestamp(session["startTimestamp"])
end_timestamp_lsl = calculate_lsl_timestamp(session["endTimestamp"])

sounds = [
    session["sound0"],
    session["sound1"],
    session["sound2"],
]

sound_timestamps = [
    session["soundTimestamp0"],
    session["soundTimestamp1"],
    session["soundTimestamp2"],
]
sound_timestamps_lsl = [calculate_lsl_timestamp(ts) for ts in sound_timestamps]

game_timestamps = [
    session["gameTimestamp0"],
    session["gameTimestamp1"],
    session["gameTimestamp2"],
]
game_timestamps_lsl = [calculate_lsl_timestamp(ts) for ts in game_timestamps]

rsq_timestamps = [
    session["rsqTimestamp0"],
    session["rsqTimestamp1"],
    session["rsqTimestamp2"],
    session["rsqTimestamp3"],
]
rsq_timestamps_lsl = [calculate_lsl_timestamp(ts) for ts in rsq_timestamps]

rsq_scores = [
    calculate_rsq_score(json.loads(session["RSQ0"])),
    calculate_rsq_score(json.loads(session["RSQ1"])),
    calculate_rsq_score(json.loads(session["RSQ2"])),
    calculate_rsq_score(json.loads(session["RSQ3"])),
]

ecg = data[polar_index]["time_series"].T[0]
ecg_timestamps = data[polar_index]["time_stamps"]

# Shimmer data:
# 0,1,2 = timestamps
# 3,4 = Uncalibrated ??
# 5,6 = PPG - What is the difference?
# 7 = ??
# 8 = GSR kOhm
# 9 = GSR uSiemens

shimmer_data = data[shimmer_index]["time_series"].T
shimmer_timestamps = data[shimmer_index]["time_stamps"]

ppg = shimmer_data[5]
gsr_kohm = shimmer_data[8]
gsr_usiemens = shimmer_data[9]

for i in range(13200):
    gsr_kohm[i] = 1700.0

gsr_kohm_detrend = scipy.signal.detrend(gsr_kohm)
original_trend = gsr_kohm - gsr_kohm_detrend

# plt.plot(gsr_kohm)
# plt.plot(gsr_kohm_detrend)
# plt.show()

filtered = hp.remove_baseline_wander(ecg, sample_rate=130.0)
filtered = hp.enhance_ecg_peaks(filtered, sample_rate=130.0, iterations=3)
wd, m = hp.process(filtered, sample_rate=130.0)
wd_segments, m_segments = hp.process_segmentwise(filtered, sample_rate=130.0, segment_width=10.0)
bpm = [round(x, 1) for x in m_segments["bpm"]]

# -------------------------- Output generation --------------------------

# Order: start, rsq0, game0, sound0, rsq1, game1, sound1, rsq2, game2, sound2, rsq3, end
ecgs = [
    ecg[(ecg_timestamps >= start_timestamp_lsl) & (ecg_timestamps <= rsq_timestamps_lsl[0])],  # initial relax
    ecg[(ecg_timestamps >= game_timestamps_lsl[0]) & (ecg_timestamps <= sound_timestamps_lsl[0])],  # game 1
    ecg[(ecg_timestamps >= sound_timestamps_lsl[0]) & (ecg_timestamps <= rsq_timestamps_lsl[1])],  # relax 1
    ecg[(ecg_timestamps >= game_timestamps_lsl[1]) & (ecg_timestamps <= sound_timestamps_lsl[1])],  # game 2
    ecg[(ecg_timestamps >= sound_timestamps_lsl[1]) & (ecg_timestamps <= rsq_timestamps_lsl[2])],  # relax 2
    ecg[(ecg_timestamps >= game_timestamps_lsl[2]) & (ecg_timestamps <= sound_timestamps_lsl[2])],  # game 3
    ecg[(ecg_timestamps >= sound_timestamps_lsl[2]) & (ecg_timestamps <= rsq_timestamps_lsl[3])],  # relax 3
]

hrs = list()
rmssds = list()
for temp_ecg in ecgs:
    temp_filtered = hp.remove_baseline_wander(temp_ecg, sample_rate=130.0)
    filtered = hp.enhance_ecg_peaks(temp_filtered, sample_rate=130.0, iterations=3)
    _, m = hp.process(filtered, sample_rate=130.0)
    hrs.append(round(m["bpm"], 1))
    rmssds.append(round(m["rmssd"], 2))

gsrs = [
    # gsr_kohm[(shimmer_timestamps >= start_timestamp_lsl) & (shimmer_timestamps <= rsq_timestamps_lsl[0])],
    # # initial relax
    # gsr_kohm[(shimmer_timestamps >= game_timestamps_lsl[0]) & (shimmer_timestamps <= sound_timestamps_lsl[0])],
    # # game 1
    # gsr_kohm[(shimmer_timestamps >= sound_timestamps_lsl[0]) & (shimmer_timestamps <= rsq_timestamps_lsl[1])],
    # # relax 1
    # gsr_kohm[(shimmer_timestamps >= game_timestamps_lsl[1]) & (shimmer_timestamps <= sound_timestamps_lsl[1])],
    # # game 2
    # gsr_kohm[(shimmer_timestamps >= sound_timestamps_lsl[1]) & (shimmer_timestamps <= rsq_timestamps_lsl[2])],
    # # relax 2
    # gsr_kohm[(shimmer_timestamps >= game_timestamps_lsl[2]) & (shimmer_timestamps <= sound_timestamps_lsl[2])],
    # # game 3
    # gsr_kohm[(shimmer_timestamps >= sound_timestamps_lsl[2]) & (shimmer_timestamps <= rsq_timestamps_lsl[3])],
    # # relax 3

    gsr_kohm_detrend[(shimmer_timestamps >= start_timestamp_lsl) & (shimmer_timestamps <= rsq_timestamps_lsl[0])],
    gsr_kohm_detrend[(shimmer_timestamps >= game_timestamps_lsl[0]) & (shimmer_timestamps <= sound_timestamps_lsl[0])],
    gsr_kohm_detrend[(shimmer_timestamps >= sound_timestamps_lsl[0]) & (shimmer_timestamps <= rsq_timestamps_lsl[1])],
    gsr_kohm_detrend[(shimmer_timestamps >= game_timestamps_lsl[1]) & (shimmer_timestamps <= sound_timestamps_lsl[1])],
    gsr_kohm_detrend[(shimmer_timestamps >= sound_timestamps_lsl[1]) & (shimmer_timestamps <= rsq_timestamps_lsl[2])],
    gsr_kohm_detrend[(shimmer_timestamps >= game_timestamps_lsl[2]) & (shimmer_timestamps <= sound_timestamps_lsl[2])],
    gsr_kohm_detrend[(shimmer_timestamps >= sound_timestamps_lsl[2]) & (shimmer_timestamps <= rsq_timestamps_lsl[3])],
]

gsr_trends = [
    np.polyval(np.polyfit(shimmer_timestamps[
                          np.searchsorted(shimmer_timestamps, start_timestamp_lsl):np.searchsorted(shimmer_timestamps,
                                                                                                   rsq_timestamps_lsl[
                                                                                                       0])], gsrs[0],
                          1),
               shimmer_timestamps[
               np.searchsorted(shimmer_timestamps, start_timestamp_lsl):np.searchsorted(shimmer_timestamps,
                                                                                        rsq_timestamps_lsl[0])]),
    np.polyval(np.polyfit(shimmer_timestamps[
                          np.searchsorted(shimmer_timestamps, game_timestamps_lsl[0]):np.searchsorted(
                              shimmer_timestamps, sound_timestamps_lsl[0])], gsrs[1], 1),
               shimmer_timestamps[
               np.searchsorted(shimmer_timestamps, game_timestamps_lsl[0]):np.searchsorted(shimmer_timestamps,
                                                                                           sound_timestamps_lsl[0])]),
    np.polyval(np.polyfit(shimmer_timestamps[
                          np.searchsorted(shimmer_timestamps, sound_timestamps_lsl[0]):np.searchsorted(
                              shimmer_timestamps, rsq_timestamps_lsl[1])], gsrs[2], 1),
               shimmer_timestamps[
               np.searchsorted(shimmer_timestamps, sound_timestamps_lsl[0]):np.searchsorted(shimmer_timestamps,
                                                                                            rsq_timestamps_lsl[1])]),
    np.polyval(np.polyfit(shimmer_timestamps[
                          np.searchsorted(shimmer_timestamps, game_timestamps_lsl[1]):np.searchsorted(
                              shimmer_timestamps, sound_timestamps_lsl[1])], gsrs[3], 1),
               shimmer_timestamps[
               np.searchsorted(shimmer_timestamps, game_timestamps_lsl[1]):np.searchsorted(shimmer_timestamps,
                                                                                           sound_timestamps_lsl[1])]),
    np.polyval(np.polyfit(shimmer_timestamps[
                          np.searchsorted(shimmer_timestamps, sound_timestamps_lsl[1]):np.searchsorted(
                              shimmer_timestamps, rsq_timestamps_lsl[2])], gsrs[4], 1),
               shimmer_timestamps[
               np.searchsorted(shimmer_timestamps, sound_timestamps_lsl[1]):np.searchsorted(shimmer_timestamps,
                                                                                            rsq_timestamps_lsl[2])]),
    np.polyval(np.polyfit(shimmer_timestamps[
                          np.searchsorted(shimmer_timestamps, game_timestamps_lsl[2]):np.searchsorted(
                              shimmer_timestamps, sound_timestamps_lsl[2])], gsrs[5], 1),
               shimmer_timestamps[
               np.searchsorted(shimmer_timestamps, game_timestamps_lsl[2]):np.searchsorted(shimmer_timestamps,
                                                                                           sound_timestamps_lsl[2])]),
    np.polyval(np.polyfit(shimmer_timestamps[
                          np.searchsorted(shimmer_timestamps, sound_timestamps_lsl[2]):np.searchsorted(
                              shimmer_timestamps, rsq_timestamps_lsl[3])], gsrs[6], 1),
               shimmer_timestamps[
               np.searchsorted(shimmer_timestamps, sound_timestamps_lsl[2]):np.searchsorted(shimmer_timestamps,
                                                                                            rsq_timestamps_lsl[3])]),
]

gsr_trends_amounts = [np.polyfit(shimmer_timestamps[
                                 np.searchsorted(shimmer_timestamps, start_timestamp_lsl):np.searchsorted(
                                     shimmer_timestamps, rsq_timestamps_lsl[0])], gsrs[0], 1)[0],
                      np.polyfit(shimmer_timestamps[
                                 np.searchsorted(shimmer_timestamps, game_timestamps_lsl[0]):np.searchsorted(
                                     shimmer_timestamps, sound_timestamps_lsl[0])], gsrs[1], 1)[0],
                      np.polyfit(shimmer_timestamps[
                                 np.searchsorted(shimmer_timestamps, sound_timestamps_lsl[0]):np.searchsorted(
                                     shimmer_timestamps, rsq_timestamps_lsl[1])], gsrs[2], 1)[0],
                      np.polyfit(shimmer_timestamps[
                                 np.searchsorted(shimmer_timestamps, game_timestamps_lsl[1]):np.searchsorted(
                                     shimmer_timestamps, sound_timestamps_lsl[1])], gsrs[3], 1)[0],
                      np.polyfit(shimmer_timestamps[
                                 np.searchsorted(shimmer_timestamps, sound_timestamps_lsl[1]):np.searchsorted(
                                     shimmer_timestamps, rsq_timestamps_lsl[2])], gsrs[4], 1)[0],
                      np.polyfit(shimmer_timestamps[
                                 np.searchsorted(shimmer_timestamps, game_timestamps_lsl[2]):np.searchsorted(
                                     shimmer_timestamps, sound_timestamps_lsl[2])], gsrs[5], 1)[0],
                      np.polyfit(shimmer_timestamps[
                                 np.searchsorted(shimmer_timestamps, sound_timestamps_lsl[2]):np.searchsorted(
                                     shimmer_timestamps, rsq_timestamps_lsl[3])], gsrs[6], 1)[0]
]

gsr_avgs = [abs(round(gsr.mean(), 1)) for gsr in gsrs]

fig, axs = plt.subplots(7, sharey=True, sharex=True)
fig.suptitle("Relaxation study - " + title)

for i in range(7):
    axs[i].plot(gsrs[i])
    axs[i].plot(gsr_trends[i])

# axs[0].set_title(
#     f"Initial Relax, hr={hrs[0]} bpm, rmssd={rmssds[0]}, grs_avg={gsr_avgs[0]:.1f} kΩ, RSQ: {rsq_scores[0]}")
# axs[1].set_title(f"Game 1, hr={hrs[1]} bpm, rmssd={rmssds[1]}, grs_avg={gsr_avgs[1]:.1f} kΩ")
# axs[2].set_title(
#     f"Relax 1, hr={hrs[2]} bpm, rmssd={rmssds[2]}, grs_avg={gsr_avgs[2]:.1f} kΩ, RSQ: {rsq_scores[1]}, {sounds[0]}")
# axs[3].set_title(f"Game 2, hr={hrs[3]} bpm, rmssd={rmssds[3]}, grs_avg={gsr_avgs[3]:.1f} kΩ")
# axs[4].set_title(
#     f"Relax 2, hr={hrs[4]} bpm, rmssd={rmssds[4]}, grs_avg={gsr_avgs[4]:.1f} kΩ, RSQ: {rsq_scores[2]}, {sounds[1]}")
# axs[5].set_title(f"Game 3, hr={hrs[5]} bpm, rmssd={rmssds[5]}, grs_avg={gsr_avgs[5]:.1f} kΩ")
# axs[6].set_title(
#     f"Relax 3, hr={hrs[6]} bpm, rmssd={rmssds[6]}, grs_avg={gsr_avgs[6]:.1f} kΩ, RSQ: {rsq_scores[3]}, {sounds[2]}")

axs[0].set_title(
    f"Initial Relax, hr={hrs[0]} bpm, rmssd={rmssds[0]}, grs_avg={gsr_avgs[0]:.1f} kΩ, trend={gsr_trends_amounts[0]:+.1f}, RSQ: {rsq_scores[0]}")
axs[1].set_title(
    f"Game 1, hr={hrs[1]} bpm, rmssd={rmssds[1]}, grs_avg={gsr_avgs[1]:.1f} kΩ, trend={gsr_trends_amounts[1]:+.1f}")
axs[2].set_title(
    f"Relax 1, hr={hrs[2]} bpm, rmssd={rmssds[2]}, grs_avg={gsr_avgs[2]:.1f} kΩ, trend={gsr_trends_amounts[2]:+.1f}, RSQ: {rsq_scores[1]}, {sounds[0]}")
axs[3].set_title(
    f"Game 2, hr={hrs[3]} bpm, rmssd={rmssds[3]}, grs_avg={gsr_avgs[3]:.1f} kΩ, trend={gsr_trends_amounts[3]:+.1f}")
axs[4].set_title(
    f"Relax 2, hr={hrs[4]} bpm, rmssd={rmssds[4]}, grs_avg={gsr_avgs[4]:.1f} kΩ, trend={gsr_trends_amounts[4]:+.1f}, RSQ: {rsq_scores[2]}, {sounds[1]}")
axs[5].set_title(
    f"Game 3, hr={hrs[5]} bpm, rmssd={rmssds[5]}, grs_avg={gsr_avgs[5]:.1f} kΩ, trend={gsr_trends_amounts[5]:+.1f}")
axs[6].set_title(
    f"Relax 3, hr={hrs[6]} bpm, rmssd={rmssds[6]}, grs_avg={gsr_avgs[6]:.1f} kΩ, trend={gsr_trends_amounts[6]:+.1f}, RSQ: {rsq_scores[3]}, {sounds[2]}")


for ax in axs:
    ax.set(ylabel='kΩ')

# plt.ylim(250, 3000)
plt.show()
