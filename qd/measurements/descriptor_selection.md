| axis | between-gait spread | replica sd | sd / range | spread / sd | \|corr\| with displacement | mutation reach (in replica sds) |
| --- | --- | --- | --- | --- | --- | --- |
| **mean |joint velocity| [rad/s]** | 0.8886 | 0.07484 | 5.6% | 11.9 | 0.93 | 13.3 |
| **actuator energy per metre [J/m]** | 39 | 3.416 | 3.8% | 11.4 | 0.72 | 13.3 |
| **cost of transport [-]** | 4.969 | 0.4353 | 3.8% | 11.4 | 0.72 | 13.3 |
| **actuator power [W]** | 2.4 | 0.2247 | 5.4% | 10.7 | 0.90 | 12.4 |
| **mean |yaw rate| [rad/s]** | 0.7911 | 0.08326 | 6.9% | 9.5 | 0.87 | 10.6 |
| **stride length [m/step]** | 0.05628 | 0.006232 | 8.0% | 9.0 | 1.00 | 9.5 |
| **mean trunk height [m]** | 0.002783 | 0.0003207 | 7.4% | 8.7 | 0.61 | 11.7 |
| **right-foot duty factor** | 0.1171 | 0.01445 | 8.6% | 8.1 | 0.82 | 8.1 |
| **trunk-height oscillation [m]** | 0.001475 | 0.0002816 | 8.9% | 5.2 | 0.83 | 7.4 |
| left-foot duty factor | 0.07 | 0.01559 | 11.4% | 4.5 | 0.66 | 5.5 |
| forward speed [m/s] | 0.2502 | 0.06727 | 11.1% | 3.7 | 1.00 | 5.1 |
| double-support fraction | 0.03714 | 0.01002 | 11.3% | 3.7 | 0.95 | 5.4 |
| **mean duty factor** | 0.02429 | 0.006856 | 9.4% | 3.5 | 0.89 | 5.4 |
| mean |lateral velocity| [m/s] | 0.1047 | 0.0317 | 10.1% | 3.3 | 0.79 | 5.0 |
| lateral drift rate [m/s] | 0.1455 | 0.05354 | 14.3% | 2.7 | 0.84 | 4.3 |
| flight fraction (both feet off) | 0.01286 | 0.005203 | 7.3% | 2.5 | 0.60 | 4.9 |
| mean tilt magnitude | 0.04357 | 0.01961 | 12.7% | 2.2 | 0.53 | 3.5 |
| step frequency [Hz] | 0.5714 | 0.2698 | 7.9% | 2.1 | 0.58 | 4.9 |
| mean forward lean (gravity_x) | 0.03022 | 0.0377 | 14.4% | 0.8 | 0.27 | 3.5 |

| axis pair | cells reached by feasible mutants | distinct cells (of 11 gaits) | \|corr\| between axes | worst spread/sd | worst mutation reach | max \|corr\| with displacement |
| --- | --- | --- | --- | --- | --- | --- |
| stride_length x torso_height_mean | 230 | 11 | 0.60 | 8.7 | 9.5 | 1.00 |
| torso_height_mean x yaw_rate | 211 | 10 | 0.83 | 8.7 | 10.6 | 0.87 |
| torso_height_mean x torso_height_osc | 211 | 8 | 0.89 | 5.2 | 7.4 | 0.83 |
| duty_right x torso_height_mean | 206 | 9 | 0.94 | 8.1 | 8.1 | 0.82 |
| torso_height_mean x joint_speed | 201 | 10 | 0.82 | 8.7 | 11.7 | 0.93 |
| stride_length x yaw_rate | 193 | 11 | 0.89 | 9.0 | 9.5 | 1.00 |
| duty_right x stride_length | 189 | 11 | 0.82 | 8.1 | 8.1 | 1.00 |
| torso_height_mean x power | 188 | 9 | 0.87 | 8.7 | 11.7 | 0.90 |
| duty_mean x torso_height_mean | 182 | 10 | 0.86 | 3.5 | 5.4 | 0.89 |
| stride_length x torso_height_osc | 181 | 10 | 0.85 | 5.2 | 7.4 | 1.00 |
| duty_right x yaw_rate | 180 | 11 | 0.96 | 8.1 | 8.1 | 0.87 |
| duty_right x torso_height_osc | 179 | 9 | 0.99 | 5.2 | 7.4 | 0.83 |
