import math


def calculate_slopes(pitch, roll):
    cross_slope = math.tan(pitch)
    running_slope = math.tan(roll)
    return cross_slope, running_slope


# Example Euler angles in radians
pitch_angle = math.radians(10)  # replace with actual pitch angle
roll_angle = math.radians(5)  # replace with actual roll angle

cross_slope, running_slope = calculate_slopes(pitch_angle, roll_angle)

print("Cross-Slope:", math.degrees(cross_slope), "degrees")
print("Running-Slope:", math.degrees(running_slope), "degrees")
