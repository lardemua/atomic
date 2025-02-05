#!/usr/bin/env python3

def generate_skew_symmetric_4x4_matrix_from_vector(vector):

    skew_symmetric_matrix = np.zeros((4,4))
    
    skew_symmetric_matrix[0,1] = -vector[0]
    skew_symmetric_matrix[0,2] = -vector[1]
    skew_symmetric_matrix[0,3] = -vector[2]

    skew_symmetric_matrix[1,0] = vector[0]
    skew_symmetric_matrix[1,2] = vector[2]
    skew_symmetric_matrix[1,3] = -vector[1]

    skew_symmetric_matrix[2,0] = vector[1]
    skew_symmetric_matrix[2,1] = -vector[2]
    skew_symmetric_matrix[2,3] = vector[0]
    
    skew_symmetric_matrix[3,0] = vector[2]
    skew_symmetric_matrix[3,1] = vector[1]
    skew_symmetric_matrix[3,2] = -vector[0]

    return skew_symmetric_matrix



def rk4_imu_integration(imu_data_0, imu_data_1, initial_orientation):
    # This function receives two imu "data points" and integrates the angular velocity and linear acceleration to calculate the angular and linear displacements between these two points.

    # Get delta_t
    t_0 = imu_data_0["header"]["stamp"]["secs"] + (10**(-9)) * imu_data_0["header"]["stamp"]["nsecs"]
    t_1 = imu_data_1["header"]["stamp"]["secs"] + (10**(-9)) * imu_data_1["header"]["stamp"]["nsecs"]
    delta_t = t_1 - t_0


    # Orientation integration
    omega_0 = imu_data_0["angular_velocity"]
    omega_1 = imu_data_1["angular_velocity"]

    q1 = initial_orientation
    omega_m_0 = generate_skew_symmetric_4x4_matrix_from_vector(omega_0)
    k1 = (1/2) * (omega_m_0 @ q1)

    q2 = initial_orientation + delta_t*(1/2)*k1
    omega_m_halfway = generate_skew_symmetric_4x4_matrix_from_vector((omega_0 + omega_1)/2)
    k2 = (1/2) * (omega_m_halfway @ q2)

    q3 = initial_orientation + delta_t*(1/2)*k2
    k3 = (1/2) * (omega_m_halfway @ q3)

    q4 = initial_orientation + delta_t*k3
    omega_m_1 = generate_skew_symmetric_4x4_matrix_from_vector(omega_1)
    k4 = (1/2) * (omega_m_1 @ q4)

    final_orientation = initial_orientation + delta_t * ((k1/6) + (k2/3) + (k3/3) + (k4/6))

    return final_orientation

def main():
    pass