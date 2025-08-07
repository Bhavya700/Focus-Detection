import numpy as np
from scipy.spatial import distance as dist

def calculate_ear(eye):
    """
    Calculates the Eye Aspect Ratio (EAR) for a single eye.
    The EAR is the ratio of the distances between the vertical eye landmarks 
    and the distances between the horizontal eye landmarks.

    Args:
        eye (np.array): A NumPy array of the (x, y)-coordinates of the eye landmarks.

    Returns:
        float: The calculated Eye Aspect Ratio.
    """
    # Compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # Compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])

    # Compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)

    return ear

def calculate_mar(mouth):
    """
    Calculates the Mouth Aspect Ratio (MAR) to detect yawning.
    The MAR is the ratio of the distance between the top and bottom lip
    to the distance between the corners of the mouth.

    Args:
        mouth (np.array): A NumPy array of the (x, y)-coordinates of the mouth landmarks.

    Returns:
        float: The calculated Mouth Aspect Ratio.
    """
    # Compute the euclidean distances between the vertical
    # mouth landmarks
    A = dist.euclidean(mouth[2], mouth[10]) # 51, 59
    B = dist.euclidean(mouth[4], mouth[8]) # 53, 57
    
    # Compute the euclidean distance between the horizontal
    # mouth landmark
    C = dist.euclidean(mouth[0], mouth[6]) # 49, 55

    # Compute the mouth aspect ratio
    mar = (A + B) / (2.0 * C)
    
    return mar

def get_head_pose(shape, image_dims):
    """
    Estimates the head pose (pitch, yaw, roll) from facial landmarks.

    Args:
        shape (np.array): 68 facial landmarks.
        image_dims (tuple): The dimensions of the image (height, width).

    Returns:
        tuple: A tuple containing pitch, yaw, and roll.
    """
    # 2D image points from the facial landmarks
    image_points = np.array([
        shape[30],    # Nose tip
        shape[8],     # Chin
        shape[36],    # Left eye left corner
        shape[45],    # Right eye right corner
        shape[48],    # Left Mouth corner
        shape[54]     # Right mouth corner
    ], dtype="double")

    # 3D model points. These are generic model points, not specific to the person.
    model_points = np.array([
        (0.0, 0.0, 0.0),             # Nose tip
        (0.0, -330.0, -65.0),        # Chin
        (-225.0, 170.0, -135.0),     # Left eye left corner
        (225.0, 170.0, -135.0),      # Right eye right corner
        (-150.0, -150.0, -125.0),    # Left Mouth corner
        (150.0, -150.0, -125.0)      # Right mouth corner
    ])

    # Camera internals
    focal_length = image_dims[1]
    center = (image_dims[1] / 2, image_dims[0] / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")

    # Assuming no lens distortion
    dist_coeffs = np.zeros((4, 1))

    # Solve for pose
    (success, rotation_vector, translation_vector) = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )

    # Convert rotation vector to rotation matrix
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

    # Decompose rotation matrix to get Euler angles
    # This function returns pitch, yaw, and roll
    angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rotation_matrix)

    # pitch (x), yaw (y), roll (z)
    pitch, yaw, roll = angles[0], angles[1], angles[2]

    return pitch, yaw, roll