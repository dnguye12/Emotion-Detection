import cv2
import numpy as np

relative = lambda landmark, shape: (int(landmark.x * shape[1]), int(landmark.y * shape[0]))
relativeT = lambda landmark, shape: (int(landmark.x * shape[1]), int(landmark.y * shape[0]), 0)

def gaze(frame, points):
    image_points = np.array([
        relative(points.landmark[4], frame.shape),  # Nose tip
        relative(points.landmark[152], frame.shape),  # Chin
        relative(points.landmark[263], frame.shape),  # Left eye left corner
        relative(points.landmark[33], frame.shape),  # Right eye right corner
        relative(points.landmark[287], frame.shape),  # Left Mouth corner
        relative(points.landmark[57], frame.shape)  # Right mouth corner
    ], dtype="double")
    
    image_points1 = np.array([
        relativeT(points.landmark[4], frame.shape),  # Nose tip
        relativeT(points.landmark[152], frame.shape),  # Chin
        relativeT(points.landmark[263], frame.shape),  # Left eye, left corner
        relativeT(points.landmark[33], frame.shape),  # Right eye, right corner
        relativeT(points.landmark[287], frame.shape),  # Left Mouth corner
        relativeT(points.landmark[57], frame.shape)  # Right mouth corner
    ], dtype="double")
    
    model_points = np.array([
        (0.0, 0.0, 0.0),  # Nose tip
        (0, -63.6, -12.5),  # Chin
        (-43.3, 32.7, -26),  # Left eye, left corner
        (43.3, 32.7, -26),  # Right eye, right corner
        (-28.9, -28.9, -24.1),  # Left Mouth corner
        (28.9, -28.9, -24.1)  # Right mouth corner
    ])
    
    Eye_ball_center_right = np.array([[-29.05], [32.7], [-39.5]])
    Eye_ball_center_left = np.array([[29.05], [32.7], [-39.5]])  # the center of the left eyeball as a vector.
    
    focal_length = frame.shape[1]
    center = (frame.shape[1] / 2, frame.shape[0] / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )

    dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                                  dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

    # 2d pupil location
    left_pupil = relative(points.landmark[468], frame.shape)
    right_pupil = relative(points.landmark[473], frame.shape)

    # Transformation between image point to world point
    _, transformation, _ = cv2.estimateAffine3D(image_points1, model_points)  # image to world transformation

    if transformation is not None:  # if estimateAffine3D succeeded
        # Left eye gaze
        pupil_world_cord_left = transformation @ np.array([[left_pupil[0], left_pupil[1], 0, 1]]).T
        S_left = Eye_ball_center_left + (pupil_world_cord_left - Eye_ball_center_left) * 10
        (eye_pupil2D_left, _) = cv2.projectPoints((int(S_left[0]), int(S_left[1]), int(S_left[2])), rotation_vector,
                                                  translation_vector, camera_matrix, dist_coeffs)
        (head_pose_left, _) = cv2.projectPoints((int(pupil_world_cord_left[0]), int(pupil_world_cord_left[1]), int(40)),
                                                rotation_vector, translation_vector, camera_matrix, dist_coeffs)
        gaze_left = left_pupil + (eye_pupil2D_left[0][0] - left_pupil) - (head_pose_left[0][0] - left_pupil)
        p1_left = (int(left_pupil[0]), int(left_pupil[1]))
        p2_left = (int(gaze_left[0]), int(gaze_left[1]))
        #cv2.line(frame, p1_left, p2_left, (0, 0, 255), 2)

        # Right eye gaze
        pupil_world_cord_right = transformation @ np.array([[right_pupil[0], right_pupil[1], 0, 1]]).T
        S_right = Eye_ball_center_right + (pupil_world_cord_right - Eye_ball_center_right) * 10
        (eye_pupil2D_right, _) = cv2.projectPoints((int(S_right[0]), int(S_right[1]), int(S_right[2])), rotation_vector,
                                                   translation_vector, camera_matrix, dist_coeffs)
        (head_pose_right, _) = cv2.projectPoints((int(pupil_world_cord_right[0]), int(pupil_world_cord_right[1]), int(40)),
                                                 rotation_vector, translation_vector, camera_matrix, dist_coeffs)
        gaze_right = right_pupil + (eye_pupil2D_right[0][0] - right_pupil) - (head_pose_right[0][0] - right_pupil)
        p1_right = (int(right_pupil[0]), int(right_pupil[1]))
        p2_right = (int(gaze_right[0]), int(gaze_right[1]))
        #cv2.line(frame, p1_right, p2_right, (0, 0, 255), 2)
        
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        forward_direction = np.array([[0,0,1]], dtype="double").T
        head_direction = rotation_matrix @ forward_direction
        
        head_direction_norm = head_direction / np.linalg.norm(head_direction)
        
        #Camera's viewing direction (z-axis)
        camera_viewing_direction = np.array([0, 0, 1])

        # Calculate the cosine of the angle using the dot product
        cos_angle = np.dot(head_direction_norm.flatten(), camera_viewing_direction)

        # Calculate the angle in degrees
        angle = np.degrees(np.arccos(cos_angle))

        # Define a threshold angle (e.g., 15 degrees)
        threshold_angle = 30

        # Determine if the user is looking at the camera
        is_looking_at_camera = angle < threshold_angle
        
        #print(is_looking_at_camera)
        
        nose_tip = image_points[0]
        
        head_direction_point = (int(nose_tip[0] + head_direction[0]), int(nose_tip[1] + head_direction[1]))
        
        return (p1_left, p2_left, p1_right, p2_right, is_looking_at_camera)
