import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

st.title("AI 체형 진단 (MediaPipe Pose)")

option = st.radio("사진 입력 방식 선택", ["사진 업로드", "웹캠 사용"])

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)
mp_drawing = mp.solutions.drawing_utils


@st.cache_data
def get_body_shape(shoulder, waist, hip):
    wsr = waist / shoulder
    hsr = hip / shoulder
    sd = abs(shoulder - hip) / ((shoulder + hip) / 2)
    awh_diff = max(shoulder, waist, hip) - min(shoulder, waist, hip)

    if wsr > 0.8 and shoulder >= hip:
        return "원형"
    elif hsr >= 1.05 and wsr <= 0.8:
        return "삼각형"
    elif awh_diff <= (max(shoulder, waist, hip) * 0.05):
        return "직사각형"
    elif sd <= 0.05 and wsr <= 0.75:
        return "모래시계"
    elif (shoulder / hip) >= 1.05 and wsr <= 0.8:
        return "역삼각형"
    else:
        return "판단불가"


# 신체 거리 계산 함수
def calc_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def extract_measurements(results, image_shape):
    if not results.pose_landmarks:
        return None

    lm = results.pose_landmarks.landmark
    h, w = image_shape

    def get_xy(part):
        return int(lm[part].x * w), int(lm[part].y * h)

    left_shoulder = get_xy(mp_pose.PoseLandmark.LEFT_SHOULDER)
    right_shoulder = get_xy(mp_pose.PoseLandmark.RIGHT_SHOULDER)
    left_hip = get_xy(mp_pose.PoseLandmark.LEFT_HIP)
    right_hip = get_xy(mp_pose.PoseLandmark.RIGHT_HIP)

    shoulder_width = calc_distance(left_shoulder, right_shoulder)
    hip_width = calc_distance(left_hip, right_hip)

    waist_y = int((left_shoulder[1] + left_hip[1]) / 2)
    waist_x_left = left_shoulder[0]
    waist_x_right = right_shoulder[0]
    waist_width = abs(waist_x_right - waist_x_left) * 0.9  # 보정값

    return shoulder_width, waist_width, hip_width


if option == "사진 업로드":
    uploaded_file = st.file_uploader("사진을 업로드하세요", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        results = pose.process(cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))

        measurements = extract_measurements(results, img_array.shape[:2])
        if measurements:
            shoulder, waist, hip = measurements
            shape = get_body_shape(shoulder, waist, hip)
            st.image(image, caption=f"판단된 체형: {shape}", use_column_width=True)
        else:
            st.warning("포즈를 인식하지 못했습니다. 전체 몸이 나온 정면 사진을 사용해주세요.")

elif option == "웹캠 사용":
    st.info("이 기능은 로컬 실행에서 지원됩니다. Streamlit Cloud에선 작동하지 않을 수 있어요!")
    run = st.checkbox('웹캠 시작')
    FRAME_WINDOW = st.image([])
    cap = cv2.VideoCapture(0)

    while run:
        ret, frame = cap.read()
        if not ret:
            st.error("카메라를 불러올 수 없습니다.")
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        measurements = extract_measurements(results, frame.shape[:2])

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        if measurements:
            shoulder, waist, hip = measurements
            shape = get_body_shape(shoulder, waist, hip)
            cv2.putText(frame, f"체형: {shape}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()
    cv2.destroyAllWindows()
