
# 1. 라이브러리 불러오기
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import pandas as pd
import tempfile
import os
from io import BytesIO

# 2. 웹사이트 기본 설정
st.set_page_config(
    page_title="YOLO 객체 탐지기",
    page_icon="🔍",
    layout="wide"
)

# 3. 제목
st.title("🔍 YOLO 이미지 객체 탐지기")
st.write("사진을 업로드하면 YOLO가 자동으로 객체를 탐지합니다.")

# 4. YOLO 모델 불러오기
@st.cache_resource
def load_model():
    return YOLO("yolo26n.pt")

try:
    model = load_model()
    st.success("YOLO 모델이 정상적으로 로드되었습니다.")

except Exception as e:
    st.error(f"모델 로딩 오류: {e}")
    st.stop()

# 5. 이미지 업로드
uploaded_file = st.file_uploader(
    "분석할 사진을 업로드하세요.",
    type=["jpg", "jpeg", "png"]
)

# 6. 사진 분석
if uploaded_file is not None:

    # 업로드한 이미지 열기
    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("업로드한 이미지")

    # 이미지 크기 450px로 제한
    st.image(
        image,
        width=450
    )

    # 신뢰도 설정
    confidence = st.slider(
        "탐지 신뢰도 설정",
        min_value=0.1,
        max_value=1.0,
        value=0.25,
        step=0.05
    )

    # 분석 버튼
    if st.button("🚀 객체 탐지 시작", type="primary"):

        with st.spinner("YOLO가 이미지를 분석하고 있습니다..."):

            # 임시 이미지 파일 생성
            with tempfile.NamedTemporaryFile(
                suffix=".jpg",
                delete=False
            ) as temp_file:

                image.save(temp_file.name)
                image_path = temp_file.name

            try:
                # 7. YOLO 추론
                results = model(
                    image_path,
                    conf=confidence
                )

                result = results[0]

                # 8. 탐지 결과 이미지 생성
                result_image = result.plot()

                # BGR -> RGB 변환
                result_image = result_image[:, :, ::-1]

                st.subheader("🎯 객체 탐지 결과")

                # 탐지 결과 이미지 크기 450px로 제한
                st.image(
                    result_image,
                    width=450
                )

                # 9. 탐지된 객체 정보
                boxes = result.boxes

                if len(boxes) > 0:

                    names = [
                        result.names[int(cls)]
                        for cls in boxes.cls
                    ]

                    confs = boxes.conf.cpu().tolist()

                    xyxy = boxes.xyxy.cpu().tolist()

                    # 객체별 정보 테이블
                    data = []

                    for i in range(len(names)):

                        data.append({
                            "번호": i + 1,
                            "객체 이름": names[i],
                            "신뢰도": f"{confs[i] * 100:.2f}%",
                            "좌표 (x1,y1,x2,y2)": [
                                round(v, 1) for v in xyxy[i]
                            ]
                        })

                    st.subheader("📋 탐지된 객체 목록")

                    df = pd.DataFrame(data)

                    st.dataframe(
                        df,
                        use_container_width=True,
                        hide_index=True
                    )

                    # 10. 객체 개수 출력
                    st.subheader("📊 탐지 통계")

                    counts = pd.Series(names).value_counts()

                    cols = st.columns(min(len(counts), 4))

                    for i, (name, count) in enumerate(counts.items()):

                        with cols[i % len(cols)]:
                            st.metric(name, f"{count}개")

                    st.success(
                        f"총 {len(names)}개의 객체를 탐지했습니다!"
                    )

                else:
                    st.warning("탐지된 객체가 없습니다.")

                # 11. 결과 이미지 다운로드
                output = BytesIO()

                Image.fromarray(result_image).save(
                    output,
                    format="JPEG"
                )

                st.download_button(
                    label="📥 결과 이미지 다운로드",
                    data=output.getvalue(),
                    file_name="yolo_result.jpg",
                    mime="image/jpeg"
                )

            except Exception as e:

                st.error(f"이미지 분석 중 오류 발생: {e}")

            finally:

                if os.path.exists(image_path):
                    os.remove(image_path)

# 12. 하단 안내
st.divider()

st.caption("YOLO26n 기반 이미지 객체 탐지 프로그램")
