import cv2
import os
import numpy as np
from PIL import Image, ImageDraw

def select_and_save_roi(image_path, config_path):
    """
    이미지에서 다각형(Polygon) ROI를 선택하고 좌표를 파일에 저장합니다.
    """
    print("ROI 설정 파일이 없습니다. ROI를 선택해주세요.")
    print("------------------------------------------------------")
    print("[마우스 좌클릭]: 점 추가")
    print("[마우스 우클릭]: 마지막 점 취소")
    print("[ENTER 키]: 선택 완료 및 저장")
    print("[c 키]: 취소 및 종료")
    print("------------------------------------------------------")
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"오류: '{image_path}' 이미지를 읽을 수 없습니다.")
        return False

    points = [] # 찍은 점들을 저장할 리스트

    # 마우스 콜백 함수
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN: # 좌클릭: 점 추가
            points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN: # 우클릭: 마지막 점 삭제
            if points:
                points.pop()

    cv2.namedWindow("Select Polygon ROI")
    cv2.setMouseCallback("Select Polygon ROI", mouse_callback)

    while True:
        display_img = img.copy()

        # 찍은 점들이 있으면 선으로 연결해서 보여줌
        if len(points) > 0:
            # 점 그리기
            for pt in points:
                cv2.circle(display_img, pt, 3, (0, 0, 255), -1)
            
            # 선 그리기 (2개 이상일 때)
            if len(points) > 1:
                pts = np.array(points, np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(display_img, [pts], False, (0, 255, 0), 2)
                
                # 마지막 점과 첫 점을 잇는 가이드 선 (다각형 닫힘 예상)
                cv2.line(display_img, points[-1], points[0], (255, 0, 0), 1)

        cv2.imshow("Select Polygon ROI", display_img)
        key = cv2.waitKey(20) & 0xFF

        if key == 13: # Enter 키
            if len(points) < 3:
                print("최소 3개의 점을 찍어야 합니다.")
                continue
            break
        elif key == ord('c'): # c 키
            print("ROI 선택이 취소되었습니다.")
            cv2.destroyAllWindows()
            return False

    cv2.destroyAllWindows()

    # 좌표 저장 (x1,y1,x2,y2,...) 형식으로 저장
    with open(config_path, 'w') as f:
        # 리스트 내의 튜플들을 풀어서 문자열로 변환
        flat_points = [str(coord) for point in points for coord in point]
        f.write(",".join(flat_points))
    
    print(f"ROI 좌표 저장 완료: {len(points)}개의 점")
    return True

def load_roi_config(config_path):
    """
    저장된 ROI 설정 파일에서 다각형 좌표를 읽어옵니다.
    """
    print("저장된 ROI 설정을 불러옵니다.")
    try:
        with open(config_path, 'r') as f:
            data = f.read().strip().split(',')
            # 1차원 리스트를 (x, y) 튜플의 리스트로 변환
            coords = [int(x) for x in data]
            points = list(zip(coords[0::2], coords[1::2]))
            return points
    except Exception as e:
        print(f"설정 파일 읽기 오류: {e}")
        return None

def apply_roi_mask(image_path, roi_points, output_path="test11_masked.jpg"):
    """
    다각형 ROI 영역 밖을 검은색으로 처리하고 결과를 저장/표시합니다.
    """
    # PIL Image 모듈로 불러오기
    try:
        original_img = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"이미지 열기 오류: {e}")
        return

    # 1. 마스크 이미지 생성 (L 모드: 흑백, 0: 검정)
    mask = Image.new("L", original_img.size, 0)
    
    # 2. 다각형 그리기 (흰색: 255으로 채움)
    draw = ImageDraw.Draw(mask)
    draw.polygon(roi_points, fill=255)

    # 3. 검은색 배경 이미지 생성
    black_bg = Image.new("RGB", original_img.size, (0, 0, 0))

    # 4. 합성 (mask가 흰색인 부분은 original_img, 검은색인 부분은 black_bg 사용)
    result_img = Image.composite(original_img, black_bg, mask)
    
    # 결과 보여주기
    result_img.show()
    
    # 저장
    result_img.save(output_path)
    print(f"처리된 이미지를 보여주고 '{output_path}'로 저장했습니다.")

def main():
    image_path = './test_data/1.jpg'
    config_path = 'roi_config.txt'

    # 이미지 파일 존재 확인
    if not os.path.exists(image_path):
        print(f"오류: '{image_path}' 파일을 찾을 수 없습니다.")
        return

    # 1. ROI 설정 파일이 없으면 ROI 선택 모드 진입
    if not os.path.exists(config_path):
        success = select_and_save_roi(image_path, config_path)
        if not success:
            return # ROI 선택 실패 또는 취소 시 종료

    # 2. ROI 설정 파일이 있으면(또는 방금 생성했으면) 불러와서 마스킹 처리
    if os.path.exists(config_path):
        roi_points = load_roi_config(config_path)
        if roi_points:
            apply_roi_mask(image_path, roi_points)

if __name__ == "__main__":
    main()