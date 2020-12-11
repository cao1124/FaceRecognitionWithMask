import cv2
from img_utils import cv_show


def main():
    # 打开摄像头
    face_cap = cv2.VideoCapture(0)

    while True:
        # Grab a single frame of video
        ret, frame = face_cap.read()
        # Quit when the input video file ends
        if not ret:
            break
        # 显示
        cv_show('face register', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            name = input('请输入您的姓名：')
            print('您的姓名是：', name)
            out_dir = 'data/face_database/nomask/' + name + '.jpg'
            cv2.imwrite(out_dir, frame)
            break
    face_cap.release()
    cv2.destroyWindow('face register')


if __name__ == '__main__':
    main()