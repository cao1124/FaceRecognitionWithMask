"""
视频测试
"""
import argparse
import cv2
from face_recognizer import FaceRecognizer
from img_utils import add_chinese_text, cv_show, list_image
from apply_mask_to_faces.MaskFace import apply_mask_to_face


def main():
    parser = argparse.ArgumentParser(description='face_recognization')
    parser.add_argument('--face_db_root', type=str, default='data/mask_nomask', help='the root path of target database')
    parser.add_argument('--input_video_path', type=str, default='data/test.mp4', help='the path of input video')
    parser.add_argument('--output_video_path', type=str, default='output.mp4', help='the path of input video')

    args = parser.parse_args()
    args.face_db_root = 'data/mask_nomask'
    args.input_video_path = 'data/test.mp4'
    args.output_video_path = 'data/output.mp4'

    # 数据库不戴口罩樣本处理
    no_mask_data = list_image(args.face_db_root, '.2.')
    mask_data = list_image(args.face_db_root, '.1.')
    for i in mask_data:
        for j in no_mask_data:
            if i.split('.1.')[0] in j:
                no_mask_data.remove(i.split('.1.')[0] + '.2.jpg')
    if len(no_mask_data) > 0:
        # apply_mask_to_face(args.face_db_root, no_mask_data)
        from apply_mask_to_faces.face_mask.maskface import cli
        for i in no_mask_data:
            pic_path = str(args.face_db_root + '/' + i)
            mask_path = 'blue'
            show = False
            model = 'hog'
            cli(pic_path, mask_path, show, model)

    recognizer = FaceRecognizer()
    recognizer.create_known_faces(args.face_db_root)
    # recognizer.test_100x()

    # 测试视频路径
    # input_movie = cv2.VideoCapture(args.input_video_path)
    # 测试摄像头
    input_movie = cv2.VideoCapture(0)

    # 视频尺寸
    video_size = (1024, 720)
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    # 保存的视频路径
    output_movie = cv2.VideoWriter(args.output_video_path, fourcc, 10, video_size)

    while True:
        # Grab a single frame of video
        ret, frame = input_movie.read()
        # Quit when the input video file ends
        if not ret:
            break

        frame = cv2.resize(frame, dsize=video_size)
        item = recognizer.recognize(frame, 0.5)
        if item:
            name, (left, top, right, bottom), cls, score = item
            # font = cv2.FONT_HERSHEY_DUPLEX
            # cv2.putText(frame, "%s %.3f" % (name, score), (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)
            # puttext中文显示
            if score > 0.6:
                if cls == 0:
                    cls = '未佩戴口罩'
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                    # Draw a label with a name below the face
                    cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 0, 255), cv2.FILLED)
                    frame = add_chinese_text(frame, "%s %.3f %s" % (name, score, cls), left, bottom, (0, 255, 255), 20)
                else:
                    cls = '佩戴口罩'
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    # Draw a label with a name below the face
                    cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 255, 0), cv2.FILLED)
                    frame = add_chinese_text(frame, "%s %.3f %s" % (name, score, cls), left, bottom, (255, 0, 0), 20)
            else:
                if cls == 0:
                    cls = '未佩戴口罩'
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                    cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 0, 255), cv2.FILLED)
                    frame = add_chinese_text(frame, "%s %s" % ('RGB活体', cls), left, bottom, (0, 255, 255), 20)
                else:
                    cls = '佩戴口罩'
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 255, 0), cv2.FILLED)
                    frame = add_chinese_text(frame, "%s %s" % ('RGB活体', cls), left, bottom, (255, 0, 0), 20)
        logo = cv2.imread('data/gdpacs_logo.jpg')
        frame[frame.shape[0] - logo.shape[0]:frame.shape[0], frame.shape[1] - logo.shape[1]:frame.shape[1]] = logo
        output_movie.write(frame)
        cv_show('GD_FaceRecognitionWithMask', frame)

        # 点击小写字母q 退出程序
        if cv2.waitKey(1) == ord('q'):
            break
        # 点击窗口关闭按钮退出程序
        if cv2.getWindowProperty('GD_FaceRecognitionWithMask', cv2.WND_PROP_VISIBLE) < 1:
            break

    # All done!
    output_movie.release()
    input_movie.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()