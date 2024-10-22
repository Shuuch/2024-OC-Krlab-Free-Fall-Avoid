import cv2
import numpy as np
import random
from ultralytics import YOLO
import time
import pygame
# ウェブカメラのを起動させる関数
def  camera_setup():
    global model, cap, frame_width, frame_height


 # YOLOモデルを読み込む
    model = YOLO("runs/detect/train/weights/best.pt")
    pygame.init()
    pygame.mixer.init()
    # ウェブカメラのキャプチャを開始する
    cap = cv2.VideoCapture(0)

    # ウェブカメラが正しく開けない場合は終了する
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()

    # ウェブカメラのフレームサイズを取得する
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return
# 開始時、終了時、ゲームクリア時の画像を読み込む関数
def image_loading():
    global start_image, end_image, clear_image
    # 開始時、終了時、ゲームクリア時の画像を読み込む
    start_image = cv2.imread('extra start.png')
    end_image = cv2.imread('gameover3.png')
    clear_image = cv2.imread('gameclear.png')  # ゲームクリアの画像
    start_image = cv2.resize(start_image, (frame_width, frame_height))
    end_image = cv2.resize(end_image, (frame_width, frame_height))
    clear_image = cv2.resize(clear_image, (frame_width, frame_height))
    return
# 障害物に使用する画像を読み込む関数
def obstacle_loading():
    global obstacle_image, obstacle_height, obstacle_width, obstacle_center_x1, obstacle_center_y1
    global obstacle_speed1, obstacle_radius1, obstacle_center_x2, obstacle_center_y2, obstacle_speed2, obstacle_radius2
    # 障害物の初期設定
    obstacle_image = cv2.imread('iwa_transparent.png', cv2.IMREAD_UNCHANGED)

    # 障害物の画像をリサイズする
    obstacle_image = cv2.resize(obstacle_image, (100, 80))
    obstacle_height, obstacle_width = obstacle_image.shape[:2]

    # 障害物の設定
    obstacle_radius1 = random.randint(20, 30)
    obstacle_center_x1 = random.randint(obstacle_radius1, frame_width - obstacle_radius1)
    obstacle_center_y1 = 0

    obstacle_radius2 = random.randint(20, 30)
    obstacle_center_x2 = random.randint(obstacle_radius2, frame_width - obstacle_radius2)
    obstacle_center_y2 = 0

    # 障害物の速度をそれぞれ設定する
    obstacle_speed1 = 20  # 障害物1の移動速度
    obstacle_speed2 = 10  # 障害物2の移動速度

    return
# 衝突回数とタイマーなどのカウンターを定義する関数
def game_count_setup():
    global collision_count, collision_limit, countdown_time, start_time, delay_start_time, bar2_start
    global delay_duration, show_end_image, show_clear_image, bar_reduce, hp_bar_count, bar2_reduce, hp_bar2_count
    # 衝突回数をカウントする変数
    collision_count = 0
    collision_limit = 102

    # HP バーの本数を保存する変数
    bar_reduce = True
    bar2_reduce = False
    hp_bar_count = 50
    hp_bar2_count = 50
    bar2_start = False

    # カウントダウンの時間設定（61秒）
    countdown_time = 61
    start_time = None  # カウントダウン開始時刻
    delay_start_time = None  # `Z`キーが押された時刻
    delay_duration = 3  # カウントダウン開始までの遅延時間（秒）

    # 終了時に表示する画像のフラグ
    show_end_image = False
    show_clear_image = False
    return
# OpenCVのウィンドウを作成し、スタート画面を作成する関数
def make_window():
    global delay_start_time
    cv2.namedWindow('Webcam', cv2.WINDOW_NORMAL)
    # 処理開始時に特定の画像を表示し、「Z」キーを押すことでメインの処理に移行
    while True:
        cv2.imshow('Webcam', start_image)
        key = cv2.waitKey(1)
        if key == 122:  # 'Z'キーのASCIIコード
            delay_start_time = time.time()  # `Z`キーが押された時刻を記録
#            playsound('gamestart.mp3')  # ゲーム開始時の音声を再生
            break
#  障害物を描画する関数
def draw_obstacle(frame, x, y):
    if obstacle_image.shape[2] == 4:  # アルファチャンネルが存在する場合
        x1 = max(x - obstacle_width // 2, 0)
        y1 = max(y - obstacle_height // 2, 0)
        x2 = min(x1 + obstacle_width, frame.shape[1])
        y2 = min(y1 + obstacle_height, frame.shape[0])

        # 障害物画像の合成範囲を計算
        obstacle_img_part = obstacle_image[
            max(0, obstacle_height // 2 - y):min(obstacle_height, frame.shape[0] - y + obstacle_height // 2),
            max(0, obstacle_width // 2 - x):min(obstacle_width, frame.shape[1] - x + obstacle_width // 2),
            :
        ]

        # 合成部分のサイズを調整
        part_height, part_width = obstacle_img_part.shape[:2]
        overlay_part = frame[y1:y2, x1:x2]

        if part_height > 0 and part_width > 0 and (overlay_part.shape[0] != part_height or overlay_part.shape[1] != part_width):
            obstacle_img_part = cv2.resize(obstacle_img_part, (overlay_part.shape[1], overlay_part.shape[0]))
            part_height, part_width = obstacle_img_part.shape[:2]

        alpha_s = obstacle_img_part[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s

        for c in range(3):  # RGBチャンネルに対して合成
            overlay_part[:, :, c] = (alpha_s * obstacle_img_part[:, :, c] + alpha_l * overlay_part[:, :, c])
    else:
        cv2.circle(frame, (x, y), obstacle_radius1, (0, 0, 255), -1)
# 障害物と物体のバウンディングボックスが重なるかどうかをチェックする関数
def is_collision(obstacle_center_x, obstacle_center_y, obstacle_radius, box):
    x1, y1, x2, y2 = [int(i) for i in box.xyxy[0]]
    closest_x = np.clip(obstacle_center_x, x1, x2)
    closest_y = np.clip(obstacle_center_y, y1, y2)
    distance = np.sqrt((closest_x - obstacle_center_x) ** 2 + (closest_y - obstacle_center_y) ** 2)
    return distance < obstacle_radius
# ウィンドウを閉じる関数
def close_window():
    # 後片付け
    cap.release()
    cv2.destroyAllWindows()


camera_setup()
image_loading()
obstacle_loading()
game_count_setup()
make_window()

# ゲーム開始時の音声が再生された後にBGMを再生する
pygame.mixer.music.load('gamebgm.mp3')
pygame.mixer.music.play(-1)  # ループ再生

# メインのループ
while True:
    # ウェブカメラからフレームを読み込む
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if not ret:
        print("Error: Failed to capture image")
        break

    # YOLOで物体検出を行う
    results = model(frame)

    # 現在の時間を取得し、カウントダウンを計算
    current_time = time.time()
    if delay_start_time is not None:
        if current_time - delay_start_time >= delay_duration:
            if start_time is None:  # 3秒の遅延後にカウントダウンを開始
                start_time = current_time
        else:
            remaining_delay = int(delay_duration - (current_time - delay_start_time))
            annotatedFrame = frame.copy()
            cv2.putText(annotatedFrame, f"Countdown starts in: {remaining_delay}", (frame_width // 2 - 150, frame_height // 2), cv2.FONT_HERSHEY_PLAIN, 2.0, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.imshow('Webcam', annotatedFrame)
            if cv2.waitKey(1) == 27:
                break
            continue
    else:
        remaining_time = countdown_time
    if start_time is not None:
        elapsed_time = current_time - start_time
        remaining_time = max(0, int(countdown_time - elapsed_time))

    # YOLOの検出結果を処理する
    if results:
        boxes = results[0].boxes
        classes = results[0].boxes.cls
        annotatedFrame = frame
        for box, cls in zip(boxes, classes):
            if is_collision(obstacle_center_x1, obstacle_center_y1, obstacle_radius1, box) or is_collision(obstacle_center_x2, obstacle_center_y2, obstacle_radius2, box):
                collision_count += 1

                if collision_count < 100:
                    if bar2_start == False:
                        hp_bar_count -=1
                        if collision_count == 50:
                            bar2_start = True
                    else:
                        hp_bar2_count -=1



                # if bar_reduce == True:
                #     hp_bar_count -= 1
                #     bar_reduce = False
                # else:
                #     bar_reduce = True
                #     if collision_count ==100:
                #         bar2_start = True

                # if bar2_start == True:
                #     if bar2_reduce == True:
                #         hp_bar2_count -= 1
                #         bar2_reduce = False
                #     else:
                #         bar2_reduce = True

                #print(f"Collision count: {collision_count}")

                elif collision_count >= collision_limit:
                    print("Collision limit reached! Exiting...")
                    show_end_image = True
                    break
    else:
        annotatedFrame = frame

    # 衝突回数とカウントダウンを表示する
    #cv2.putText(annotatedFrame, f"Collisions: {collision_count}", (10, 30), cv2.FONT_HERSHEY_PLAIN, 2.0, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(annotatedFrame, f"HP [", (5, frame_height - 20), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(annotatedFrame, f"|" * (hp_bar2_count), (70, frame_height - 20), cv2.FONT_HERSHEY_PLAIN, 2.0, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(annotatedFrame, f"|" * (hp_bar_count), (70, frame_height - 20), cv2.FONT_HERSHEY_PLAIN, 2.0, (255, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(annotatedFrame, f"]", (frame_width - 170, frame_height - 20), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(annotatedFrame, f"Time: {remaining_time}", (frame_width - 150, frame_height - 20), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2, cv2.LINE_AA)

    # 障害物を描画する
    draw_obstacle(annotatedFrame, obstacle_center_x1, obstacle_center_y1)
    draw_obstacle(annotatedFrame, obstacle_center_x2, obstacle_center_y2)

    # 障害物を下に移動する
    obstacle_center_y1 += obstacle_speed1
    obstacle_center_y2 += obstacle_speed2

    # 障害物が画面の下端に達したら、再度上部に配置する
    if obstacle_center_y1 > frame_height + obstacle_radius1:
        obstacle_center_x1 = random.randint(obstacle_radius1, frame_width - obstacle_radius1)
        obstacle_center_y1 = 0

    if obstacle_center_y2 > frame_height + obstacle_radius2:
        obstacle_center_x2 = random.randint(obstacle_radius2, frame_width - obstacle_radius2)
        obstacle_center_y2 = 0

    # ウェブカメラの映像を表示する
    cv2.imshow('Webcam', annotatedFrame)

    # カウントダウンが0になったか、衝突回数が上限に達したら終了画像を表示する
    if remaining_time <= 0:
        show_clear_image = True
        pygame.mixer.music.stop()  # BGMを停止
        #playsound('gameclear.mp3')  # ゲームクリア時の音声を再生
        pygame.mixer.music.load('gameclear.mp3')
        pygame.mixer.music.play(1)


    if show_clear_image:
        cv2.imshow('Webcam', clear_image)
        cv2.waitKey(0)  # ユーザーがキーを押すまで待機する
        break

    if show_end_image:
        pygame.mixer.music.stop()  # BGMを停止
        #playsound('gameover.mp3')  # ゲームオーバー時の音声を再生
        pygame.mixer.music.load('gameover.mp3')
        pygame.mixer.music.play(1)
        cv2.imshow('Webcam', end_image)
        cv2.waitKey(0)  # ユーザーがキーを押すまで待機する
        break

    # ESCキーを押したら終了する
    if cv2.waitKey(1) == 27:
        break

close_window()