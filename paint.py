import cv2
import numpy as np
import os
from handTracker import findDistances,findError,MediapipeHands,detectCustomGesture
import json
import pickle
import time
import sys

f=open('settings.json')
settings=json.load(f)
del f

drawState='Standby'
statusState='Standby'
color='white'
brush_size=20
popup_active=False
popup_start_time=0
popup_duration=10
popup_cooldown_until=0

camera=cv2.VideoCapture(settings['camera_port'],cv2.CAP_DSHOW)
if not camera.isOpened():
    print(
        'Could not open camera. '\
        f"camera_port={settings['camera_port']} (settings.json). "
        'Try changing camera_port to 0/1/2, close other apps using the camera, '
        'and allow camera permissions in Windows Settings.'
    )
    sys.exit(1)
# If your webcam driver crops/zooms when forced to a mismatched aspect ratio,
# set camera_width/camera_height in settings.json to your camera's native mode
# (e.g. 640x480 for 4:3 or 1280x720 for 16:9). We'll letterbox to window size.
cam_w = int(settings.get('camera_width', settings['window_width']))
cam_h = int(settings.get('camera_height', settings['window_height']))
camera.set(cv2.CAP_PROP_FRAME_WIDTH, cam_w)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_h)
camera.set(cv2.CAP_PROP_FPS,settings['fps'])
camera.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG'))

cv2.namedWindow('OpenCV Paint',cv2.WINDOW_NORMAL)
if settings['fullscreen']:
    cv2.setWindowProperty('OpenCV Paint',cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

gesturenames=[]
knowngestures=[]
prevcanvas=np.zeros([settings['window_height'],settings['window_width'],3],dtype=np.uint8)
fps=0
fpsfilter=settings['fpsfilter']
starttime=time.time()
savetime=-1
run=True

if os.path.exists('gesture_data.pkl'):
    with open('gesture_data.pkl','rb') as f:
        gesturenames=pickle.load(f)
        knowngestures=pickle.load(f)
else:
    print('No gesture data found')
    sys.exit()

findhands=MediapipeHands(
    model_complexity=settings['model_complexity'],
    min_detection_confidence=settings['min_detection_confidence'],
    min_tracking_confidence=settings['min_tracking_confidence']
)
threshold=settings['confidence']
keypoints=settings['keypoints']
color_idx=['red','orange','yellow','green','cyan','blue','purple','pink','white','black']

def build_birthday_popup(width, height, elapsed):
    # Nền pastel gradient ấm áp (hồng nhạt → vàng nhạt)
    popup = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        t_val = y / max(1, height - 1)
        r = int(255 * (1 - t_val) + 255 * t_val)
        g = int(240 * (1 - t_val) + 253 * t_val)
        b = int(245 * (1 - t_val) + 231 * t_val)
        popup[y, :] = (b, g, r)  # BGR

    # ── Hoa rơi nhẹ (chấm tròn pastel) ──────────────────────────────
    petal_colors = [
        (193, 168, 249),  # tím nhạt
        (169, 230, 168),  # xanh mint
        (245, 211, 246),  # hồng pastel
        (246, 211, 151),  # vàng nhạt
        (181, 213, 249),  # xanh dương nhạt
    ]
    t_anim = float(elapsed)
    rng = np.random.RandomState(42)
    seeds_x = rng.randint(0, max(1, width), 55)
    seeds_y = rng.randint(0, max(1, height), 55)
    for i in range(55):
        x = int((seeds_x[i] + 70 * t_anim * (0.5 + (i % 3) * 0.3)) % max(1, width))
        y = int((seeds_y[i] + (80 + i % 4 * 25) * t_anim) % max(1, height))
        r = 3 + (i % 3)
        color_p = petal_colors[i % len(petal_colors)]
        cv2.circle(popup, (x, y), r, color_p, -1, lineType=cv2.LINE_AA)
        cv2.circle(popup, (x, y), r + 1, (255, 255, 255), 1, lineType=cv2.LINE_AA)

    # ── Ngôi sao trang trí 4 góc ─────────────────────────────────────
    def draw_star(img, cx, cy, size, clr):
        for angle in range(0, 360, 72):
            rad = np.radians(angle - 90)
            x1 = int(cx + size * np.cos(rad))
            y1 = int(cy + size * np.sin(rad))
            rad2 = np.radians(angle - 90 + 36)
            x2 = int(cx + size * 0.38 * np.cos(rad2))
            y2 = int(cy + size * 0.38 * np.sin(rad2))
            cv2.line(img, (cx, cy), (x1, y1), clr, 2, lineType=cv2.LINE_AA)
            cv2.line(img, (cx, cy), (x2, y2), clr, 1, lineType=cv2.LINE_AA)

    star_color = (150, 210, 246)  # vàng pastel (BGR)
    draw_star(popup, 45, 45, 22, star_color)
    draw_star(popup, width - 45, 45, 18, (200, 170, 249))
    draw_star(popup, 55, height - 55, 16, (168, 230, 168))
    draw_star(popup, width - 50, height - 50, 20, star_color)

    # ── Bánh kem giản dị ─────────────────────────────────────────────
    cake_w  = int(width * 0.38)
    cake_h  = int(height * 0.19)
    cx_cake = width // 2
    cake_x1 = cx_cake - cake_w // 2
    cake_y1 = int(height * 0.60)
    cake_x2 = cake_x1 + cake_w
    cake_y2 = cake_y1 + cake_h

    # Thân bánh
    cv2.rectangle(popup, (cake_x1, cake_y1), (cake_x2, cake_y2), (220, 200, 255), -1, lineType=cv2.LINE_AA)
    # Lớp kem trên
    cv2.rectangle(popup, (cake_x1, cake_y1), (cake_x2, cake_y1 + 20), (245, 235, 255), -1, lineType=cv2.LINE_AA)
    # Sọc trang trí
    cv2.rectangle(popup, (cake_x1, cake_y1 + 28), (cake_x2, cake_y1 + 36), (249, 168, 193), -1, lineType=cv2.LINE_AA)
    # Chấm trang trí nhỏ trên bánh
    dot_positions = [0.2, 0.4, 0.6, 0.8]
    dot_clrs = [(255, 255, 200), (200, 255, 230), (255, 200, 230), (200, 220, 255)]
    for dp, dc in zip(dot_positions, dot_clrs):
        dx = int(cake_x1 + cake_w * dp)
        dy = cake_y1 + cake_h // 2 + 8
        cv2.circle(popup, (dx, dy), 5, dc, -1, lineType=cv2.LINE_AA)

    # ── Nến + ngọn lửa nhấp nháy ─────────────────────────────────────
    candle_colors = [(193, 168, 249), (168, 230, 168), (246, 213, 152), (181, 213, 249)]
    candle_xs = [cake_x1 + cake_w // 5,
                 cake_x1 + 2 * cake_w // 5,
                 cake_x1 + 3 * cake_w // 5,
                 cake_x1 + 4 * cake_w // 5]
    flame_flicker = int(3 * np.sin(t_anim * 8.0))
    for ci, (cdx, cc) in enumerate(zip(candle_xs, candle_colors)):
        cv2.rectangle(popup, (cdx - 5, cake_y1 - 40), (cdx + 5, cake_y1), cc, -1, lineType=cv2.LINE_AA)
        fy = cake_y1 - 44 + flame_flicker + ci % 2 * 2
        cv2.ellipse(popup, (cdx, fy), (6, 11), 0, 0, 360, (100, 210, 255), -1, lineType=cv2.LINE_AA)
        cv2.ellipse(popup, (cdx, fy), (3, 6),  0, 0, 360, (200, 245, 255), -1, lineType=cv2.LINE_AA)

    # ── Tiêu đề chính ─────────────────────────────────────────────────
    title    = 'CHUC MUNG SINH NHAT!'   # không dấu vì OpenCV không hỗ trợ Unicode
    subtitle = 'Chuc ban that nhieu suc khoe va hanh phuc'
    note     = '(Happy Birthday to you!)'

    title_scale    = max(0.9, min(width, height) / 460.0)
    subtitle_scale = max(0.55, min(width, height) / 820.0)
    note_scale     = max(0.45, min(width, height) / 1000.0)

    (tw, _), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_DUPLEX, title_scale, 2)
    tx = max(10, (width - tw) // 2)
    ty = int(height * 0.22)

    # Bóng hồng nhẹ
    cv2.putText(popup, title, (tx + 2, ty + 2),
                cv2.FONT_HERSHEY_DUPLEX, title_scale, (180, 140, 200), 4, lineType=cv2.LINE_AA)
    # Chữ trắng sạch
    cv2.putText(popup, title, (tx, ty),
                cv2.FONT_HERSHEY_DUPLEX, title_scale, (255, 255, 255), 2, lineType=cv2.LINE_AA)

    # Hàng trái tim ♥ ♥ ♥
    hearts = '~ v  v  v  v  v ~'
    (hw, _), _ = cv2.getTextSize(hearts, cv2.FONT_HERSHEY_SIMPLEX, subtitle_scale, 1)
    cv2.putText(popup, hearts,
                (max(8, (width - hw) // 2), ty + int(38 * title_scale)),
                cv2.FONT_HERSHEY_SIMPLEX, subtitle_scale, (193, 140, 210), 1, lineType=cv2.LINE_AA)

    # Subtitle
    (sw, _), _ = cv2.getTextSize(subtitle, cv2.FONT_HERSHEY_SIMPLEX, subtitle_scale, 2)
    sx = max(8, (width - sw) // 2)
    sy = int(height * 0.38)
    cv2.putText(popup, subtitle, (sx + 1, sy + 1),
                cv2.FONT_HERSHEY_SIMPLEX, subtitle_scale, (100, 80, 120), 2, lineType=cv2.LINE_AA)
    cv2.putText(popup, subtitle, (sx, sy),
                cv2.FONT_HERSHEY_SIMPLEX, subtitle_scale, (255, 255, 255), 1, lineType=cv2.LINE_AA)

    # Note nhỏ (tiếng Anh phụ đề)
    (nw, _), _ = cv2.getTextSize(note, cv2.FONT_HERSHEY_SIMPLEX, note_scale, 1)
    cv2.putText(popup, note,
                (max(8, (width - nw) // 2), sy + int(28 * subtitle_scale)),
                cv2.FONT_HERSHEY_SIMPLEX, note_scale, (170, 150, 190), 1, lineType=cv2.LINE_AA)

    # ── Footer đếm ngược ──────────────────────────────────────────────
    remain = max(0, int(np.ceil(popup_duration - elapsed)))
    footer = f'Tu dong dong sau {remain}s  |  Nhan phim bat ky de dong'
    cv2.putText(popup, footer,
                (20, height - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 130, 170), 1, lineType=cv2.LINE_AA)

    return popup

def trigger_birthday_popup():
    global popup_active, popup_start_time, popup_cooldown_until
    now = time.time()
    if now < popup_cooldown_until:
        return
    popup_active = True
    popup_start_time = now
    popup_cooldown_until = now + popup_duration + 2
    cv2.namedWindow('Birthday Surprise', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Birthday Surprise', 900, 520)

def resize_with_letterbox(frame, target_w, target_h):
    """Resize preserving aspect ratio and pad to target size (no cropping)."""
    src_h, src_w = frame.shape[:2]
    if src_w == 0 or src_h == 0:
        return frame
    scale = min(target_w / src_w, target_h / src_h)
    new_w = max(1, int(src_w * scale))
    new_h = max(1, int(src_h * scale))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x0 = (target_w - new_w) // 2
    y0 = (target_h - new_h) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas

def convert_toBNW(frame):
    frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    frame=cv2.cvtColor(frame,cv2.COLOR_GRAY2BGR)
    objectFrame=np.zeros(frame.shape, dtype=np.uint8)
    frame=cv2.addWeighted(frame,.8,objectFrame,.9,0)
    return frame

def clearcanvas():
    global prevcanvas
    prevcanvas=np.zeros([settings['window_height'],settings['window_width'],3],dtype=np.uint8)

def preprocess(frame,statusState,fps):
    # UI constants
    top_h = 60
    bottom_h = 60
    side_w = 80
    w = settings['window_width']
    h = settings['window_height']

    # Translucent panels
    ui = np.zeros_like(frame)
    panel_color = (20, 20, 20)
    border_color = (60, 60, 60)
    ui_alpha = 0.55

    # Panels: top palette, right brush size, bottom status bar
    cv2.rectangle(ui, (0, 0), (w, top_h), panel_color, -1, lineType=cv2.LINE_AA)
    cv2.rectangle(ui, (w - side_w, top_h), (w, h - bottom_h), panel_color, -1, lineType=cv2.LINE_AA)
    cv2.rectangle(ui, (0, h - bottom_h), (w, h), panel_color, -1, lineType=cv2.LINE_AA)

    frame = cv2.addWeighted(frame, 1.0, ui, ui_alpha, 0)

    # Borders
    cv2.line(frame, (0, top_h), (w, top_h), border_color, 2, lineType=cv2.LINE_AA)
    cv2.line(frame, (w - side_w, top_h), (w - side_w, h - bottom_h), border_color, 1, lineType=cv2.LINE_AA)
    cv2.line(frame, (0, h - bottom_h), (w, h - bottom_h), border_color, 1, lineType=cv2.LINE_AA)

    def draw_text(text, org, scale=1.0, color=(255, 255, 255), thickness=2):
        # simple shadow for readability
        cv2.putText(frame, text, (org[0] + 2, org[1] + 2), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 1, lineType=cv2.LINE_AA)
        cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, lineType=cv2.LINE_AA)

    # Top color swatches
    swatch_count = 10
    swatch_w = w // swatch_count
    pad = 6
    for i in range(swatch_count):
        x1 = i * swatch_w + pad
        x2 = (i + 1) * swatch_w - pad
        y1 = pad
        y2 = top_h - pad
        name = color_idx[i]
        cv2.rectangle(frame, (x1, y1), (x2, y2), settings['color_swatches'][name], -1, lineType=cv2.LINE_AA)
        # subtle border, highlight selected
        is_selected = (name == color)
        thickness = 3 if is_selected else 1
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), thickness, lineType=cv2.LINE_AA)

    # Right brush size selector (6 levels)
    size_levels = settings['brush_size']
    slot_h = max(1, (h - top_h - bottom_h) // len(size_levels))
    cx = w - side_w // 2
    for idx, sz in enumerate(size_levels):
        cy = top_h + slot_h // 2 + idx * slot_h
        is_selected = (sz == brush_size)
        ring_color = (255, 255, 255) if is_selected else (180, 180, 180)
        ring_thickness = 3 if is_selected else 1
        cv2.circle(frame, (cx, cy), sz, (255, 255, 255), -1, lineType=cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), sz + 2, ring_color, ring_thickness, lineType=cv2.LINE_AA)

    # Bottom status + quick actions
    draw_text(f'{statusState}', (20, h - 20), scale=1.0)

    # Current color preview (with label if eraser)
    preview_x1 = w // 7
    preview_x2 = 2 * w // 7
    preview_y1 = h - bottom_h + 10
    preview_y2 = h - 10
    cv2.rectangle(frame, (preview_x1, preview_y1), (preview_x2, preview_y2), settings['color_swatches'][color], -1, lineType=cv2.LINE_AA)
    cv2.rectangle(frame, (preview_x1, preview_y1), (preview_x2, preview_y2), (255, 255, 255), 2, lineType=cv2.LINE_AA)
    if color == 'black':
        draw_text('Eraser', (preview_x1 + 18, h - 20), scale=0.9)

    # Current brush preview
    brush_cx = int(2 * w // 6)
    brush_cy = h - bottom_h // 2
    preview_r = brush_size - 4 if brush_size == 30 else brush_size
    cv2.circle(frame, (brush_cx, brush_cy), max(1, preview_r), (255, 255, 255), -1, lineType=cv2.LINE_AA)
    cv2.circle(frame, (brush_cx, brush_cy), max(1, preview_r) + 2, (255, 255, 255), 2, lineType=cv2.LINE_AA)

    # Hints + FPS (kept minimal)
    draw_text('C: clear', (int(2.3 * w // 6), h - 20), scale=0.9)
    draw_text('S: save', (int(3.25 * w // 6), h - 20), scale=0.9)
    draw_text('Q: quit', (int(3.48 * w // 5), h - 20), scale=0.9)
    draw_text(f'{int(fps)} FPS', (int(3.45 * w // 4), h - 20), scale=0.9)

    return frame

def saveimage():
    global savetime
    filename=''
    for i in range(6):
        filename+=f'{time.localtime()[i]}'
    cv2.imwrite(os.path.join('pictures',filename+f'.jpeg'),frame)
    savetime=time.time()

def mouseclick(event,xpos,ypos,*args,**kwargs):
    global color,brush_size,run,prevcanvas
    if event==cv2.EVENT_LBUTTONDOWN:
        if ypos>0 and ypos<60:
            if xpos>0 and xpos<settings['window_width']//10:
                color='red'
            elif xpos>settings['window_width']//10 and xpos<2*settings['window_width']//10:
                color='orange'
            elif xpos>2*settings['window_width']//10 and xpos<3*settings['window_width']//10:
                color='yellow'
            elif xpos>3*settings['window_width']//10 and xpos<4*settings['window_width']//10:
                color='green'
            elif xpos>4*settings['window_width']//10 and xpos<5*settings['window_width']//10:
                color='cyan'
            elif xpos>5*settings['window_width']//10 and xpos<6*settings['window_width']//10:
                color='blue'
            elif xpos>6*settings['window_width']//10 and xpos<7*settings['window_width']//10:
                color='purple'
            elif xpos>7*settings['window_width']//10 and xpos<8*settings['window_width']//10:
                color='pink'
            elif xpos>8*settings['window_width']//10 and xpos<9*settings['window_width']//10:
                color='white'
            else:
                color='black'
        if xpos>settings['window_width']-60 and xpos<settings['window_width'] and ypos>60 and ypos<settings['window_height']-60:
            diff=(settings['window_height']-120)//6
            if ypos>60 and ypos<60+diff:
                brush_size=5
            elif ypos>60+diff and ypos<60+2*diff:
                brush_size=10
            elif ypos>60+2*diff and ypos<60+3*diff:
                brush_size=15
            elif ypos>60+3*diff and ypos<60+4*diff:
                brush_size=20
            elif ypos>60+4*diff and ypos<60+5*diff:
                brush_size=25
            else:
                brush_size=30
        if xpos>0 and xpos<int(3.3*settings['window_width']//4) and ypos>settings['window_height']-60:
            if xpos>0 and xpos<int(3.15*settings['window_width']//6):
                clearcanvas()
            elif xpos>int(3.15*settings['window_width']//6) and xpos<int(3.4*settings['window_width']//5):
                saveimage()
            else:
                run=False

cv2.setMouseCallback('OpenCV Paint',mouseclick)

while run:
    dt=time.time()-starttime
    starttime=time.time()
    currentfps=1/dt
    fps=fps*fpsfilter+(1-fpsfilter)*currentfps
    ok,frame=camera.read()
    if not ok or frame is None:
        continue
    frame=cv2.flip(frame,1)
    # Keep full camera view (no crop) by letterboxing to the configured window.
    if frame.shape[1] != settings['window_width'] or frame.shape[0] != settings['window_height']:
        frame = resize_with_letterbox(frame, settings['window_width'], settings['window_height'])
    if settings['coloured_background']==False:
        frame=convert_toBNW(frame)
    canvas=prevcanvas
    statusState='Standby'
    handlandmarks,handstype=findhands.handsdata(frame)
    for idx,handtype in enumerate(handstype):
        if handtype==settings['command_hand']:
            distMatrix=findDistances(handlandmarks[idx])
            error,idx2=findError(knowngestures,distMatrix, keypoints)
            if error<threshold and idx!=-1:
                drawState=gesturenames[idx2]
            else:
                drawState='Standby'

            customGesture=detectCustomGesture(handlandmarks[idx])
            if customGesture is None:
                statusState=drawState
            else:
                statusState=customGesture
                if str(customGesture).lower()=='middlefinger':
                    trigger_birthday_popup()

            frame=findhands.drawLandmarks(frame, [handlandmarks[idx]],False)
            break
    if settings['command_hand'] not in handstype:
        drawState='Standby'
        statusState='Standby'
    for idx,handtype in enumerate(handstype):
        if handtype==settings['brush_hand']:
            cv2.circle(frame,(handlandmarks[idx][8][0],handlandmarks[idx][8][1]),brush_size,settings['color_swatches'][color],-1)
            #! color swatches logic
            if handlandmarks[idx][8][1]<60:
                if handlandmarks[idx][8][0]>0 and handlandmarks[idx][8][0]<settings['window_width']//10:
                    color='red'
                elif handlandmarks[idx][8][0]>settings['window_width']//10 and handlandmarks[idx][8][0]<2*settings['window_width']//10:
                    color='orange'
                elif handlandmarks[idx][8][0]>2*settings['window_width']//10 and handlandmarks[idx][8][0]<3*settings['window_width']//10:
                    color='yellow'
                elif handlandmarks[idx][8][0]>3*settings['window_width']//10 and handlandmarks[idx][8][0]<4*settings['window_width']//10:
                    color='green'
                elif handlandmarks[idx][8][0]>4*settings['window_width']//10 and handlandmarks[idx][8][0]<5*settings['window_width']//10:
                    color='cyan'
                elif handlandmarks[idx][8][0]>5*settings['window_width']//10 and handlandmarks[idx][8][0]<6*settings['window_width']//10:
                    color='blue'
                elif handlandmarks[idx][8][0]>6*settings['window_width']//10 and handlandmarks[idx][8][0]<7*settings['window_width']//10:
                    color='purple'
                elif handlandmarks[idx][8][0]>7*settings['window_width']//10 and handlandmarks[idx][8][0]<8*settings['window_width']//10:
                    color='pink'
                elif handlandmarks[idx][8][0]>8*settings['window_width']//10 and handlandmarks[idx][8][0]<9*settings['window_width']//10:
                    color='white'
                else:
                    color='black'
            #! brush size logic
            if handlandmarks[idx][8][0]>settings['window_width']-60 and handlandmarks[idx][8][0]<settings['window_width'] and handlandmarks[idx][8][1]>60 and handlandmarks[idx][8][1]<settings['window_height']-60:
                diff=(settings['window_height']-120)//6
                if handlandmarks[idx][8][1]>60 and handlandmarks[idx][8][1]<60+diff:
                    brush_size=5
                elif handlandmarks[idx][8][1]>60+diff and handlandmarks[idx][8][1]<60+2*diff:
                    brush_size=10
                elif handlandmarks[idx][8][1]>60+2*diff and handlandmarks[idx][8][1]<60+3*diff:
                    brush_size=15
                elif handlandmarks[idx][8][1]>60+3*diff and handlandmarks[idx][8][1]<60+4*diff:
                    brush_size=20
                elif handlandmarks[idx][8][1]>60+4*diff and handlandmarks[idx][8][1]<60+5*diff:
                    brush_size=25
                else:
                    brush_size=30
            #! paint logic
            if drawState=='Draw':
                cv2.circle(canvas,(handlandmarks[idx][8][0],handlandmarks[idx][8][1]),brush_size,settings['color_swatches'][color],-1)
    frame=cv2.addWeighted(frame,.6,canvas,1,1)
    frame=preprocess(frame, statusState,fps)

    if popup_active:
        elapsed = time.time() - popup_start_time
        if elapsed <= popup_duration:
            popup_frame = build_birthday_popup(settings['window_width'], settings['window_height'], elapsed)
            cv2.imshow('Birthday Surprise', popup_frame)
        else:
            popup_active = False
            cv2.destroyWindow('Birthday Surprise')

    prevcanvas=canvas
    if time.time()-savetime<=1:
        cv2.putText(frame,f'Image saved succesfully',(settings['window_width']//2-380,settings['window_height']//2),cv2.FONT_HERSHEY_SIMPLEX,2,(10,250,10),2)
    cv2.imshow('OpenCV Paint',frame)
    key=cv2.waitKey(1)
    if key & 0xff==ord('q'):
        run=False
    if key & 0xff==ord('s'):
        saveimage()
    if key & 0xff==ord('c'):
        clearcanvas()
cv2.destroyAllWindows()
camera.release()