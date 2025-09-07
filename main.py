import csv
import cv2 as cv
from cvzone.HandTrackingModule import HandDetector
import time

# -------------------- Camera & Detector --------------------
cap = cv.VideoCapture(0)
cap.set(3, 1920)
cap.set(4, 1080)

detector = HandDetector(detectionCon=0.8)

# -------------------- Palette helper --------------------
def hex_to_bgr(hexstr: str):
    """Convert '#RRGGBB' to (B, G, R) tuple for OpenCV."""
    h = hexstr.lstrip('#')
    r = int(h[0:2], 16)
    g = int(h[2:4], 16)
    b = int(h[4:6], 16)
    return (b, g, r)

COLOR_NAVY = hex_to_bgr("#001741")
COLOR_BLUE = hex_to_bgr("#B1CCDE")
COLOR_WHITE = hex_to_bgr("#EFEFEF")
COLOR_RED = hex_to_bgr("#770000")

SELECTION_DELAY = 4.0  # seconds

# -------------------- Text helpers --------------------
def wrap_text_by_pixel(text, font, scale, thickness, max_width):
    """
    Wrap `text` into lines whose pixel width <= max_width,
    using cv.getTextSize for accurate measurements.
    """
    if not text:
        return [""]

    words = text.split()
    lines = []
    current = ""

    for word in words:
        test = (current + " " + word).strip() if current else word
        w = cv.getTextSize(test, font, scale, thickness)[0][0]
        if w <= max_width:
            current = test
        else:
            if current:
                lines.append(current)
            # if the single word itself is too wide -> split it by characters
            word_width = cv.getTextSize(word, font, scale, thickness)[0][0]
            if word_width <= max_width:
                current = word
            else:
                # greedy char-splitting
                part = ""
                for ch in word:
                    t = part + ch
                    if cv.getTextSize(t, font, scale, thickness)[0][0] <= max_width:
                        part = t
                    else:
                        if part:
                            lines.append(part)
                        part = ch
                current = part
    if current:
        lines.append(current)
    return lines

def draw_multiline_text_rect(frame, text_lines, pos, scale, thickness, offset, border, colorB, colorT, font):
    """
    Draw filled rectangle + multiple lines of text inside. Returns (frame, bbox).
    bbox = (x1, y1, x2, y2)
    """
    x, y = pos
    sizes = [cv.getTextSize(line, font, scale, thickness)[0] for line in text_lines]
    max_w = max(w for w,h in sizes)
    line_h = max(h for w,h in sizes)
    spacing = int(line_h * 0.4)

    total_height = len(text_lines) * (line_h + spacing) - spacing + 2 * offset
    total_width = max_w + 2 * offset

    # background
    cv.rectangle(frame, (x - border, y - border), (x + total_width + border, y + total_height + border), colorB, -1)

    # text lines
    ty = y + offset + line_h
    for line in text_lines:
        cv.putText(frame, line, (x + offset, ty), font, scale, colorT, thickness, cv.LINE_AA)
        ty += line_h + spacing

    return frame, (x - border, y - border, x + total_width + border, y + total_height + border)

def draw_singleline_text_rect(frame, text, pos, scale, thickness, offset, border, colorB, colorT, font):
    return draw_multiline_text_rect(frame, [text], pos, scale, thickness, offset, border, colorB, colorT, font)

# -------------------- MCQ class --------------------
class MCQ():
    def __init__(self, data):
        self.question = data[0]
        self.choice1 = data[1]
        self.choice2 = data[2]
        self.choice3 = data[3]
        self.choice4 = data[4]
        self.answer = int(data[5])
        self.userAns = None

    def update(self, cursor, bboxs):
        for i, box in enumerate(bboxs):
            x1, y1, x2, y2 = box
            if x1 < cursor[0] < x2 and y1 < cursor[1] < y2:
                self.userAns = i + 1
                # selection border (navy)
                cv.rectangle(frame, (x1, y1), (x2, y2), COLOR_NAVY, 8)

# -------------------- Load questions --------------------
getFile = "Que.csv"
with open(getFile, newline='\n') as f:
    reader = csv.reader(f)
    datafile = list(reader)[1:]

mcqList = [MCQ(q) for q in datafile]
quesNumber = 0
qTotal = len(datafile)

last_selection_time = time.time() - SELECTION_DELAY

# -------------------- Window setup --------------------
cv.namedWindow("IEEE ENSI SB Quiz", cv.WINDOW_NORMAL)
cv.resizeWindow("IEEE ENSI SB Quiz", 1280, 720)

# -------------------- Main loop --------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera read failed.")
        break

    # Remove cvzone's drawing to avoid unexpected colors
    hands, frame = detector.findHands(frame, draw=False, flipType=True)

    frame_h, frame_w = frame.shape[0], frame.shape[1]
    margin_x = int(0.04 * frame_w)
    margin_y = int(0.03 * frame_h)

    if quesNumber < qTotal:
        mcq = mcqList[quesNumber]

        # ---- Header ----
        header_scale = 1.2
        header_th = 2
        frame, header_box = draw_singleline_text_rect(
            frame, f"Total Questions: {qTotal}", [margin_x, margin_y],
            scale=header_scale, thickness=header_th, offset=12, border=4,
            colorB=COLOR_NAVY, colorT=COLOR_WHITE, font=cv.FONT_HERSHEY_DUPLEX
        )

        # compute top for question area
        q_top = header_box[3] + margin_y

        # ---- Question area (dynamic scale & wrapping) ----
        question_area_width = frame_w - 2 * margin_x
        # give question area around 25-30% of frame height (adjustable)
        max_question_height = int(0.28 * frame_h)

        # try scales from large to small to fit vertically and horizontally
        q_font = cv.FONT_HERSHEY_COMPLEX
        q_th = 2
        q_offset = 20
        q_border = 6
        chosen_q_scale = None
        q_scales_try = [1.6, 1.4, 1.2, 1.0, 0.9, 0.8, 0.7]
        for sc in q_scales_try:
            q_lines = wrap_text_by_pixel(mcq.question, q_font, sc, q_th, question_area_width - 2 * q_offset)
            sizes = [cv.getTextSize(l, q_font, sc, q_th)[0] for l in q_lines]
            if sizes:
                line_h = max(h for w,h in sizes)
            else:
                line_h = cv.getTextSize("T", q_font, sc, q_th)[0][1]
            spacing = int(line_h * 0.4)
            total_height = len(q_lines) * (line_h + spacing) - spacing + 2 * q_offset
            if total_height <= max_question_height:
                chosen_q_scale = sc
                chosen_q_lines = q_lines
                break
        if chosen_q_scale is None:  # fallback: smallest scale and allow vertical scroll-ish clipping
            chosen_q_scale = q_scales_try[-1]
            chosen_q_lines = wrap_text_by_pixel(mcq.question, q_font, chosen_q_scale, q_th, question_area_width - 2 * q_offset)

        frame, qbox = draw_multiline_text_rect(
            frame, chosen_q_lines, [margin_x, q_top],
            scale=chosen_q_scale, thickness=q_th, offset=q_offset, border=q_border,
            colorB=COLOR_BLUE, colorT=COLOR_NAVY, font=q_font
        )

        # ---- Choices area (two columns) ----
        # columns width:
        col_gap = int(0.04 * frame_w)
        column_width = int((frame_w - 3 * margin_x) / 2)  # two columns + middle margin_x
        choices_top = qbox[3] + margin_y
        # reserve space for progress bar at bottom
        progress_reserve = int(0.15 * frame_h)
        available_height_for_choices = frame_h - choices_top - progress_reserve - margin_y

        choice_font = cv.FONT_HERSHEY_SIMPLEX
        choice_th = 2
        choice_offset = 15
        choice_border = 4

        choices = [mcq.choice1, mcq.choice2, mcq.choice3, mcq.choice4]
        # try scales to fit choices vertically in two rows
        choice_scale_candidates = [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7]
        chosen_choice_scale = None
        chosen_choice_lines = None
        for cs in choice_scale_candidates:
            # wrap each choice to its column width
            wrapped = [wrap_text_by_pixel(c, choice_font, cs, choice_th, column_width - 2 * choice_offset) for c in choices]
            # compute bbox heights
            heights = []
            for lines in wrapped:
                if lines:
                    sizes = [cv.getTextSize(l, choice_font, cs, choice_th)[0] for l in lines]
                    line_h = max(h for w,h in sizes)
                    spacing = int(line_h * 0.4)
                    total_h = len(lines) * (line_h + spacing) - spacing + 2 * choice_offset
                else:
                    # empty fallback
                    total_h = int(30 * cs) + 2 * choice_offset
                heights.append(total_h)
            # top row height and bottom row height
            row1_h = max(heights[0], heights[1])
            row2_h = max(heights[2], heights[3])
            vertical_gap_between_rows = int(0.07 * frame_h)
            total_needed = row1_h + row2_h + vertical_gap_between_rows
            if total_needed <= available_height_for_choices:
                chosen_choice_scale = cs
                chosen_choice_lines = wrapped
                chosen_choice_heights = heights
                break
        if chosen_choice_scale is None:
            chosen_choice_scale = choice_scale_candidates[-1]
            chosen_choice_lines = [wrap_text_by_pixel(c, choice_font, chosen_choice_scale, choice_th, column_width - 2 * choice_offset) for c in choices]
            sizes = []
            for lines in chosen_choice_lines:
                if lines:
                    s = [cv.getTextSize(l, choice_font, chosen_choice_scale, choice_th)[0] for l in lines]
                    line_h = max(h for w,h in s)
                    spacing = int(line_h * 0.4)
                    total_h = len(lines) * (line_h + spacing) - spacing + 2 * choice_offset
                else:
                    total_h = int(30 * chosen_choice_scale) + 2 * choice_offset
                sizes.append(total_h)
            chosen_choice_heights = sizes

        # compute exact positions for each choice
        col1_x = margin_x
        col2_x = margin_x + column_width + margin_x  # using margin_x as column gap
        # row1 y start
        y_row = choices_top
        # compute heights for row 1 & 2
        row1_h = max(chosen_choice_heights[0], chosen_choice_heights[1])
        row2_h = max(chosen_choice_heights[2], chosen_choice_heights[3])
        # draw row1 col1
        frame, box1 = draw_multiline_text_rect(
            frame, chosen_choice_lines[0], [col1_x, y_row],
            scale=chosen_choice_scale, thickness=choice_th, offset=choice_offset, border=choice_border,
            colorB=COLOR_WHITE, colorT=COLOR_NAVY, font=choice_font
        )
        frame, box2 = draw_multiline_text_rect(
            frame, chosen_choice_lines[1], [col2_x, y_row],
            scale=chosen_choice_scale, thickness=choice_th, offset=choice_offset, border=choice_border,
            colorB=COLOR_WHITE, colorT=COLOR_NAVY, font=choice_font
        )
        # row2 y
        y_row2 = y_row + row1_h + int(0.07 * frame_h)
        frame, box3 = draw_multiline_text_rect(
            frame, chosen_choice_lines[2], [col1_x, y_row2],
            scale=chosen_choice_scale, thickness=choice_th, offset=choice_offset, border=choice_border,
            colorB=COLOR_WHITE, colorT=COLOR_NAVY, font=choice_font
        )
        frame, box4 = draw_multiline_text_rect(
            frame, chosen_choice_lines[3], [col2_x, y_row2],
            scale=chosen_choice_scale, thickness=choice_th, offset=choice_offset, border=choice_border,
            colorB=COLOR_WHITE, colorT=COLOR_NAVY, font=choice_font
        )

        choice_boxes = [box1, box2, box3, box4]

        # ---- Hand interaction ----
        if hands:
            lmList = hands[0]["lmList"]
            cursor = lmList[8][:2]
            cursor2 = lmList[12][:2]
            length, _, _ = detector.findDistance(cursor, cursor2)

            # subtle cursor
            cv.circle(frame, tuple(cursor), 10, COLOR_NAVY, -1)
            cv.circle(frame, tuple(cursor), 12, COLOR_WHITE, 2)

            if 20 <= length <= 30 and (time.time() - last_selection_time) >= SELECTION_DELAY:
                mcq.update(cursor, choice_boxes)
                if mcq.userAns is not None:
                    last_selection_time = time.time()
                    quesNumber += 1

    else:
        # Quiz ended screen (keeps palette)
        score = sum(1 for m in mcqList if m.userAns == m.answer)
        score = round((score / qTotal) * 100, 2)

        frame, _ = draw_singleline_text_rect(
            frame, f"Total Questions Solved: {qTotal}", [int(0.18 * frame_w), int(0.14 * frame_h)],
            scale=1.4, thickness=2, offset=12, border=4,
            colorB=COLOR_NAVY, colorT=COLOR_WHITE, font=cv.FONT_HERSHEY_DUPLEX
        )
        frame, _ = draw_singleline_text_rect(
            frame, "Your Quiz Completed", [int(0.28 * frame_w), int(0.22 * frame_h)],
            scale=1.6, thickness=2, offset=16, border=5,
            colorB=COLOR_BLUE, colorT=COLOR_NAVY, font=cv.FONT_HERSHEY_COMPLEX
        )
        frame, _ = draw_singleline_text_rect(
            frame, f"Your Score: {score}%", [int(0.28 * frame_w), int(0.30 * frame_h)],
            scale=1.6, thickness=2, offset=16, border=5,
            colorB=COLOR_RED, colorT=COLOR_WHITE, font=cv.FONT_HERSHEY_DUPLEX
        )
        frame, _ = draw_singleline_text_rect(
            frame, "Press Q to Exit", [int(0.18 * frame_w), int(0.40 * frame_h)],
            scale=1.2, thickness=2, offset=12, border=4,
            colorB=COLOR_WHITE, colorT=COLOR_NAVY, font=cv.FONT_HERSHEY_SIMPLEX
        )

    # ---- Progress bar (navy fill + white frame) ----
    if qTotal > 0:
        progress_pct = (quesNumber / qTotal)
    else:
        progress_pct = 0.0
    progress_left = margin_x
    progress_right = frame_w - margin_x - int(0.12 * frame_w)  # leave room for percent box
    progress_y1 = frame_h - int(0.12 * frame_h)
    progress_y2 = progress_y1 + int(0.03 * frame_h)
    filled_x = progress_left + int((progress_right - progress_left) * progress_pct)
    cv.rectangle(frame, (progress_left, progress_y1), (filled_x, progress_y2), COLOR_NAVY, cv.FILLED)
    cv.rectangle(frame, (progress_left, progress_y1), (progress_right, progress_y2), COLOR_WHITE, max(3, int(0.005*frame_h)))

    # percent box
    pct_text = f"{int(progress_pct * 100)}%"
    pct_x = progress_right + int(0.02 * frame_w)
    pct_y = progress_y1 - int(0.01 * frame_h)
    frame, _ = draw_singleline_text_rect(
        frame, pct_text, [pct_x, pct_y],
        scale=1.2, thickness=2, offset=10, border=4,
        colorB=COLOR_NAVY, colorT=COLOR_WHITE, font=cv.FONT_HERSHEY_DUPLEX
    )

    cv.imshow("IEEE ENSI SB Quiz", frame)

    # if user closed the window via the X button -> break cleanly
    if cv.getWindowProperty("IEEE ENSI SB Quiz", cv.WND_PROP_VISIBLE) < 1:
        break

    key = cv.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# cleanup
cap.release()
cv.destroyAllWindows()
