import cv2
import streamlit as st
import mediapipe as mp
import numpy as np
import tempfile
import time
from PIL import Image, ImageColor
import os
import cvzone
import numpy as np
from skimage.filters import gaussian
from test import evaluate
import streamlit as st
from cvzone.PoseModule import PoseDetector

def sharpen(img):
    img = img * 1.0
    gauss_out = gaussian(img, sigma=5)

    alpha = 1.5
    img_out = (img - gauss_out) * alpha + img

    img_out = img_out / 255.0

    mask_1 = img_out < 0
    mask_2 = img_out > 1

    img_out = img_out * (1 - mask_1)
    img_out = img_out * (1 - mask_2) + mask_2
    img_out = np.clip(img_out, 0, 1)
    img_out = img_out * 255
    return np.array(img_out, dtype=np.uint8)


def hair(image, parsing, part=17, color=[230, 50, 20]):
    b, g, r = color      #[10, 50, 250]       # [10, 250, 10]
    tar_color = np.zeros_like(image)
    tar_color[:, :, 0] = b
    tar_color[:, :, 1] = g
    tar_color[:, :, 2] = r
    np.repeat(parsing[:, :, np.newaxis], 3, axis=2)

    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    tar_hsv = cv2.cvtColor(tar_color, cv2.COLOR_BGR2HSV)

    if part == 12 or part == 13:
        image_hsv[:, :, 0:2] = tar_hsv[:, :, 0:2]
    else:
        image_hsv[:, :, 0:1] = tar_hsv[:, :, 0:1]

    changed = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)

    if part == 17:
        changed = sharpen(changed)
    

    changed[parsing != part] = image[parsing != part]
    return changed

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

detector = PoseDetector()
shirtFolderPath = "./Shirts"
listShirts = os.listdir(shirtFolderPath)
# print(listShirts)
fixedRatio = 262 / 190  
shirtRatioHeightWidth = 581 / 440
imageNumber = 0
#imgButtonRight = cv2.imread("Resources/button.png", cv2.IMREAD_UNCHANGED)
#imgButtonLeft = cv2.flip(imgButtonRight, 1)
counterRight = 0
counterLeft = 0
selectionSpeed = 10
DEMO_IMAGE1 = 'imgs/6.jpg'


DEMO_IMAGE = 'demo.jpg'
DEMO_VIDEO = 'demo1.mp4'

st.title('Virtual Threads')


st.markdown(
    """
    <style>
    [data-testid="stSiderbar"][aria-expanded="true"] > div:first-child{
        width : 350 px
    }
    [data-testid="stSiderbar"][aria-expanded="false"] > div:first-child{
        width : 350 px
        margin-left: -350px
    }
    </style>
    """,

    unsafe_allow_html=True,
)

st.sidebar.title('Try On Sidebar')
st.sidebar.subheader('parameters')


@st.cache()
def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):


    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        r = width/float(w)
        dim = (int(w*r), height)

    else:
        r = width/float(w)
        dim = (width, int(w*r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    return resized


app_mode = st.sidebar.selectbox(
    'Choose the app mode', ['About App', 'Run with Makeup', 'Run on Clothes'])

################################################# FIRST PAGE ###########################################################

if app_mode == 'About App':

    st.markdown(
        'In this application we are giving you a live try on for you to experience Clothes virtually.')

    st.markdown(
        """
    <style>
    [data-testid="stSiderbar"][aria-expanded="true"] > div:first-child{
        width : 350 px
    }
    [data-testid="stSiderbar"][aria-expanded="false"] > div:first-child{
        width : 350 px
        margin-left: -350px
    }
    </style>
    """,

        unsafe_allow_html=True,
    )
    st.video('https://www.youtube.com/watch?v=nWcGhuX6N7w')

elif app_mode == 'Run with Makeup':
    

    st.set_option('deprecation.showfileUploaderEncoding', False)
    

    st.sidebar.title('Virtual Makeup')

    table = {
            'hair': 17,
            'upper_lip': 12,
            'lower_lip': 13,
        }

    # use_webcame = st.sidebar.button('Use WebCam')
    # close_webcame = st.sidebar.button('Close Webcam')
    DEMO_IMAGE1 = 'imgs/6.jpg'


    st.markdown(
        """
    <style>
    [data-testid="stSiderbar"][aria-expanded="true"] > div:first-child{
        width : 350 px
    }
    [data-testid="stSiderbar"][aria-expanded="false"] > div:first-child{
        width : 350 px
        margin-left: -350px
    }
    </style>
    """,

        unsafe_allow_html=True,
    )

    # st.sidebar.markdown('----')
    # shirts = st.sidebar.selectbox(
    # 'Choose your shirt', ['Blue Shirt', 'Green Shirt', 'Red Shirt'])

    # st.sidebar.markdown('----')

    st.markdown("## Try It ON")


    # capture_image = st.sidebar.checkbox('Capture Image', False)

    # if capture_image:
    #     cap = cv2.VideoCapture(0)
    #     _, frame = cap.read()
    #     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #     image = frame

    
    img_file_buffer = st.sidebar.file_uploader("Upload an image", type=[ "jpg", "jpeg",'png'])

    if img_file_buffer is not None:
        image = np.array(Image.open(img_file_buffer))
        demo_image = img_file_buffer

    else:
        demo_image = DEMO_IMAGE1
        image = np.array(Image.open(demo_image))
        
        #st.set_option('deprecation.showfileUploaderEncoding', False)

    new_image = image.copy()

    
    col1, col2 = st.columns(2)

    with col1:
        st.subheader('Original Image')
        st.image(image,use_column_width = True)


    cp = 'cp/79999_iter.pth'
    ori = image.copy()
    h,w,_ = ori.shape

    #print(h)
    #print(w)
    image = cv2.resize(image,(1024,1024))


    parsing = evaluate(demo_image, cp)
    parsing = cv2.resize(parsing, image.shape[0:2], interpolation=cv2.INTER_NEAREST)

    parts = [table['hair'], table['upper_lip'], table['lower_lip']]

    hair_color = st.sidebar.color_picker('Pick the Hair Color', '#000')
    hair_color = ImageColor.getcolor(hair_color, "RGB")

    lip_color = st.sidebar.color_picker('Pick the Lip Color', '#edbad1')

    lip_color = ImageColor.getcolor(lip_color, "RGB")



    colors = [hair_color, lip_color, lip_color]

    for part, color in zip(parts, colors):
        image = hair(image, parsing, part, color)

    image = cv2.resize(image,(w,h))


    
    with col2:
        st.subheader('Output Image')
        st.image(image, use_column_width=True)


elif app_mode == 'Run on Clothes':



    st.set_option('deprecation.showfileUploaderEncoding', False)

    use_webcame = st.sidebar.button('Use WebCame')
    close_webcame = st.sidebar.button('Close Webcame')

    st.markdown(
        """
    <style>
    [data-testid="stSiderbar"][aria-expanded="true"] > div:first-child{
        width : 350 px
    }
    [data-testid="stSiderbar"][aria-expanded="false"] > div:first-child{
        width : 350 px
        margin-left: -350px
    }
    </style>
    """,

        unsafe_allow_html=True,
    )

    st.sidebar.markdown('----')
    shirts = st.sidebar.selectbox(
    'Choose your shirt', ['Blue Shirt', 'Green Shirt',"Red Shirt", 'Suit','Top'])

    st.sidebar.markdown('----')

    st.markdown("## Try It ON")

    ## WE GET OUR VIDEO INPUT ##
    stframe = st.empty()
    video_file_buffer = st.sidebar.file_uploader(
        "Upload a Video", type=["mp4", "mov", "avi", "asf", "m4v"])
    tffile = tempfile.NamedTemporaryFile(delete=False)

    if not video_file_buffer:
        if use_webcame:
            vid = cv2.VideoCapture(0)

        else:
            vid = cv2.VideoCapture(DEMO_VIDEO)
            tffile.name = DEMO_VIDEO

    else:
        tffile.write(video_file_buffer.read())
        vid = cv2.VideoCapture(tffile.name)

    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_input = int(vid.get(cv2.CAP_PROP_FPS))

    ## RECORDING PART ##
    codec = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    out = cv2.VideoWriter('output1.m4v', codec, fps_input, (width, height))

    #drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=1)

    # FACE MESH
    while vid.isOpened():
        success, img = vid.read()
        img = detector.findPose(img,draw=False)
        # img = cv2.flip(img,1)
        lmList, bboxInfo = detector.findPosition(img, bboxWithHands=False, draw=False)

        if (shirts == "Blue Shirt"):
            imageNumber = 0
        if (shirts == "Green Shirt"):
             imageNumber = 1
        if (shirts == "Red Shirt"):
             imageNumber = 2
        if (shirts == "Suit"):
            imageNumber = 5
        if (shirts == "Top"):
            imageNumber = 4

        if lmList:
        # center = bboxInfo["center"]
            lm11 = lmList[11][1:3]
            lm12 = lmList[12][1:3]
            imgShirt = cv2.imread(os.path.join(shirtFolderPath, listShirts[imageNumber]), cv2.IMREAD_UNCHANGED)

        widthOfShirt = int((lm11[0] - lm12[0]) * fixedRatio)
        print(widthOfShirt)
        imgShirt = cv2.resize(imgShirt, (widthOfShirt, int(widthOfShirt * shirtRatioHeightWidth)))
        currentScale = (lm11[0] - lm12[0]) / 190
        offset = int(44 * currentScale), int(48 * currentScale)

        try:
            img = cvzone.overlayPNG(img, imgShirt, (lm12[0] - offset[0], lm12[1] - offset[1]))
        except:
            pass

        if close_webcame:
            vid.release()
            # frame = cv2.resize(img, (0, 0), fx=0.8, fy=0.8)
            # stframe.image(DEMO_VIDEO, channels='BGR', use_column_width=False)
            #cv2.destroyAllWindows()
            break

        frame = cv2.resize(img, (0, 0), fx=0.8, fy=0.8)
        stframe.image(frame, channels='BGR', use_column_width=False)
    #     st.subheader('Output Image')
    #     st.image(out_image, use_column_width=True)
