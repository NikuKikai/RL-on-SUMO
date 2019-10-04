import cv2
import os

def gen_video_all(path):
    '''
    Iterates over all sub folders in path, and converts all captures to video.
    :param path:
    :return:
    '''
    raise NotImplementedError

def gen_video(path, delete_images=True):
    '''
    This function sticks together all screenshots and produce a video.avi file to impress Shaul.
    source 1: https://stackoverflow.com/questions/44947505/how-to-make-a-movie-out-of-images-in-python
    source 2: https://stackoverflow.com/questions/16615662/how-to-write-text-on-a-image-in-windows-using-python-opencv2
    source 3: https://docs.opencv.org/3.4/dd/d9e/classcv_1_1VideoWriter.html

    :param path: Where from to take screenshots and where to save the video.
    :param delete_images: Either to delete screenshots after video production.
    '''
    image_folder = path
    video_name = os.path.join(path,'video.avi')

    image_names = [img for img in os.listdir(image_folder) if (img.endswith(".PNG") or img.endswith(".png"))]
    if len(image_names) == 0:
        print("There is no images in the folder")
        return

    frame = cv2.imread(os.path.join(image_folder, image_names[0]))
    height, width, layers = frame.shape

    fps = 24
    video = cv2.VideoWriter(video_name, 0, fps, (width, height))

    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (20, int(height/3))
    fontScale = 1
    fontColor = (0, 0, 0)
    lineType = 2

    for image_name in image_names:
        img = cv2.imread(os.path.join(image_folder, image_name))
        epsiode = image_name.split('_')[1]
        step = image_name.split('_')[3].split('.')[0]
        text = 'Episode:'+str(epsiode)+'  Step:'+str(step)
        img = cv2.putText(img, text, bottomLeftCornerOfText, font, fontScale, fontColor, lineType)
        video.write(img)

    cv2.destroyAllWindows()
    video.release()

    if delete_images:
        for file_name in os.listdir(image_folder):
            if file_name.endswith('.PNG') or file_name.endswith('.png'):
                os.remove(os.path.join(image_folder, file_name))

# def main():
#     gen_video('C:\\Users\\Pavel\\Desktop\\RL-on-SUMO\\Capture_test\\2019_10_04_07_12_25_186097_fixed_q_targets_israel_single_intersection\\capture\\Episode_0')
#
# main()