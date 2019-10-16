import cv2
import os
from tqdm import tqdm
import argparse

def gen_video(path, delete_images=True):
    '''
    This function iterates over all folders with screenshots,
    sticks together all screenshots and produce a video.avi file to impress Shaul.

    TIP 1: call this function after the episodes loop.
    TIP 2: give a path with enough free space. The video will take Gigas.

    source 1: https://stackoverflow.com/questions/44947505/how-to-make-a-movie-out-of-images-in-python
    source 2: https://stackoverflow.com/questions/16615662/how-to-write-text-on-a-image-in-windows-using-python-opencv2
    source 3: https://docs.opencv.org/3.4/dd/d9e/classcv_1_1VideoWriter.html

    :param path: Where from to take screenshots and where to save the video.
    :param delete_images: Either to delete screenshots after video production.
    '''
    video_name = os.path.join(path,'video.avi')

    image_paths = []

    # Iterates over 'Episode_0', 'Episode_10', ...
    # sub-folders and gathers all image paths.
    for episode_folder_name in os.listdir(path):
        episode_folder_path = os.path.join(path, episode_folder_name)
        if os.path.isdir(episode_folder_path):
            for img in os.listdir(episode_folder_path):
                if (img.endswith(".PNG") or img.endswith(".png")):
                    image_paths.append(os.path.join(episode_folder_path, img))

    if len(image_paths) == 0:
        print("There is no images in the folders")
        return

    frame = cv2.imread(os.path.join(path, image_paths[0]))
    height, width, layers = frame.shape

    fps = 24 # frames per second.

    # create empty video object.
    video = cv2.VideoWriter(video_name, 0, fps, (width, height))

    # parameters of the text to add.
    font = cv2.FONT_HERSHEY_SIMPLEX
    h = 50
    position1 = (20, h)
    fontScale = 1
    fontColor = (0, 0, 0)
    lineType = 2

    # Iterate over all image paths, read the images one by one,
    # add text (Episode, Step, Waiting time) to each image.
    # Append all frames to the video.
    for image_path in tqdm(image_paths):
        image_name = os.path.basename(image_path)
        img = cv2.imread(image_path)

        # Adding text with data.
        epsiode = image_name.split('episode')[1].split('_')[1]
        step = image_name.split('step')[1].split('_')[1]
        num_of_j = len(image_name.split('gne'))-1
        for idx in range(num_of_j):
            position = (20, h + (idx+1)*50)
            junction = image_name.split('gne')[idx+1].split('_')[0]
            wt = image_name.split('gne')[idx + 1].split('_')[1]
            text = junction+' total w.t. [sec]: '+wt
            # write text on image.
            img = cv2.putText(img, text, position, font, fontScale, fontColor, lineType)

        text1 = 'Episode:'+str(epsiode)+'  Step:'+str(step)

        # write text on image.
        img = cv2.putText(img, text1, position1, font, fontScale, fontColor, lineType)
        if num_of_j == 2:
            img = cv2.putText(img, 'J0', (490, 300), font, fontScale, fontColor, lineType)
            img = cv2.putText(img, 'J6', (818, 300), font, fontScale, fontColor, lineType)

        # add frame to video.
        video.write(img)

    cv2.destroyAllWindows()
    video.release()


def main():
    parser = argparse.ArgumentParser(description="Creates video from collection of images")
    parser.add_argument("-dp", "--dir-path", type=str, dest='path', required=True,
                        help='path to directory with images')
    args = parser.parse_args()
    gen_video(args.path)


if __name__ == '__main__':
    main()
