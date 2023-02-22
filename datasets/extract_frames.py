import cv2
import time
import os
import glob

def video_to_frames(input_loc, output_loc):
    """Function to extract frames from input video file
    and save them as separate frames in an output directory.
    Args:
        input_loc: Input video file.
        output_loc: Output directory to save the frames.
    Returns:
        None
    """
    try:
        os.mkdir(output_loc)
    except OSError:
        pass
    # Log the time
    time_start = time.time()
    # Start capturing the feed
    cap = cv2.VideoCapture(input_loc + ".mp4")
    # Find the number of frames
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    print ("Number of frames: ", video_length)
    count = 0
    print ("Converting video..\n")
    # Start converting the video
    while cap.isOpened():
        # Extract the frame
        ret, frame = cap.read()
        if not ret:
            continue
        # Write the results back to output location.
        if count % 5 ==0:
            cv2.imwrite(output_loc + "/S11_" + input_loc.split('/')[-1] + "_%#06d.jpg" % (count+1), frame)
        count = count + 1
        # If there are no more frames left
        if (count > (video_length-1)):
            # Log the time again
            time_end = time.time()
            # Release the feed
            cap.release()
            # Print stats
            print ("Done extracting frames.\n%d frames extracted" % count)
            print ("It took %d seconds for conversion." % (time_end-time_start))
            break

if __name__=="__main__":

    # input_loc = 'Directions_1.54138969'
    output_loc = "./images/"
    
    old_file_list = glob.glob("extracted/S11/Videos/*.mp4")
    new_file_list = [i.replace(" ", "_") for i in old_file_list]
    for i in range(len(old_file_list)):
        os.rename(old_file_list[i], new_file_list[i])  
    file_list = [i[:-4] for i in new_file_list]
    for i in range(len(file_list)):
        video_to_frames(file_list[i], output_loc)