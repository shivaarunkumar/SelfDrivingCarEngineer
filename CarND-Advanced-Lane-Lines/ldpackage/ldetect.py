import cv2
import os
from tqdm import tqdm
import ldpackage.process as fp

class LaneDetector:
    """Lane Detection from MP4 Video Captures"""
    def __init__(self,inputVideo,outputVideo):        
        # Cache Values
        self.input = inputVideo
        self.output = outputVideo
        #Temporaritly Open File to retrieve metadata
        self.__inputCapture= cv2.VideoCapture(inputVideo)
        if (self.__inputCapture.isOpened()== False): 
            assert("Error opening input video stream or file")        
        self.fps = self.__inputCapture.get(cv2.CAP_PROP_FPS)
        self.frame_height = int(self.__inputCapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_width = int(self.__inputCapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_size = (self.frame_width, self.frame_height)
        self.frame_count = self.__inputCapture.get(cv2.CAP_PROP_FRAME_COUNT)
        self.__inputCapture.release()

    def process(self,startframe=0,nframes=-1):
        # Open Output Video File Stream
        self.__outputHandle = cv2.VideoWriter(self.output, 
                                            0x7634706d , 
                                            self.fps, 
                                            self.frame_size)
        
        if (self.__outputHandle.isOpened()== False): 
            assert("Error opening output video stream or file")

        # Open Input Video File Stream
        self.__inputCapture= cv2.VideoCapture(self.input)
        if (self.__inputCapture.isOpened()== False): 
            assert("Error opening input video stream or file")
        
        frame_limit = self.frame_count if nframes==-1 else nframes
        count = 0
        # Initialize Progress bar
        pbar = tqdm(total=frame_limit)
        lines = fp.Lines()
        while(self.__inputCapture.isOpened()):
            if count<frame_limit:
                pbar.update(1)
                            
                ret, frame = self.__inputCapture.read()
                if ret == True:
                    # result = self.FrameProcessor.process(frame)
                    if count>=startframe: 
                        result,lines = fp.process_frame(frame,lines)
                        self.__outputHandle.write(result)
                    count+=1                        
                else:
                    break
            
            else:
                break
    def saveframes(self,outdir):
        try: 
            os.mkdir(outdir) 
        except OSError as error: 
            print(error)  
            return
        # Open Input Video File Stream
        self.__inputCapture= cv2.VideoCapture(self.input)
        if (self.__inputCapture.isOpened()== False): 
            assert("Error opening input video stream or file")
        success,image = self.__inputCapture.read()
        count = 0
        pbar = tqdm(total=self.frame_count)
        while success:
            cv2.imwrite(os.path.join(outdir,"frame%d.jpg" % count), image)     # save frame as JPEG file      
            success,image = self.__inputCapture.read()
            count += 1
            pbar.update(1)            
    def __del__(self):
        try:
            self.__inputCapture.release()
        except:
            pass
        try:
            self.__outputHandle.release()
        except:
            pass
        