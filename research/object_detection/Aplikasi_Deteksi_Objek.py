import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import imutils
import tkinter as Tk
from PIL import Image, ImageTk
import tkinter.filedialog
from tkinter.filedialog import askopenfilename

        
###########################################################################

class FrameUjiGambar(Tk.Toplevel):
    """"""
    #----------------------------------------------------------------------
    def __init__(self, original):
        """Constructor"""
        self.original_frame = original
        Tk.Toplevel.__init__(self)
        self.geometry("800x500")
        self.resizable(width=False, height=False)
        self.title("UJI GAMBAR") 
        self.configure(background="#d46a6a")
        self.grid_rowconfigure(0, weight=1)

        button1 = Tk.Button(self, text="Pilih Gambar",width=14,background="#aa3939",font=("consolas",12,'bold'), command=self.pilih_gambar)
        button1.grid(row=0, column=0, sticky='nw', padx = 20, pady=20)        
        button1 = Tk.Button(self, text="Kembali",width=14,background="#aa3939",font=("consolas",12,'bold'), command=self.onClose)
        button1.grid(row=0, column=0, sticky='sw', padx = 20)
        self.Label1 = Tk.Label(self, background="#d46a6a",font=("consolas",12))
        self.Label1.grid(row=2, column=1, sticky='new', padx = 20)
        
    #----------------------------------------------------------------------
    def pilih_gambar(self):
        global panelA
        filename = askopenfilename()
        panelA = None
	
        if len(filename) > 0:
            MODEL_NAME = 'inference_graph'
            IMAGE_NAME = (filename)

            self.Label1.config(text = "Sedang memuat gambar...")
            self.Label1.update_idletasks()            

            CWD_PATH = os.getcwd()
            PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')
            PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap.pbtxt')
            PATH_TO_IMAGE = os.path.join(CWD_PATH,IMAGE_NAME)

            NUM_CLASSES = 5
            label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
            categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
            category_index = label_map_util.create_category_index(categories)

            detection_graph = tf.Graph()
            with detection_graph.as_default():
                od_graph_def = tf.GraphDef()
                with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                    serialized_graph = fid.read()
                    od_graph_def.ParseFromString(serialized_graph)
                    tf.import_graph_def(od_graph_def, name='')

                sess = tf.Session(graph=detection_graph)


            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            # Load image using OpenCV and
            # expand image dimensions to have shape: [1, None, None, 3]
            # i.e. a single-column array, where each item in the column has the pixel RGB value
            image = cv2.imread(IMAGE_NAME)
            image_expanded = np.expand_dims(image, axis=0)
            orig = image.copy()

            # Perform the actual detection by running the model with the image as input
            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: image_expanded})

            # Draw the results of the detection (aka 'visulaize the results')

            vis_util.visualize_boxes_and_labels_on_image_array(
                image,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=2,
                min_score_thresh=0.80)
            # OpenCV represents images in BGR order; however PIL represents
            # images in RGB order, so we need to swap the channels
            im = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Convert the Image object into a TkPhoto object
            im = Image.fromarray(im)
            im = ImageTk.PhotoImage(im)
            self.Label1.config(text="Proses selesai!")
            # if the panels are None, initialize them
            if panelA == None :
                # the first panel will store our original image
                panelA = Tk.Label(self, image=im, width=550, background="#d46a6a")
                panelA.image = im
                panelA.grid(row=0, column=1, sticky='ne', padx = 20, pady=20)
                        
            # otherwise, update the image panels
            else:
                # update the pannels
                panelA.configure(image=im)
                panelA.image = im


    #----------------------------------------------------------------------
    def onClose(self):
        """"""
        self.destroy()
        self.original_frame.show()
    #----------------------------------------------------------------------
    def input_model(self):
        """"""
        d = self.fmodel.get()
        modelname = askopenfilename(initialdir='.\output',filetypes=[('model file', '*model')])
        if len(modelname) > 0:
            if len(d) > 0:
                self.fmodel.delete(0, 'end')
        self.fmodel.insert(0,modelname)
    
###########################################################################


class FrameTentang(Tk.Toplevel):
    """"""
    #----------------------------------------------------------------------
    def __init__(self, original):
        """Constructor"""
        self.original_frame = original
        Tk.Toplevel.__init__(self)
        self.geometry("800x500")
        self.resizable(width=False, height=False)
        self.title("TENTANG")
        self.configure(background="#d46a6a")

        img3 = Image.open('./assets/home-03.jpg')
        photoimg3 = ImageTk.PhotoImage(img3)
        title3 = Tk.Label(self,image=photoimg3, background="#d46a6a")
        title3.image = photoimg3
        title3.grid(row=0, column=0, columnspan=10, sticky = 'sew', pady=10)
        
        button1 = Tk.Button(self, text="Kembali",width=14,background="#aa3939",font=("consolas",12,'bold'), command=self.onClose)
        button1.grid(row=1, column =4, columnspan=2)

        img4 = Image.open('./assets/home-04.jpg')
        photoimg4 = ImageTk.PhotoImage(img4)
        title4 = Tk.Label(self,image=photoimg4, background="#d46a6a")
        title4.image = photoimg4
        title4.grid(row=20, column=0, columnspan=10, sticky = 'sew')

        
    #----------------------------------------------------------------------
    def onClose(self):
        """"""
        self.destroy()
        self.original_frame.show()

    
###########################################################################
class MyApp(object):
    """"""
    #----------------------------------------------------------------------
    def __init__(self, parent):
        """Constructor"""
        self.root = parent
        self.root.title("Main frame")
        self.frame = Tk.Frame(parent)
        self.frame.grid()
        

        self.root.title("Object Detector")
        self.root.configure(background="#d46a6a")
        self.frame = Tk.Frame(parent)
        self.frame.grid()
        
        img = Image.open('./assets/home-01.jpg')
        photoimg = ImageTk.PhotoImage(img)
        title = Tk.Label(image=photoimg, background="#d46a6a")
        title.image = photoimg
        title.grid(row=0, columnspan=2, sticky = 'new')
        
        button1 = Tk.Button(text="Deteksi Objek",width=16, background="#aa3939",font=("consolas",12,'bold'),command=self.test_data)
        button1.grid(row=1, column = 0, sticky = 'ne', padx = 10, pady=10)
        button1 = Tk.Button(text="Tentang",width=16, background="#aa3939",font=("consolas",12,'bold'),command=self.about)
        button1.grid(row=1, column = 1, sticky = 'ws', padx = 10, pady=10)
        
        img2 = Image.open('./assets/home-02.jpg')
        photoimg2 = ImageTk.PhotoImage(img2)
        title2 = Tk.Label(image=photoimg2, background="#d46a6a")
        title2.image = photoimg2
        title2.grid(row=3, columnspan=2, sticky = 'sew', pady=13)
        
    #----------------------------------------------------------------------
    def hide(self):
        """"""
        self.root.withdraw()
    #----------------------------------------------------------------------
    def show(self):
        """"""
        self.root.update()
        self.root.deiconify()        
    #----------------------------------------------------------------------
    def about(self):
        """"""
        self.hide()
        subFrame = FrameTentang(self)
    #----------------------------------------------------------------------
    def test_data(self):
        """"""
        self.hide()
        subFrame = FrameUjiGambar(self)
            
#--------------------------------------------------------------------------
if __name__ == "__main__":
    sys.path.append("..")
    from utils import label_map_util
    from utils import visualization_utils as vis_util
    root = Tk.Tk()
    root.geometry("800x500")
    root.resizable(width=False, height=False)
    app = MyApp(root)
    root.mainloop()
