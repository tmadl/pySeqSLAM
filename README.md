This is a Python port of Niko Sünderhauf's [OpenSeqSLAM code](http://www.tu-chemnitz.de/etit/proaut/mitarbeiter/niko.html).

SeqSLAM performs place recognition by matching sequences of images. 

Quick start: 
 - Download the Nordland dataset:
 
     ```cd datasets/norland; ./getDataset.bash; ```
     
 - Run demo: 

     ```
     cd pyseqslam
     python demo.py
     ```
     
     (This will match the spring sequence of the nordland dataset against the winter sequence. To change which datasets are used, set the environment variables DATASET_1_PATH and DATASET_2_PATH)

[1] Michael Milford and Gordon F. Wyeth (2012). SeqSLAM: Visual Route-Based Navigation for Sunny Summer Days and Stormy Winter Nights. In Proc. of IEEE Intl. Conf. on Robotics and Automation (ICRA)
