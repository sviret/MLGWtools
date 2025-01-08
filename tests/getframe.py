from MLGWtools.generators import generator as gd

#
# Here you put the config files to produce the training 
# and test samples
#

frame_cfg = "MLGWtools/tests/samples/Frame.csv"

test=gd.Generator(paramFile=frame_cfg)
test.buildFrame()
test.saveFrame()  


