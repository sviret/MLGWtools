from MLGWtools.generators import generator as gd

#
# Here you put the config files to produce the simulated
# data frame
#

frame_cfg = "MLGWtools/generators/samples/Frame.csv"

test=gd.Generator(paramFile=frame_cfg)
test.buildFrame()
test.saveFrame()  


