from MLGWtools.generators import generator as gd

#
# Here you put the config files to produce the training 
# and test samples
#

train_cfg = "MLGWtools/tests/samples/Huerta_train.csv"
test_cfg = "MLGWtools/tests/samples/Huerta_test.csv"

test=gd.Generator(paramFile=test_cfg)
test.buildSample()
test.saveGenerator()  

train=gd.Generator(paramFile=train_cfg)
train.buildSample()
train.saveGenerator()

