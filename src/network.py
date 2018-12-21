import tensorflow as tf
from networks.net_adapter import Network

class PredictNet():
    def __init__(self, feature, training, share_weight = True, fuse_method = 'concat'):
        self.feature = feature
        self.training = training
        self.branch_count = 0
        self.share_weight = share_weight
        self.fuse_method = fuse_method
        
    def extract_feature(self, inputs):
        scope = 'branch' if self.share_weight else 'branch_%d'%self.branch_count
        with tf.variable_scope(scope, reuse = tf.AUTO_REUSE):
            branch_net = Network(self.training).feed(inputs)
            self.process_branch(branch_net)
        self.branch_count += 1
        return branch_net.output()
        
    def fusion(self, features):
        if self.fuse_method == 'concat':
            return tf.concat(features, axis = -1)
        elif self.fuse_method == 'subtract':
            return tf.subtract(features[0], features[1])
        return None

    def inference(self):
        net = Network(self.training)

        feature_front = self.extract_feature(self.feature['front_view'])
        feature_rear = self.extract_feature(self.feature['rear_view'])

        # implementment by sub class
        net.feed(self.fusion([feature_front, feature_rear]))
        self.process_fusion(net)
        return {'sh':net.output()}

    def process_branch(self):
        raise NotImplementedError('Virtual function, must be implemented by the subclass.')

    def process_fusion(self):
        raise NotImplementedError('Virtual function, must be implemented by the subclass.')

class BasedNet(PredictNet):
    def process_fusion(self, net):
        net.conv(1024).conv(512).conv(256).conv(128).conv(64)
        net.denses([2048,1024,512,256,128,48])

class FixedVGG(BasedNet):
    def process_branch(self, net):
        net.vgg_places365(trainable = False)

class FinetuneVGG(BasedNet):
    def process_branch(self, net):
        net.vgg_places365(trainable = True)

class FromScratchVGG(BasedNet):
    def process_branch(self, net):
        net.vgg_places365(trainable = True, from_scratch = True)

class FixedResnet(BasedNet):
    def process_branch(self, net):
        net.res_places365(trainable = False)

class FinetuneResnet(BasedNet):
    def process_branch(self, net):
        net.res_places365(trainable = True)

class FromScratchResnet(BasedNet):
    def process_branch(self, net):
        net.res_places365(trainable = True, from_scratch = True)

class FixedAlexnet(BasedNet):
    def process_branch(self, net):
        net.alex_places365(trainable = False)

class FinetuneAlexnet(BasedNet):
    def process_branch(self, net):
        net.alex_places365(trainable = True)

class FromScratchAlexnet(BasedNet):
    def process_branch(self, net):
        net.alex_places365(trainable = True, from_scratch = True)

class FixedGooglenet(BasedNet):
    def process_branch(self, net):
        net.googlenet_places365(trainable = False)

class FinetuneGooglenet(BasedNet):
    def process_branch(self, net):
        net.googlenet_places365(trainable = True)

class FromScratchGooglenet(BasedNet):
    def process_branch(self, net):
        net.googlenet_places365(trainable = True, from_scratch = True)

# test code
if __name__ == '__main__':
    net = FinetuneResnet({
        'front_view':tf.ones([1,224,224,3],dtype=tf.float32),
        'rear_view':tf.ones([1,224,224,3],dtype=tf.float32)}, 
    training = True)

    sh = net.inference()

    sess = tf.Session()

    sess.run(tf.global_variables_initializer())

    print(sess.run(sh))

