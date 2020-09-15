from torch.nn import DataParallel
from torch.cuda import is_available
from torch import load, device, set_grad_enabled
from model import Generator,Discriminator
DEVICE = device("cuda" if is_available() else "cpu")


class SuperGenerator():
    def __init__(self):

        self.g_plane = self.initialize_model('Plane').eval()
        self.g_chair = self.initialize_model('Chair').eval()
        self.d_chair = self.initialize_model_d.eval()

    def initialize_model(self, model_type):
        '''
        Loads and initializes the model based on model size
        '''
        g = Generator(200, 64)
        g_model = DataParallel(g)
        g_checkpoint = load('plane_generator.tar' if model_type ==
                            'Plane' else 'chair_generator.tar', map_location=DEVICE)
        g_model.load_state_dict(g_checkpoint["model_state_dict"])
        return g_model


    def generate(self, z, model_type):
        '''
        Generates a cube_len*cube_len*cube_len model in voxels
        '''
        with set_grad_enabled(False):
            return self.g_plane(z).view(64, 64, 64) if model_type == 'Plane' else self.g_chair(z).view(64, 64, 64)
        
class SuperDiscriminator():
    def __init__(self):
        self.d_chair = self.initialize_model.eval()
    
    def initialize_model(self):
        d = Discriminator(False, 64, 0.3, 'wgan-gp')
        d_model = DataParallel(d)
        d_checkpoint = load('chair_discriminator.tar', map_location=DEVICE)
        d_model.load_state_dict(d_checkpoint["model_state_dict"])
        return d_model
    
    def discriminate(self, fake):
        with set_grad_enabled(False):
            return self.d_chair(fake)