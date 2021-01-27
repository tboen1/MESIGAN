from config import get_arguments
from SinGAN.manipulate import *
from SinGAN.training import *
from SinGAN.imresize import imresize
import SinGAN.functions as functions
import torch.nn.functional as F
from torch.autograd import Variable
import copy

def numpy_2_torch(x, channels = 3, not_cuda = False):
    if channels == 3:
        x = x[:,:,:,None]
        x = x.transpose((3, 2, 0, 1))/255
    else:
        x = color.rgb2gray(x)
        x = x[:,:,None,None]
        x = x.transpose(3, 2, 0, 1)
    x = torch.from_numpy(x)
    if not(not_cuda):
        x = move_to_gpu(x)
    x = x.type(torch.cuda.FloatTensor) if not(not_cuda) else x.type(torch.FloatTensor)
    x = norm(x)
    return x

def move_to_gpu(t):
    if (torch.cuda.is_available()):
        t = t.to(torch.device('cuda'))
    return t

def norm(x):
    out = (x -0.5) *2
    return out.clamp(-1, 1)

def show_image(tensor, show, label = 'blank'):
  array = functions.torch2uint8(tensor)
  if show:
    print(array.shape)
    if label != 'blank':
      print(label)  
    plt.imshow(array)
    plt.show()
  return array

def sigmoid_scaling(input_tensor, alpha=1, conjugate=False):
    tensor = copy.deepcopy(input_tensor)
    size = tensor.size()
    N = size[3]-1
    for n in range(N+1):
        x = torch.tensor((2*n/N)-1)*alpha
        factor = torch.sigmoid(x)
        if conjugate:
            tensor[:,:,:,n] = (1-factor)*input_tensor[:,:,:,n]
        else:
            tensor[:,:,:,n] = factor*input_tensor[:,:,:,n]
    return tensor

def sigmoid(x, alpha):
  return 1/(1+np.exp(-alpha*x))

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def prob_map(tensor, conjugate = False):
    b, c, h, w = tensor.size()
    field = np.ones([h,w])
    row, col = field.shape
    for i in range(row):
      for j in range(col):
        #score = sigmoid(j-col/2, 0.1)*gaussian(i,row/2,row/2)
        score = sigmoid(j-col/2,0.1)
        field[i][j] = score
    #field = field*0.2   
    if conjugate:
        field = 1-field

    '''converting to torch tensors'''
    field = numpy_2_torch(field, channels=1)
    print(field.size())
    field = field.repeat(1,3,1,1)

    return field

def reinforcement_sigmoid(anchor_tensor, image_tensor, direction, n):
    #[batch, channels, height, width]
    size = anchor_tensor.size()

    if direction == 'L':
        lim = int(size[3]/2)
        anchor = anchor_tensor[:,:,:,lim:]
        anchor = anchor[:,:,:,:lim]
        variable = image_tensor[:,:,:,:lim]
        #print('image: {}'.format(image_tensor.size()))
        #print('variable: {}'.format(variable.size()))
        #print('anchor: {}'.format(anchor.size()))


        show_image(anchor, True, label = 'anchor')
        show_image(variable, True, label = 'variable')

        alphas = [1, 2, 3, 4, 5, 6, 7, 8, 9, 1]

        if n in [9]:
        #variable_image = sigmoid_scaling(variable, alpha = alphas[n])
          variable_image = prob_map(variable)*variable
        #anchoring_image = sigmoid_scaling(anchor, alpha = alphas[n], conjugate = True)
          anchoring_image = prob_map(anchor, conjugate = True)*anchor

          image_tensor[:,:,:,:lim] =  variable_image + anchoring_image

    return image_tensor

def reinforcementX(anchor_tensor, image_tensor, direction):
    #[batch, channels, height, width]
    size = anchor_tensor.size()

    if direction == 'T':
        lim = int(size[2]/2)
        anchor = anchor_tensor[:,:,lim:,:]
        anchor = anchor[:,:,:lim,:]
        image_tensor[:,:,:int(size[2]/2),:] = anchor
    if direction == 'B':
        lim = int(size[2]/2)
        anchor = anchor_tensor[:,:,:lim,:]
        anchor = anchor[:,:,lim:,:]
        image_tensor[:,:,lim:,:] = anchor
    if direction == 'L':
        lim = int(size[3]/2)
        anchor = anchor_tensor[:,:,:,lim:]
        anchor = anchor[:,:,:,:lim]
        image_tensor[:,:,:,:lim] = anchor
    if direction == 'R':
        lim = int(size[3]/2)
        anchor = anchor_tensor[:,:,:,:lim]
        anchor = anchor[:,:,:,lim:]
        image_tensor[:,:,:,lim:] = anchor
    return image_tensor

def reinforcement(anchor_tensor, image_tensor, direction):
    #[batch, channels, height, width]
    #print(direction.size())
    #show_image(direction, show=True)
    direction = direction + 1
    direction = direction/2
    #show_image(direction, show=True)
    #image_tensor = (1-direction)*anchor_tensor + direction*image_tensor
    image_tensor = direction*anchor_tensor + (1-direction)*image_tensor
    #print(1-direction)
    #show_image(image_tensor, show=True)
    return image_tensor

def SinGAN_anchor_generate(Gs,Zs,reals,NoiseAmp,opt,in_s=None,scale_v=1,scale_h=1,n=0,gen_start_scale=0,num_samples=1, 
anchor_image=None, direction=None, transfer = None, noise_solutions=None, factor=None, base=None, insert_limit=0):
    
    #### Loading in Anchor if Needed #####
    anchor = anchor_image
    if anchor is not None:
      anchors = []
      anchor = functions.np2torch(anchor_image, opt)
      anchor_ = imresize(anchor,opt.scale1,opt)
      anchors = functions.creat_reals_pyramid(anchor_,anchors,opt) #high key hacky code
    if direction is not None:
      directions = []
      direction = functions.np2torch(direction, opt)
      direction_ = imresize(direction,opt.scale1,opt)
      directions = functions.creat_reals_pyramid(direction_,directions,opt) #high key hacky code
    if base is not None:
      bases = []
      base = functions.np2torch(base, opt)
      base_ = imresize(base,opt.scale1,opt)
      bases = functions.creat_reals_pyramid(base_,bases,opt) #high key hacky code
    #### MY CODE ####
    
    #if torch.is_tensor(in_s) == False:
    if in_s is None:
        in_s = torch.full(reals[0].shape, 0, device=opt.device)
    images_cur = []
    for G,Z_opt,noise_amp in zip(Gs,Zs,NoiseAmp):
        pad1 = ((opt.ker_size-1)*opt.num_layer)/2
        m = nn.ZeroPad2d(int(pad1))
        nzx = (Z_opt.shape[2]-pad1*2)*scale_v
        nzy = (Z_opt.shape[3]-pad1*2)*scale_h

        images_prev = images_cur
        images_cur = []

        for i in range(0,num_samples,1):
            if n == 0: #COARSEST SCALE
                z_curr = functions.generate_noise([1,nzx,nzy], device=opt.device)
                z_curr = z_curr.expand(1,3,z_curr.shape[2],z_curr.shape[3])
                z_curr = m(z_curr)
            else:
                z_curr = functions.generate_noise([opt.nc_z,nzx,nzy], device=opt.device)
                z_curr = m(z_curr)

            z_orig = z_curr

            if images_prev == []: #FIRST GENERATION IN COARSEST SCALE
                I_prev = m(in_s)               

            else: #NOT FIRST GENERATION, BUT AT COARSEST SCALE
                I_prev = images_prev[i]
                I_prev = imresize(I_prev,1/opt.scale_factor, opt) #upscale
                #print(n)
                if opt.mode != "SR": 
                    I_prev = I_prev[:, :, 0:round(scale_v * reals[n].shape[2]), 0:round(scale_h * reals[n].shape[3])]
                    I_prev = m(I_prev)
                    I_prev = I_prev[:,:,0:z_curr.shape[2],0:z_curr.shape[3]]
                    I_prev = functions.upsampling(I_prev,z_curr.shape[2],z_curr.shape[3]) #make it fit padded noise
                else: 
                    #prev_before = I_prev #MY ADDITION
                    I_prev = m(I_prev)    

            if n < gen_start_scale: #anything less than final
                z_curr = Z_opt #Z_opt comes from trained pyramid....
            z_in = noise_amp*(z_curr)+I_prev

            if noise_solutions is not None:
                z_curr = noise_solutions[n]

                z_in = (1-factor)*noise_amp*(z_curr)+I_prev + factor*noise_amp*z_orig #adds in previous image to z_opt'''

            I_curr = G(z_in.detach(),I_prev)
            if base is not None:
                if n == insert_limit:
                  I_curr = bases[n]*factor + I_curr*(1-factor)

            if anchor is not None and direction is not None:
                anchor_curr = anchors[n]
                I_curr = reinforcement(anchor_curr, I_curr, directions[n])
                #I_curr = reinforcement_sigmoid(anchor_curr, I_curr, direction, n)
            ###### ENFORCE LH = ANCHOR FOR IMAGE #######

            if n == opt.stop_scale: #hacky code
                if anchor is not None and direction is not None:
                    anchor_curr = anchors[n]
                    I_curr = reinforcement(anchor_curr, I_curr, direction)
                    #I_curr = reinforcement_sigmoid(anchor_curr, I_curr, direction, n)
                array = functions.convert_image_np(I_curr.detach())
            images_cur.append(I_curr)
        n+=1
    return array


def invert_model(test_image, model_name, scales2invert = None, penalty=1e-3, show=True):
    '''test_image is an array, model_name is a name'''
    Noise_Solutions = []

    parser = get_arguments()
    parser.add_argument('--input_dir', help='input image dir', default='Input/Images')
  
    parser.add_argument('--mode', default='RandomSamples')
    opt = parser.parse_args("")
    opt.input_name = model_name
    opt.reg = penalty

    if model_name == 'islands2_basis_2.jpg': #HARDCODED
        opt.scale_factor = 0.6

    opt = functions.post_config(opt)

    ### Loading in Generators 
    Gs,Zs,reals,NoiseAmp = functions.load_trained_pyramid(opt) 
    for G in Gs:
        G = functions.reset_grads(G,False)
        G.eval()

    ### Loading in Ground Truth Test Images
    reals = [] #deleting old real images 
    real = functions.np2torch(test_image, opt)
    functions.adjust_scales2image(real, opt)

    real_ = functions.np2torch(test_image, opt)
    real = imresize(real_,opt.scale1,opt)
    reals = functions.creat_reals_pyramid(real,reals,opt) 
    
    ### General Padding 
    pad_noise = int(((opt.ker_size - 1) * opt.num_layer) / 2) 
    m_noise = nn.ZeroPad2d(int(pad_noise))

    pad_image = int(((opt.ker_size - 1) * opt.num_layer) / 2)
    m_image = nn.ZeroPad2d(int(pad_image)) 

    I_prev = None
    REC_ERROR = 0

    if scales2invert is None:
      scales2invert = opt.stop_scale+1

    for scale in range(scales2invert): 
    #for scale in range(3):

        #Get X, G
        X = reals[scale]
        G = Gs[scale]
        noise_amp = NoiseAmp[scale]

        #Defining Dimensions 
        opt.nc_z = X.shape[1]
        opt.nzx = X.shape[2]
        opt.nzy = X.shape[3]
        
        #getting parameters for prior distribution penalty
        pdf = torch.distributions.Normal(0, 1)
        alpha = opt.reg
        #alpha = 1e-2

        #Defining Z 
        if scale == 0:
            z_init = functions.generate_noise([1,opt.nzx,opt.nzy], device=opt.device) #only 1D noise 
        else:
            z_init = functions.generate_noise([3,opt.nzx, opt.nzy], device=opt.device) #otherwise move up to 3d noise 

        z_init = Variable(z_init.cuda(), requires_grad=True) #variable to optimize

        #Building I_prev
        if I_prev == None: #first scale scenario 
            in_s = torch.full(reals[0].shape, 0, device=opt.device) #all zeros  
            I_prev = in_s
            I_prev = m_image(I_prev) #padding 

        else: #otherwise take the output from the previous scale and upsample 
            I_prev = imresize(I_prev,1/opt.scale_factor, opt) #upsamples
            I_prev = m_image(I_prev)
            I_prev = I_prev[:,:,0:X.shape[2]+10,0:X.shape[3]+10] #making sure that precision errors don't mess anything up 
            I_prev = functions.upsampling(I_prev,X.shape[2]+10,X.shape[3]+10) #seems to be redundant 

        LR = [2e-3, 2e-2, 2e-1, 2e-1, 2e-1, 2e-1, 2e-1, 2e-1, 2e-1, 2e-1, 2e-1]
        Zoptimizer = torch.optim.RMSprop([z_init], lr=LR[scale]) #Defining Optimizer
        x_loss = [] #for plotting 
        epochs = [] #for plotting 

        niter = [200, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400]
        for epoch in range(niter[scale]): #Gradient Descent on Z

            if scale == 0:
                noise_input = m_noise(z_init.expand(1,3,opt.nzx,opt.nzy)) #expand and padd 
            else:
                noise_input = m_noise(z_init) #padding 
                
            z_in = noise_amp*noise_input + I_prev 
            G_z = G(z_in, I_prev)  

            x_recLoss = F.mse_loss(G_z, X) #MSE loss 

            logProb = pdf.log_prob(z_init).mean() #Gaussian loss 

            loss = x_recLoss - (alpha * logProb.mean())
                
            Zoptimizer.zero_grad()
            loss.backward()
            Zoptimizer.step()

            #losses['rec'].append(x_recLoss.data[0])
            #print('Image loss: [%d] loss: %0.5f' % (epoch, x_recLoss.item()))
            #print('Noise loss: [%d] loss: %0.5f' % (epoch, z_recLoss.item()))
            x_loss.append(loss.item())
            epochs.append(epoch)

            REC_ERROR = x_recLoss
                
        if show:
            plt.plot(epochs, x_loss, label = 'x_loss')
            plt.legend()
            plt.show()
        
        
        I_prev = G_z.detach() #take final output, maybe need to edit this line something's very very fishy 

        _ = show_image(X, show, 'target')
        reconstructed_image = show_image(I_prev, show, 'output')
        _ = show_image(noise_input.detach().cpu(), show, 'noise')

        Noise_Solutions.append(noise_input.detach())
    return Noise_Solutions, reconstructed_image, REC_ERROR

def generate(model_name, anchor_image = None, direction = None, transfer = None, noise_solutions = None, 
factor = 0.25, base = None, insert_limit = 4):
    #direction = 'L, R, T, B' 

    parser = get_arguments()
    parser.add_argument('--input_dir', help='input image dir', default='Input/Images')
    parser.add_argument('--mode', help='random_samples | random_samples_arbitrary_sizes', 
                    default='random_samples')
    # for random_samples:
    parser.add_argument('--gen_start_scale', type=int, help='generation start scale', default=0)
    opt = parser.parse_args("")
    opt.input_name = model_name

    if model_name == 'islands2_basis_2.jpg': #HARDCODED
        opt.scale_factor = 0.6

    opt = functions.post_config(opt)
    Gs = []
    Zs = []
    reals = []
    NoiseAmp = []

    real = functions.read_image(opt)
    #opt.input_name = anchor #CHANGE TO ANCHOR HERE
    anchor = functions.read_image(opt)

    functions.adjust_scales2image(real, opt)
    Gs, Zs, reals, NoiseAmp = functions.load_trained_pyramid(opt)
    in_s = functions.generate_in2coarsest(reals,1,1,opt)

    array = SinGAN_anchor_generate(Gs, Zs, reals, NoiseAmp, opt, gen_start_scale=opt.gen_start_scale, 
    anchor_image = anchor_image, direction = direction, transfer = transfer, 
    noise_solutions = noise_solutions, factor = factor, base = base, insert_limit = insert_limit)
    return array

def test_generate(model_name, anchor_image = None, direction = None, transfer = None, noise_solutions = None, 
factor = 0.25, base = None, insert_limit = 4):
    #direction = 'L, R, T, B' 

    parser = get_arguments()
    parser.add_argument('--input_dir', help='input image dir', default='Input/Images')
    parser.add_argument('--mode', help='random_samples | random_samples_arbitrary_sizes', 
                    default='random_samples')
    # for random_samples:
    parser.add_argument('--gen_start_scale', type=int, help='generation start scale', default=0)
    opt = parser.parse_args("")
    opt.input_name = model_name

    opt = functions.post_config(opt)
    Gs = []
    Zs = []
    reals = []
    NoiseAmp = []

    opt.input_name = 'island_basis_0.jpg' #grabbing image that exists...
    real = functions.read_image(opt)
    #opt.input_name = anchor #CHANGE TO ANCHOR HERE
    #anchor = functions.read_image(opt)
    functions.adjust_scales2image(real, opt)

    opt.input_name = 'test1.jpg' #grabbing model that we want 
    Gs, Zs, reals, NoiseAmp = functions.load_trained_pyramid(opt)

    #dummy stuff for dimensions
    reals = []
    real_ = real
    real = imresize(real_,opt.scale1,opt)
    reals = functions.creat_reals_pyramid(real,reals,opt)
    in_s = functions.generate_in2coarsest(reals,1,1,opt)

    array = SinGAN_anchor_generate(Gs, Zs, reals, NoiseAmp, opt, gen_start_scale=opt.gen_start_scale, 
    anchor_image = anchor_image, direction = direction, transfer = transfer, 
    noise_solutions = noise_solutions, factor = factor, base = base, insert_limit = insert_limit)
    return array

def test_pyramid(images):
    parser = get_arguments()
    parser.add_argument('--input_dir', help='input image dir', default='Input/Images')
    #parser.add_argument('--input_name', help='input image name', required=True)
    parser.add_argument('--mode', help='task to be done', default='train')
    opt = parser.parse_args("")
    opt.input_name = 'blank'
    opt = functions.post_config(opt)
    
    real = functions.np2torch(images[0], opt)
    functions.adjust_scales2image(real, opt)

    all_reals = []
    for image in images:
        reals = []
        real_ = functions.np2torch(image, opt)
        real = imresize(real_,opt.scale1,opt)
        reals = functions.creat_reals_pyramid(real,reals,opt)
        all_reals.append(reals)

    return np.array(all_reals).T

def train_model(input_name):
    parser = get_arguments()
    parser.add_argument('--input_dir', help='input image dir', default='Input/Images')
    parser.add_argument('--input_name', help='input image name', required=True)
    parser.add_argument('--mode', help='task to be done', default='train')
    opt = parser.parse_args("")
    opt = functions.post_config(opt)
    Gs = []
    Zs = []
    reals = []
    NoiseAmp = []
    dir2save = functions.generate_dir2save(opt)

    if (os.path.exists(dir2save)):
        print('trained model already exist')
    else:
        try:
            os.makedirs(dir2save)
        except OSError:
            pass
        real = functions.read_image(opt)
        functions.adjust_scales2image(real, opt)
        train(opt, Gs, Zs, reals, NoiseAmp)
        SinGAN_generate(Gs,Zs,reals,NoiseAmp,opt)
