import os
import torch
import paddle
import pickle
import numpy as np
from paddle_swinir import SwinIR as SwinIR_P
from torch_swinir import SwinIR as SwinIR_T

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def save(state_dicts, file_name):
    def convert(state_dict):
        model_dict = {}

        for k, v in state_dict.items():
            if isinstance(
                    v,
                (paddle.static.Variable, paddle.Tensor)):
                model_dict[k] = v.numpy()
            else:
                model_dict[k] = v

        return model_dict

    final_dict = {}
    for k, v in state_dicts.items():
        if isinstance(
                v,
            (paddle.static.Variable, paddle.Tensor)):
            final_dict = convert(state_dicts)
            break
        elif isinstance(v, dict):
            final_dict[k] = convert(v)
        else:
            final_dict[k] = v

    with open(file_name, 'wb') as f:
        pickle.dump(final_dict, f, protocol=2)
        

if __name__ == '__main__':
    
    '''
    pretrained_model = 'iter_400000_weight.pdparams'
    para_state_dict = paddle.load(pretrained_model)
    keys = para_state_dict['generator'].keys()
    for i, key in enumerate(keys):
        print(key)
    '''

    
    print('========================== TORCH ==========================')               
    torch_net = SwinIR_T(upscale=1, in_chans=3, img_size=128, window_size=8,
                   img_range=1., depths=[ 6, 6, 6, 6, 6, 6 ], embed_dim=180, num_heads=[ 6, 6, 6, 6, 6, 6 ],
                   mlp_ratio=2, upsampler='', resi_connection='1conv')
   
    torch_chechpoint = torch.load("model_zoo/swinir/005_colorDN_DFWB_s128w8_SwinIR-M_noise15.pth")
    torch_net.load_state_dict(torch_chechpoint['params'], strict=True)
    torch_net.eval()
    torch_net = torch_net.cuda()
    
    torch_model_state_dict = torch_net.state_dict()
    keys = torch_model_state_dict.keys()
    
    paddle_weights_dic = {}
    for i, k in enumerate(keys):
        # print(i, k, torch_model_state_dict[k].shape, torch_model_state_dict[k].ndim)
        value = torch_model_state_dict[k].cpu().numpy()
        if torch_model_state_dict[k].ndim == 2 and 'weight' in k:
            value = value.transpose(1, 0)
        paddle_weights_dic[k] = value
        
        
    
    print('========================== PADDLE ==========================') 
    paddle_net = SwinIR_P(upscale=1, in_chans=3, img_size=128, window_size=8,
                   img_range=1., depths=[ 6, 6, 6, 6, 6, 6 ], embed_dim=180, num_heads=[ 6, 6, 6, 6, 6, 6 ],
                   mlp_ratio=2, upsampler='', resi_connection='1conv')
    '''  
    paddle_model_state_dict = paddle_net.state_dict()
    keys = paddle_model_state_dict.keys()
    for i, k in enumerate(keys):
        print(i, k, paddle_model_state_dict[k].shape, paddle_model_state_dict[k].ndim)
    '''
    
    paddle_net.set_dict(paddle_weights_dic)
    paddle_net.eval()
    # paddle.save(paddle_net.state_dict(), './model/paddle_model.pdparams')
    state_dicts = {}
    state_dicts['params'] = paddle_net.state_dict()
    save(state_dicts, './model_zoo/paddle/005_colorDN_DFWB_s128w8_SwinIR-M_noise15.pdparams')
    
    print('========================== TEST ==========================') 
    input_torch = torch.rand(1, 3, 128, 128).cuda()
    input_paddle = paddle.to_tensor(input_torch.cpu().numpy())
    output_torch = torch_net(input_torch).cpu().detach().numpy()
    output_paddle = paddle_net(input_paddle).numpy()
    print(np.mean(np.abs(output_torch - output_paddle)))
    
    