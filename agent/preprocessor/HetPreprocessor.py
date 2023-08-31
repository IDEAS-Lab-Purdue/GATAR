import torch
import torch.nn as nn


class Preprocessor(nn.Module):
    # input_size: (B, C, H, W)
    # output_size: (B, C', H, W)

    
    def __init__(self, config=None):
        super(Preprocessor, self).__init__()
        if config is None:
            raise Exception("config is None")
        self.history=config['history']
        self.mul = config['mul']
        
        

    def forward(self, x):
        '''
        x: (B, C, H, W)
        output: (B, C', H, W)
        '''
        c_new=3
        if self.history:
            c_new+=1
        if self.mul:
            c_new+=1
        output = torch.zeros(x.shape[0], c_new, x.shape[2], x.shape[3], device=x.device)
        output=[]
        output.append(x[:,0,:,:])
        output.append(self.create_target_map_batch(x[:,2,:,:]))
        output.append(self.create_cost_map_batch(x[:,1,:,:]))


        if self.history:
            output.append(self.create_target_map_batch(x[:,3,:,:],sigma_x=0.5,sigma_y=0.5))
        if self.mul:
            output.append(output[1]*output[2])

        output=torch.stack(output,dim=1)
        
        return output
    
    def create_history_map(self, x):
        pass
    
    def create_cost_map_batch(self, bool_matrix, sigma_x=1, sigma_y=1):
        '''
        channel 1
        x: (B,  H, W)
        output: (B,  H, W)
        '''
        B, H, W = bool_matrix.shape

        # 找到所有目标的坐标，并考虑批次
        target_positions = torch.nonzero(bool_matrix>0)
        target_b, target_x, target_y = target_positions.split(1, dim=1)
        assert target_b.shape[0] == target_x.shape[0] == target_y.shape[0]
        N = target_b.shape[0]
        if N == 0:
            return bool_matrix
        
        # 生成坐标网格并进行适当的扩展以匹配目标位置
        x_coor = torch.arange(H).view(1, -1, 1).float().expand(N,H,1).to(bool_matrix.device)
        y_coor = torch.arange(W).view(1, 1, -1).float().expand(N,1,W).to(bool_matrix.device)

        dist_x = x_coor - target_x.unsqueeze(1).float()
        dist_y = y_coor - target_y.unsqueeze(1).float()

        gaussian_values = torch.exp(- (dist_x**2 / (2 * sigma_x**2) + dist_y**2 / (2 * sigma_y**2)))

        # 初始化一个全0矩阵来累加每个位置的高斯值
        batched_gaussians = torch.zeros(B, H, W, device=bool_matrix.device)

        for idx, (b, x, y) in enumerate(target_positions):
            batched_gaussians[b] += gaussian_values[idx]

        gaussian_matrix = batched_gaussians

        # 归一化
        max_values = gaussian_matrix.view(B, -1).max(dim=1, keepdim=True).values.view(B, 1, 1)
        #turn zero to one to avoid nan
        max_values[max_values==0]=1
        gaussian_matrix /= max_values


        return gaussian_matrix


    def create_extended_cost_map(self, observation):
        '''
        observation: (1, C, H, W)
        '''
        assert observation.shape[0] == 1
        ego_position = torch.nonzero(observation[0, 1, :, :] > 0)
        targeted_position = torch.nonzero(observation[0, 2, :, :] > 0)
        raise NotImplementedError
        
        
        print(ego_position.shape)
        
    
    def create_oc_map(self, x):
        raise NotImplementedError
        
    
 

    def create_target_map_batch(self, bool_matrix, sigma_x=1, sigma_y=1):
        '''
        x: (B,  H, W)
        output: (B,  H, W)
        '''
        
        B, H, W = bool_matrix.shape

        # 找到所有目标的坐标，并考虑批次
        target_positions = torch.nonzero(bool_matrix>0)
        target_b, target_x, target_y = target_positions.split(1, dim=1)
        assert target_b.shape[0] == target_x.shape[0] == target_y.shape[0]
        N = target_b.shape[0]
        if N == 0:
            return bool_matrix
        
        # 生成坐标网格并进行适当的扩展以匹配目标位置
        x_coor = torch.arange(H).view(1, -1, 1).float().expand(N,H,1).to(bool_matrix.device)
        y_coor = torch.arange(W).view(1, 1, -1).float().expand(N,1,W).to(bool_matrix.device)

        dist_x = x_coor - target_x.unsqueeze(1).float()
        dist_y = y_coor - target_y.unsqueeze(1).float()

        gaussian_values = torch.exp(- (dist_x**2 / (2 * sigma_x**2) + dist_y**2 / (2 * sigma_y**2)))

        # 初始化一个全0矩阵来累加每个位置的高斯值
        batched_gaussians = torch.zeros(B, H, W, device=bool_matrix.device)

        for idx, (b, x, y) in enumerate(target_positions):
            batched_gaussians[b] += gaussian_values[idx]

        gaussian_matrix = batched_gaussians

        # 归一化
        max_values = gaussian_matrix.view(B, -1).max(dim=1, keepdim=True).values.view(B, 1, 1)
        #turn zero to one to avoid nan
        max_values[max_values==0]=1
        gaussian_matrix /= max_values

        return gaussian_matrix


#testing code

# if __name__ == '__main__':

#     preprocessor=Preprocessor()
#     bool_matrix = torch.zeros(2,10, 10)
#     bool_matrix[0, 2, 3] = 1
#     bool_matrix[0, 5, 5] = 1
#     bool_matrix[1, 7, 7] = 1
#     bool_matrix[1, 8, 8] = 1
#     import matplotlib.pyplot as plt
#     fig=plt.figure()
#     #add subplot for each agent
#     for i in range(bool_matrix.shape[0]):
#         plt.subplot(1,bool_matrix.shape[0],i+1)
#         plt.imshow(bool_matrix[i,:,:])
#     plt.savefig('bool_matrix.png')

#     gaussian_matrix = preprocessor.create_target_map_batch(bool_matrix)
#     fig=plt.figure()
#     #add subplot for each agent
#     for i in range(bool_matrix.shape[0]):
#         plt.subplot(1,bool_matrix.shape[0],i+1)
#         plt.imshow(gaussian_matrix[i,:,:])
#     plt.savefig('gaussian_matrix.png')



