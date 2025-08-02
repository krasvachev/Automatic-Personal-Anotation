import numpy as np

class PCA2D:

    def __init__(self, image_size = None):
        self.__data_init = False
        self.__computed = False
        self.__projected_mat = None

        if(image_size != None):
            self.__init_size(image_size)   

    def __init_size(self, matrix):
        self.__data_init = True
        self.__matrix_width = matrix.shape[2]
        self.__matrix_height = matrix.shape[1]
        self.__matrix_list = matrix

    def __calculate_cov_matrix(self):
        arr_mean = np.mean(self.__matrix_list, 0)
        self.__arr_mean = arr_mean

        self.__mat_mean_sub = self.__matrix_list[:,:,:] - arr_mean[np.newaxis,:,:]
        result_arr_w = np.zeros((self.__matrix_width, self.__matrix_width)) #cov matrix is width * width!, dim 1
        result_arr_h = np.zeros((self.__matrix_height, self.__matrix_height)) #cov matrix is height * height!, dim 0

        mat_count = self.__mat_mean_sub.shape[0]

        for x in range(mat_count):
            curr_arr = self.__mat_mean_sub[x,:,:]
            result_arr_w += np.dot(curr_arr.T, curr_arr)
            result_arr_h += np.dot(curr_arr, curr_arr.T)

        result_arr_w = result_arr_w / mat_count
        result_arr_h = result_arr_h / mat_count

        return result_arr_w, result_arr_h

    def __extract_eigenvectors(self, eigval, eigvec, percent):
        part_var = sum(eigval) * percent
        cur_sum = 0
        cur_ind = 0
        while (cur_ind < eigval.shape[0]):
            cur_sum += eigval[cur_ind]
            if(cur_sum >=  part_var):
                break
            cur_ind += 1

        res_eigval = eigval[0:cur_ind]
        res_eigvec = eigvec[:, 0:cur_ind]
        return res_eigval, res_eigvec

    def add_matrix(self, matrix):
        if(self.__data_init == False):
            self.__init_size(matrix)
            self.__data_init = True
        else:
            if(matrix.shape[1] != self.__matrix_width or matrix.shape[0] != self.__matrix_height):
                raise Exception("Inconsistent image dimensions!")
                
            self.__matrix_list = np.dstack((self.__matrix_list, matrix))

    def load_data_bulk(self, data):
        if(self.__data_init == False):
            self.__init_size(data)
            self.__data_init = True

    def compute_pca(self, percent_variance):
        if not(0.0 <= percent_variance <= 1.0):
            raise Exception("Percentage must be a floating point between 0 and 1")

        cov_mat_w, cov_mat_h = self.__calculate_cov_matrix() 
        w_w, v_w = np.linalg.eig(cov_mat_w)
        w_h, v_h = np.linalg.eig(cov_mat_h)

        idx_w = w_w.argsort()[::-1]
        idx_h = w_h.argsort()[::-1]

        w_w = w_w[idx_w]
        v_w = v_w[:,idx_w]

        w_h = w_h[idx_h]
        v_h = v_h[:,idx_h]

        self.__eigval_w, self.__eigvec_w  = self.__extract_eigenvectors(w_w, v_w, percent_variance)
        self.__eigval_h, self.__eigvec_h  = self.__extract_eigenvectors(w_h, v_h, percent_variance)
        self.__computed = True

    def get_projected_data(self):
        if(self.__computed):
            if(self.__projected_mat == None):
                Y_temp = np.einsum('ij,kil->kjl', self.__eigvec_h, self.__mat_mean_sub)
                self.__projected_mat = np.dot(Y_temp, self.__eigvec_w)
                
                return self.__projected_mat
        
        return None


    def project_data(self, data):
        if(self.__computed):
            sub_data = data[:,:,:] - self.__arr_mean[np.newaxis,:,:]
            Y_temp = np.einsum('ij,kil->kjl', self.__eigvec_h, sub_data)
            proj_data = np.dot(Y_temp, self.__eigvec_w)

            return proj_data
        
        return None

    def get_eigenvectors(self):
        if(self.__computed):
            return self.__eigvec_w, self.__eigvec_h
        return None
