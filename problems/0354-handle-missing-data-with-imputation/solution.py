import numpy as np
from scipy import stats



def impute_missing_data(data: np.ndarray, strategy: str = 'mean') -> np.ndarray:
    """
    Impute missing values in a 2D array using the specified strategy.
    
    Args:
        data: 2D numpy array with missing values represented as np.nan
        strategy: Imputation strategy - 'mean', 'median', or 'mode'
        
    Returns:
        2D numpy array with missing values imputed
    """
    # missing_values_idx = []
    # existing_values = []
    # if strategy == 'mean':
    #     for i in range(len(data[0])):
    #         col = data[:, i]
    #         if np.isnan(col).any():
    #             for j in range(len(col)):
    #                 if np.isnan(col[j]):
    #                     missing_values_idx.append([i,j])
    #                 else:
    #                     existing_values.append({i,col[j]})
    #     means = {}
    #     for idx, value in existing_values:
    #         means[idx] = means.get(idx, 0) + value  
    #     print(means, "means")


    #     # mean = np.mean(existing_values)
    #             # print(data[:, i])
    #     print("existying", existing_values)   
    #     print("missing", missing_values_idx)
    #     return data
    
    if strategy == 'mean':
        missing_ele = np.nanmean(data, axis=0)
        for ele in range(data.shape[1]):
            for sub_ele in range(len(data[:, ele])):
                if np.isnan(data[:, ele][sub_ele]):
                    data[sub_ele][ele] = missing_ele[ele]
        return data


    elif strategy == 'median':
        missing_ele = np.nanmedian(data, axis=0)
        for ele in range(data.shape[1]):
            for sub_ele in range(len(data[:, ele])):
                if np.isnan(data[:, ele][sub_ele]):
                    data[sub_ele][ele] = missing_ele[ele]
        return data
    
    elif strategy == 'mode':
        result = stats.mode(data, axis=0, nan_policy='omit')
        for ele in range(data.shape[1]):
            for sub_ele in range(len(data[:, ele])):
                if np.isnan(data[:, ele][sub_ele]):
                    data[sub_ele][ele] = result.mode[ele]
        return data
      