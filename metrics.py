import numpy as np

class Metrics:
    def __init__(self):
        self.data = {
            t: {
                'distance': {'ukf': [], 'pf': [], 'actual': []},
                'velocity': {'ukf': [], 'pf': [], 'actual': []}
            } for t in ['t1','t2','t3']
        }

    def update(self, target, actual, ukf_est, pf_est):
        for key in ['distance','velocity']:
            self.data[target][key]['actual'].append(actual[key])
            self.data[target][key]['ukf'].append(ukf_est[key])
            self.data[target][key]['pf'].append(pf_est[key])

    def compute_rmse(self, y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        return np.sqrt(np.mean((y_pred - y_true)**2)) # Rumus RMSE

    def compute_mae(self, y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        return np.mean(np.abs(y_pred - y_true)) # Rumus MAE

    def compute_mbe(self, y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        return np.mean(y_pred - y_true) # Rumus MBE

    def get_metrics(self):
        result = {}
        for t in ['t1','t2','t3']:
            result[t] = {}
            for key in ['distance','velocity']:
                actual = self.data[t][key]['actual']
                ukf_est = self.data[t][key]['ukf']
                pf_est = self.data[t][key]['pf']
                
                if len(actual) == 0:
                    continue
                
                # RMSE
                result[t][f'rmse_{key}_ukf'] = self.compute_rmse(actual, ukf_est)
                result[t][f'rmse_{key}_pf']  = self.compute_rmse(actual, pf_est)
                
                # MAE
                result[t][f'mae_{key}_ukf'] = self.compute_mae(actual, ukf_est)
                result[t][f'mae_{key}_pf']  = self.compute_mae(actual, pf_est)
                
                # MBE
                result[t][f'mbe_{key}_ukf']  = self.compute_mbe(actual, ukf_est)
                result[t][f'mbe_{key}_pf']   = self.compute_mbe(actual, pf_est)
        
        return result

    def reset(self, target):
        for key in ['distance', 'velocity']:
            self.data[target][key]['actual'].clear()
            self.data[target][key]['ukf'].clear()
            self.data[target][key]['pf'].clear()