
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

class DepthVisualizer:
    @staticmethod
    def plot_depth_map(rgb_img, depth_map, pred_depth, save_path=None):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(rgb_img)
        axes[0].set_title('RGB Image')
        axes[1].imshow(depth_map, cmap='plasma')
        axes[1].set_title('Ground Truth Depth')
        axes[2].imshow(pred_depth, cmap='plasma')
        axes[2].set_title('Predicted Depth')
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    @staticmethod
    def create_comparison_table(results_dict):
        df = pd.DataFrame(results_dict)
        return df.to_markdown()