# # import matplotlib.pyplot as plt
# # import numpy as np
# # from collections import defaultdict

# # class TrainingVisualizer:
# #     def __init__(self):
# #         self.history = defaultdict(list)
# #         self.steps = []
# #         self.current_step = 0
        
# #         # Set up the plot style
# #         plt.style.use('seaborn')
        
# #     def update(self, log_stats):
# #         """Update history with new statistics"""
# #         self.current_step += 1
# #         self.steps.append(self.current_step)
        
# #         for key, value in log_stats.items():
# #             self.history[key].append(value)
    
# #     def plot(self, save_path=None):
# #         """Plot all training metrics"""
# #         metrics = ["tot_loss", "rl_loss", "pt_loss", "pg_loss", "reg_loss", 
# #                   "reward", "rev_kl", "stu_lens", "mixed_lens"]

# #         # Create a 3x3 grid of subplots
# #         fig, axes = plt.subplots(3, 3, figsize=(15, 15))
# #         fig.suptitle('Training Metrics Over Time', fontsize=16)
        
# #         # Flatten axes for easier iteration
# #         axes = axes.flatten()
        
# #         for idx, metric in enumerate(metrics):
# #             ax = axes[idx]
# #             if metric in self.history:
# #                 ax.plot(self.steps, self.history[metric], 
# #                        label=metric, linewidth=2)
# #                 ax.set_title(metric)
# #                 ax.set_xlabel('Steps')
# #                 ax.grid(True)
                
# #                 # Add moving average
# #                 window = min(50, len(self.history[metric]))
# #                 if window > 1:
# #                     moving_avg = np.convolve(self.history[metric], 
# #                                            np.ones(window)/window, 
# #                                            mode='valid')
# #                     ax.plot(self.steps[window-1:], moving_avg, 
# #                           '--', label=f'{window}-step MA', 
# #                           alpha=0.7)
                
# #                 ax.legend()
        
# #         plt.tight_layout()
        
# #         if save_path:
# #             plt.savefig(save_path)
# #         plt.show()


# import matplotlib.pyplot as plt
# import numpy as np
# from collections import defaultdict
# import os

# class TrainingVisualizer:
#     def __init__(self, log_dir='training_plots'):
#         self.history = defaultdict(list)
#         self.steps = []
#         self.current_step = 0
#         self.log_dir = log_dir
        
#         # Create log directory if it doesn't exist
#         os.makedirs(self.log_dir, exist_ok=True)
        
#         # Set up the plot style
#         plt.style.use('bmh')
#         print("TrainingVisualizer initialized. Plots will be saved to:", self.log_dir)
        
#     def update(self, log_stats):
#         """Update history with new statistics"""
#         self.current_step += 1
#         self.steps.append(self.current_step)
        
#         for key, value in log_stats.items():
#             if isinstance(value, (int, float)):  # Only store numeric values
#                 self.history[key].append(value)
        
#         print(f"Step {self.current_step}: Updated with {len(log_stats)} metrics")
#         print("Current metrics:", list(self.history.keys()))
    
#     def plot(self, save_path=None):
#         """Plot all training metrics"""
#         if not self.history:
#             print("No data to plot yet!")
#             return
            
#         metrics = ["tot_loss", "rl_loss", "pt_loss", "pg_loss", "reg_loss", 
#                   "reward", "rev_kl", "stu_lens", "mixed_lens"]
        
#         print(f"Plotting {len(metrics)} metrics...")
        
#         # Create a 3x3 grid of subplots
#         fig, axes = plt.subplots(3, 3, figsize=(15, 15))
#         fig.suptitle('Training Metrics Over Time', fontsize=16)
        
#         # Flatten axes for easier iteration
#         axes = axes.flatten()
        
#         plotted_metrics = 0
#         for idx, metric in enumerate(metrics):
#             ax = axes[idx]
#             if metric in self.history and len(self.history[metric]) > 0:
#                 ax.plot(self.steps, self.history[metric], 
#                        label=metric, linewidth=2)
#                 ax.set_title(metric)
#                 ax.set_xlabel('Steps')
#                 ax.grid(True)
                
#                 # Add moving average
#                 window = min(50, len(self.history[metric]))
#                 if window > 1:
#                     moving_avg = np.convolve(self.history[metric], 
#                                            np.ones(window)/window, 
#                                            mode='valid')
#                     ax.plot(self.steps[window-1:], moving_avg, 
#                           '--', label=f'{window}-step MA', 
#                           alpha=0.7)
                
#                 ax.legend()
#                 plotted_metrics += 1
#             else:
#                 print(f"No data available for metric: {metric}")
        
#         print(f"Successfully plotted {plotted_metrics} metrics")
        
#         plt.tight_layout()
        
#         if save_path:
#             try:
#                 plt.savefig(save_path)
#                 print(f"Plot saved to: {save_path}")
#             except Exception as e:
#                 print(f"Error saving plot: {e}")
        
#         plt.show()
#         plt.close()  # Clean up


import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import os

class TrainingVisualizer:
    def __init__(self, log_dir='training_plots'):
        self.history = defaultdict(list)
        self.steps = []
        self.current_step = 0
        self.log_dir = log_dir
        
        # Create log directory if it doesn't exist
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Set up the plot style using a built-in style
        plt.style.use('bmh')  # Using 'bmh' style which is included with matplotlib
        print("TrainingVisualizer initialized. Plots will be saved to:", self.log_dir)
        
    def update(self, log_stats):
        """Update history with new statistics"""
        self.current_step += 1
        self.steps.append(self.current_step)
        
        for key, value in log_stats.items():
            if isinstance(value, (int, float)):  # Only store numeric values
                self.history[key].append(value)
        
        print(f"Step {self.current_step}: Updated with {len(log_stats)} metrics")
        print("Current metrics:", list(self.history.keys()))
    
    def plot(self, save_path=None):
        """Plot all training metrics"""
        if not self.history:
            print("No data to plot yet!")
            return
        
        # Core metrics that we always want to plot in a specific order
        core_metrics = ["tot_loss", "rl_loss", "pt_loss", "pg_loss", "reg_loss", 
                       "reward", "rev_kl", "stu_lens", "mixed_lens", "lm_loss", "ds_loss"]
        
        # Add any additional metrics from history that aren't in core_metrics
        all_metrics = list(set(core_metrics) | set(self.history.keys()))
        
        print(f"Plotting {len(all_metrics)} metrics...")
        
        # Calculate number of subplots needed
        n_metrics = len(all_metrics)
        n_cols = 3
        n_rows = (n_metrics + n_cols - 1) // n_cols  # Ceiling division
        
        # Create a grid of subplots with the right dimensions
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        fig.suptitle('Training Metrics Over Time', fontsize=16)
        
        # Ensure axes is always a 2D array
        if n_rows == 1:
            axes = np.array([axes])
        axes = np.array(axes).reshape(n_rows, -1)
        
        plotted_metrics = 0
        for idx, metric in enumerate(all_metrics):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]
            
            if metric in self.history and len(self.history[metric]) > 0:
                ax.plot(self.steps, self.history[metric], 
                       label=metric, linewidth=2)
                ax.set_title(metric)
                ax.set_xlabel('Steps')
                ax.grid(True)
                
                # Add moving average
                window = min(50, len(self.history[metric]))
                if window > 1:
                    moving_avg = np.convolve(self.history[metric], 
                                           np.ones(window)/window, 
                                           mode='valid')
                    ax.plot(self.steps[window-1:], moving_avg, 
                          '--', label=f'{window}-step MA', 
                          alpha=0.7)
                
                ax.legend()
                plotted_metrics += 1
            else:
                print(f"No data available for metric: {metric}")
                ax.text(0.5, 0.5, f'No data for {metric}', 
                       horizontalalignment='center',
                       verticalalignment='center',
                       transform=ax.transAxes)
        
        # Hide any unused subplots
        for idx in range(len(all_metrics), n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].set_visible(False)
        
        print(f"Successfully plotted {plotted_metrics} metrics")
        
        plt.tight_layout()
        
        if save_path:
            try:
                # Create directory if it doesn't exist
                save_dir = os.path.dirname(save_path)
                if save_dir:
                    os.makedirs(save_dir, exist_ok=True)
                
                plt.savefig(save_path)
                print(f"Plot saved to: {save_path}")
            except Exception as e:
                print(f"Error saving plot: {e}")
        
        plt.show()
        plt.close()  # Clean up