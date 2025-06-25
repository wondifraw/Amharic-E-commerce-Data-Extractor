import matplotlib.pyplot as plt


def comparing_times(models, times):


    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(models, times, color=['blue', 'green', 'orange'])

    ax.set_title('Model Training Time Comparison', fontsize=16)
    ax.set_ylabel('Training Time (minutes)', fontsize=12)
    ax.set_xlabel('Models', fontsize=12)


    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  
                    textcoords="offset points",
                    ha='center', va='bottom')


    plt.tight_layout()
    plt.show()