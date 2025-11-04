from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.experiment.data_config import DATA_CONFIG_MAP
import matplotlib.pyplot as plt


DATASET_PATH = "./datasets/gr1_arms_only.Nut_pouring_task"
EMBODIMENT_TAG = "gr1"
#VIDEO_BACKEND = "decord"
VIDEO_BACKEND = "torchvision_av"
data_config = DATA_CONFIG_MAP["fourier_gr1_arms_only"]


def main():
    modality_config = data_config.modality_config()
    dataset = LeRobotSingleDataset(
        dataset_path=DATASET_PATH,
        modality_configs=modality_config,
        video_backend=VIDEO_BACKEND,
        video_backend_kwargs=None,
        transforms=None, 
        embodiment_tag=EMBODIMENT_TAG,
    )

    traj_id = 0
    max_steps = 150

    images = []
    sample_images = 6

    for step_count in range(max_steps):
        data_point = dataset.get_step_data(traj_id, step_count)
        if step_count % (max_steps // sample_images) == 0:
            image = data_point["video.ego_view"][0]
            images.append(image)


    # Plot the images in a row
    fig, axes = plt.subplots(nrows=1, ncols=sample_images, figsize=(16, 4))

    for i, ax in enumerate(axes):
        ax.imshow(images[i])
        ax.axis("off")
        
        
    # Save to file
    plt.savefig("dataset_images.png", bbox_inches="tight", dpi=300)
    plt.close(fig) 
    print("Images saved as: dataset_images.png")
        
        
if __name__ == "__main__":
    main()