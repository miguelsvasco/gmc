import os
import umap
import torch
import pickle
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.manifold import TSNE

CB_color_cycle = [
    "#377eb8",
    "#ff7f00",
    "#4daf4a",
    "#f781bf",
    "#a65628",
    "#984ea3",
    "#999999",
    "#e41a1c",
    "#dede00",
]


def plot_DCA_components_with_UMAP(
    experiment_relative_path: str, experiment_id: str,
):
    # e.g. evaluation/gmc_mhd/log_4/id_14/results_dca_evaluation/, joint_m0

    experiment_logs_dir = os.path.join(experiment_relative_path, experiment_id, "logs")

    with open(
        os.path.join(experiment_relative_path, experiment_id, "DCA/network_stats.pkl"),
        "rb",
    ) as f:
        network_stats = pickle.load(f)

    input_array = np.load(os.path.join(experiment_logs_dir, "input_array.npy"))
    input_array_comp_labels = np.load(
        os.path.join(experiment_logs_dir, "input_array_comp_labels.npy")
    )

    modality1, modality2 = experiment_id.split("_")
    reducer = umap.UMAP(random_state=42)
    reducer.fit(input_array)
    embedding = reducer.transform(input_array)

    # discard outliers from visualization
    no_outliers_idx = np.where(
        input_array_comp_labels <= network_stats.first_trivial_component_idx
    )[0]

    R_idx = no_outliers_idx[np.where(no_outliers_idx < network_stats.num_R)[0]]
    E_idx = no_outliers_idx[np.where(no_outliers_idx >= network_stats.num_R)[0]]
    plt.clf()
    plt.scatter(
        embedding[R_idx, 0],
        embedding[R_idx, 1],
        color=CB_color_cycle[0],
        s=50,
        marker="d",
        label=f" ",
        zorder=1,
    )
    plt.scatter(
        embedding[E_idx, 0],
        embedding[E_idx, 1],
        color=CB_color_cycle[1],
        s=35,
        marker=".",
        label=f" ",
        alpha=0.7,
    )
    plt.title(f"blue = {modality1}- orange = {modality2}")
    plt.legend(frameon=False)
    lgnd = plt.legend(frameon=False, fontsize=15)
    lgnd.legendHandles[0]._sizes = [100]
    lgnd.legendHandles[1]._sizes = [100]

    model = experiment_relative_path.split("/")[1]
    name = f"ICML_{model}_{experiment_id}_no_outliers"
    plt.axis("off")
    plt.savefig(
        os.path.join(experiment_relative_path, experiment_id, "visualization", name),
        dpi=100,
    )


def viz_latent_space_mnist_tsne(
    model_name, scenario, model, dataloader, mods, num_points, logger, results_dir
):

    model.eval()

    if mods == [0, 1]:
        obs_type = "joint"
    elif mods == [0]:
        obs_type = "image"
    elif mods == [1]:
        obs_type = "label"
    else:
        raise ValueError("Incorrect mod list selected")

    # Data collection for samples
    with torch.no_grad():

        latent_samples = []
        labels = []
        for batch_idx, data in enumerate(tqdm(dataloader)):

            input_data = [
                data[0],
                torch.nn.functional.one_hot(data[1], num_classes=10).float(),
            ]
            label = data[1].numpy()

            # Drop modalities (if required)
            model_data = []
            if mods is not None:
                for j in range(len(input_data)):
                    if j not in mods:
                        model_data.append(None)
                    else:
                        model_data.append(input_data[j])
            else:
                model_data = input_data

            # Forward Pass
            latent = model.encode(model_data)

            for sample, lab in zip(latent, label):
                latent_samples.append(sample.cpu().numpy())
                labels.append(lab)

            # Stupid way to break the loader
            if len(labels) >= num_points:
                num_points = len(labels)
                break

    # Process Data
    latent_samples = np.array(latent_samples)
    latent_samples = latent_samples.reshape(num_points, model.latent_dim)

    # Fit TSNE
    latent_2dim = TSNE(n_components=2).fit_transform(latent_samples)

    # Plot latent space
    plt.style.use("ggplot")
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    cmap = plt.cm.tab20b
    cmaplist = [cmap(i) for i in range(cmap.N)]
    bounds = np.linspace(0, 10, 11)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    scat = ax.scatter(
        latent_2dim[:, 0], latent_2dim[:, 1], c=labels, cmap=cmap, norm=norm
    )
    cb = plt.colorbar(scat, ticks=[i for i in range(10)])
    cb.set_label("Labels")
    ax.set_title(f"TSNE plot for {model_name} in {scenario} - colour coded by Labels")

    plt.savefig(
        os.path.join(
            results_dir, f"tsne_{model_name}_{scenario}_N{num_points}_{obs_type}.png"
        )
    )

    logger.log_artifact(
        filepath=os.path.join(
            results_dir, f"tsne_{model_name}_{scenario}_N{num_points}_{obs_type}.png"
        ),
        name="tsne_"
        + str(model_name)
        + "_"
        + str(scenario)
        + "_N"
        + str(num_points)
        + "_"
        + str(obs_type)
        + ".png",
    )

    return
